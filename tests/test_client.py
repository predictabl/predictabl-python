"""Unit tests for the client against an in-memory fake API (no network)."""

from __future__ import annotations

import io
import json
from datetime import date, datetime

import httpx
import pytest

import predictabl as pb
from predictabl import errors
from predictabl._http import HOSTED_URL, Transport
from predictabl._progress import Progress

from .fake_api import FakeApi, err, ok

BASE = "http://test/api/v1"


def make(api: FakeApi | None = None, **kw) -> tuple[pb.Client, FakeApi]:
    api = api or FakeApi()
    http = httpx.Client(transport=api.transport())
    c = pb.Client(api_key="pk_test", base_url=BASE, poll_interval=0, _http=http, **kw)
    return c, api


def responder(*responses):
    """A transport that answers with the given responses in order."""
    queue = list(responses)
    seen = []

    def handle(req):
        seen.append(req)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return httpx.MockTransport(handle), seen


# ----------------------------------------------------------- configuration


def test_defaults_to_the_hosted_app_and_reads_the_env(monkeypatch):
    monkeypatch.delenv("PREDICTABL_BASE_URL", raising=False)
    monkeypatch.setenv("PREDICTABL_API_KEY", "pk_env")
    c = pb.Client()
    assert c.base_url == HOSTED_URL and c.api_key == "pk_env"
    monkeypatch.setenv("PREDICTABL_BASE_URL", "http://localhost:8000/api/v1/")
    assert pb.Client().base_url == "http://localhost:8000/api/v1"


def test_sends_the_key_and_identifies_the_client():
    c, api = make()
    c.account()
    _, _, _, headers = api.calls[-1]
    assert headers["authorization"] == "Bearer pk_test"
    assert headers["user-agent"].startswith("predictabl-python/")


def test_web_url_follows_the_api_url():
    assert pb.Client(base_url=HOSTED_URL).web_url == "https://app.predictabl.tech"
    assert (
        pb.Client(base_url="http://localhost:8000/api/v1").web_url
        == "http://localhost:3000"
    )


# ------------------------------------------------------------------ errors


@pytest.mark.parametrize(
    "response,cls,code",
    [
        (
            httpx.Response(401, json={"detail": "invalid api token"}),
            errors.AuthenticationError,
            None,
        ),
        (
            err("BUILD_LIMIT_REACHED", "Free plan: 2 of 2 used.", 402),
            errors.LimitReached,
            "BUILD_LIMIT_REACHED",
        ),
        (
            httpx.Response(403, json={"detail": "scope 'inference' does not allow POST"}),
            errors.PermissionDenied,
            None,
        ),
        (err("NOT_FOUND", "Run not found", 404), errors.NotFound, "NOT_FOUND"),
        (
            err("PREDICT_UNAVAILABLE", "no time axis", 422),
            errors.InvalidRequest,
            "PREDICT_UNAVAILABLE",
        ),
        (httpx.Response(500, text="boom"), errors.ApiError, None),
    ],
)
def test_refusals_become_typed_errors_with_the_servers_message(response, cls, code):
    transport, _ = responder(response)
    t = Transport("pk", BASE, http=httpx.Client(transport=transport))
    with pytest.raises(cls) as exc:
        t.post("/x")
    assert exc.value.code == code
    assert exc.value.message  # never empty
    assert isinstance(exc.value, errors.PredictablError)


def test_a_401_says_what_to_do_with_or_without_a_key():
    refused = httpx.Response(401, json={"detail": "unauthenticated"})
    transport, _ = responder(refused)
    t = Transport(None, BASE, http=httpx.Client(transport=transport))
    with pytest.raises(errors.AuthenticationError) as exc:
        t.get("/account")
    assert "No API key" in str(exc.value) and "PREDICTABL_API_KEY" in str(exc.value)

    transport, _ = responder(refused)
    t = Transport("pk_wrong", BASE, http=httpx.Client(transport=transport))
    with pytest.raises(errors.AuthenticationError) as exc:
        t.get("/account")
    assert "refused" in str(exc.value) and "revoked" in str(exc.value)
    assert exc.value.status == 401


def test_reads_are_retried_on_a_gateway_error_but_writes_are_not(monkeypatch):
    import predictabl._http as h

    monkeypatch.setattr(h.time, "sleep", lambda s: None)  # no real waiting
    transport, seen = responder(httpx.Response(503), ok({"fine": True}))
    t = Transport("pk", BASE, http=httpx.Client(transport=transport))
    assert t.get("/x") == {"fine": True}
    assert len(seen) == 2

    transport, seen = responder(httpx.Response(503), ok({"fine": True}))
    t = Transport("pk", BASE, http=httpx.Client(transport=transport))
    with pytest.raises(errors.ApiError):
        t.post("/x")
    assert len(seen) == 1, "a POST must never be silently repeated"


# -------------------------------------------------------------- build flow


def test_connect_waits_until_the_data_is_read():
    c, api = make()
    data = c.connect(_tmp_csv(), quiet=True)
    assert data.status == "ready" and data.table_count == 1
    method, path, _, headers = api.calls[0]
    assert (method, path) == ("POST", "/connections/from-csv")
    assert headers["content-type"].startswith("multipart/form-data")


def test_connect_accepts_a_dataframe():
    pd = pytest.importorskip("pandas")
    c, api = make()
    c.connect(pd.DataFrame({"a": [1, 2]}), name="frame", quiet=True)
    assert api.calls[0][1] == "/connections/from-csv"


def test_connect_of_a_missing_file_fails_before_any_request():
    c, api = make()
    with pytest.raises(errors.InvalidRequest):
        c.connect("/no/such/file.csv", quiet=True)
    assert api.calls == []


def test_question_to_model_in_one_call_with_progress():
    c, api = make()
    updates = []
    data = c.connect(_tmp_csv(), quiet=True)
    model = data.predict(
        "Will an order arrive late?", quiet=True, on_progress=updates.append
    )
    assert model.ready and model.id == "r1"
    assert model.meaning == "probability that the order is late"
    steps = [(u.step, u.status) for u in updates]
    for step in ("Understanding", "Planning", "Training", "Final model"):
        assert (step, "done") in steps, f"{step} never finished: {steps}"
    # Waited for OUR question job, not the older finished one.
    assert ("Understanding", "done") in steps
    assert any(u.step == "Final model" and "auc 0.8712" in u.detail for u in updates)
    runs = [b for (m, p, b, _) in api.calls if (m, p) == ("POST", "/runs")]
    assert runs == [{"spec_id": "s1", "preset": "full_quality", "engine": "auto"}]


def test_an_older_finished_job_is_not_mistaken_for_ours():
    """The first job poll returns an earlier, finished job; the client must
    keep waiting for the job its own request created."""
    c, api = make(FakeApi(older_job_done=True))
    data = c.connect(_tmp_csv(), quiet=True)
    plan = data.plan("late?", quiet=True)
    job_polls = [p for (m, p, _, _) in api.calls if p.endswith("/opportunities/job")]
    assert len(job_polls) >= 3
    assert plan.id == "s1"


def test_plan_shows_target_features_and_exclusions():
    c, _ = make()
    plan = c.connect(_tmp_csv(), quiet=True).plan("late?", quiet=True)
    assert plan.features == ["f1", "f3"]  # disabled features are not listed
    text = str(plan)
    assert "Order delivered after promised date" in text
    assert "Excluded delivered_at: dated after the outcome" in text


def test_refine_returns_the_next_version_and_leaves_the_old_one():
    c, _ = make()
    plan = c.connect(_tmp_csv(), quiet=True).plan("late?", quiet=True)
    newer = plan.refine("drop courier", quiet=True)
    assert (plan.version, newer.version) == (1, 2)


def test_a_failed_plan_raises_with_the_servers_reason():
    c, _ = make(FakeApi(fail_plan=True, polls_until_spec=1))
    data = c.connect(_tmp_csv(), quiet=True)
    with pytest.raises(errors.BuildFailed, match="bad SQL"):
        data.plan("late?", quiet=True)


def test_a_failed_run_raises_with_code_and_suggestions():
    c, _ = make(FakeApi(fail_run=True))
    data = c.connect(_tmp_csv(), quiet=True)
    with pytest.raises(errors.BuildFailed) as exc:
        data.predict("late?", quiet=True)
    assert exc.value.code == "BUILD_LIMIT_REACHED"
    assert "add_own_key" in exc.value.suggestions


def test_report_summarises_the_tested_model():
    c, _ = make()
    rep = c.model("r1").report()
    assert (rep["metric"], rep["score"], rep["score_std"]) == ("auc", 0.8712, 0.004)
    assert rep["top_features"] == ["days_late_before"]
    assert rep["url"] == "http://test/runs/r1"


def test_report_reads_the_metric_object_current_servers_send():
    # 0.1.1 crashed here ("unhashable type: 'dict'"), so every train() and
    # predict() that waited for the model failed at the very end.
    c, _ = make(FakeApi(primary_metric={"name": "auc", "value": 0.86}))
    rep = c.model("r1").report()
    assert (rep["metric"], rep["score"], rep["test_score"]) == ("auc", 0.8712, 0.86)


# ---------------------------------------------------------------- scoring


def test_rows_are_sent_in_batches_the_server_accepts():
    c, api = make()
    rows = [{"x": i} for i in range(2500)]
    preds = c.model("r1").predict(rows=rows)
    sizes = [
        len(b["rows"]) for (m, p, b, _) in api.calls if p.endswith("/predict-records")
    ]
    assert sizes == [1000, 1000, 500]
    assert len(preds) == preds.count == 2500
    assert preds.meaning == "probability that the order is late"


def test_one_dict_dates_and_numpy_values_are_sent_as_json():
    np = pytest.importorskip("numpy")
    c, api = make()
    c.model("r1").predict(
        rows={
            "when": datetime(2026, 1, 2, 3, 4),
            "day": date(2026, 1, 2),
            "n": np.int64(3),
        }
    )
    body = [b for (m, p, b, _) in api.calls if p.endswith("/predict-records")][0]
    assert body["rows"] == [{"when": "2026-01-02T03:04:00", "day": "2026-01-02", "n": 3}]


def test_ids_match_exactly_not_as_substrings():
    c, _ = make()
    preds = c.model("r1").predict(ids=["1"])
    assert [p["entity_id"] for p in preds] == ["1"], "'1' must not match '10' or '11'"


def test_window_and_limit_map_to_the_api():
    c, api = make()
    c.model("r1").predict(
        since=date(2026, 1, 1),
        until="2026-01-31",
        since_days=7,
        as_of=date(2026, 2, 1),
        limit=99999,
        filters=[{"column": "city", "op": "eq", "value": "Milano"}],
        include=["city"],
    )
    body = [b for (m, p, b, _) in api.calls if p.endswith("/predict")][0]
    assert body["since"] == "2026-01-01" and body["until"] == "2026-01-31"
    assert body["since_days"] == 7 and body["cutoff"] == "2026-02-01"
    assert body["top"] == 2000, "limit is capped at the server's maximum"
    assert body["filters"][0]["value"] == "Milano" and body["include"] == ["city"]


def test_predictions_to_pandas_flattens_context():
    pytest.importorskip("pandas")
    c, _ = make()
    df = c.model("r1").predict().to_pandas()
    assert {"entity_id", "prediction", "city"} <= set(df.columns)


# --------------------------------------------------------------- progress


def test_progress_outside_a_terminal_prints_each_change_once():
    out = io.StringIO()
    p = Progress(stream=out)
    p.running("Training", "tree 10")
    p.running("Training", "tree 10")  # a poll with nothing new
    p.running("Training", "tree 20")
    p.done("Training", "kept 20 trees")
    lines = out.getvalue().splitlines()
    assert lines == [
        "… Training       tree 10",
        "… Training       tree 20",
        "✓ Training       kept 20 trees",
    ]


def test_quiet_prints_nothing_but_still_reports():
    out, got = io.StringIO(), []
    p = Progress(quiet=True, on_update=got.append, stream=out)
    p.running("Planning")
    p.done()
    assert out.getvalue() == "" and [u.status for u in got] == ["running", "done"]


def test_a_broken_callback_does_not_break_the_build():
    p = Progress(quiet=True, on_update=lambda u: 1 / 0)
    p.running("Planning")
    p.done()


# -------------------------------------------------------------------- CLI


def test_cli_whoami_and_a_clean_error(monkeypatch, capsys):
    from predictabl import __main__ as cli

    c, _ = make()
    monkeypatch.setattr(cli, "Client", lambda **kw: c)
    assert cli.main(["whoami"]) == 0
    assert json.loads(capsys.readouterr().out)["email"] == "a@b.c"

    class Broken:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            raise errors.AuthenticationError("Missing or invalid API key.")

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(cli, "Client", Broken)
    assert cli.main(["whoami"]) == 1
    assert "invalid API key" in capsys.readouterr().err


# ----------------------------------------------------------------- helpers


def _tmp_csv():
    import tempfile
    from pathlib import Path

    p = Path(tempfile.mkdtemp()) / "orders.csv"
    p.write_text("id,amount,late\n1,10,0\n2,20,1\n")
    return p


def test_predictions_print_readably():
    from predictabl.client import Predictions

    one = Predictions._from(
        {
            "meaning": "probability that customer churned",
            "predictions": [{"entity_id": "0", "prediction": 0.8213}],
        }
    )
    assert repr(one) == "Probability that customer churned: 0.82"
    many = Predictions._from(
        {
            "meaning": "probability that customer churned",
            "count": 500,
            "predictions": [
                {"entity_id": f"C-{i}", "prediction": 0.5} for i in range(12)
            ],
        }
    )
    text = str(many)
    assert text.splitlines()[0] == "Probability that customer churned"
    assert "  C-0  0.50" in text and "C-10" not in text
    assert "… and 2 more" in text and "12 of 500 returned" in text
    cls = Predictions._from(
        {
            "meaning": "probability of the predicted class (tier)",
            "predictions": [
                {"entity_id": "a", "prediction": 0.7, "predicted_class": "gold"}
            ],
        }
    )
    assert str(cls).endswith(": gold (0.70)")
    assert list(one) == [{"entity_id": "0", "prediction": 0.8213}]  # still a plain list


def test_connect_a_postgres_database_reads_it_in_place(capsys):
    c, api = make()
    url = "postgresql://ro_user:s3cret@db.acme.com:5432/sales"
    updates = []
    data = c.connect(url, on_progress=updates.append)
    assert data.status == "ready"
    posts = [(p, b) for m, p, b, _ in api.calls if m == "POST"]
    assert posts[0] == (
        "/connections",
        {"name": "sales on db.acme.com", "kind": "postgres", "config": {"dsn": url}},
    )
    assert posts[1][0] == "/connections/c1/introspect"
    shown = " ".join(f"{u.step} {u.detail}" for u in updates) + capsys.readouterr().err
    assert "s3cret" not in shown and "ro_user" not in shown
    assert "sales on db.acme.com" in shown
    assert not any(p == "/connections/from-csv" for _, p, _, _ in api.calls)


def test_connect_refuses_unsupported_databases_clearly():
    c, api = make()
    with pytest.raises(pb.InvalidRequest, match="Postgres"):
        c.connect("mysql://u:p@host/db", quiet=True)
    assert api.calls == []


def test_upload_still_works_but_warns():
    c, api = make()
    with pytest.warns(DeprecationWarning, match="connect"):
        data = c.upload(_tmp_csv(), quiet=True)
    assert data.status == "ready"
    assert "upload" not in pb.__all__ and "connect" in pb.__all__
