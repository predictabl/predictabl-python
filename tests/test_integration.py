"""End-to-end against a real Predictabl backend.

Skipped unless pointed at one:

    PREDICTABL_TEST_URL=http://localhost:8010/api/v1 \\
    PREDICTABL_TEST_KEY=pk_...            # a `build` token
    PREDICTABL_TEST_INFERENCE_KEY=pk_...  # an `inference` token, same user
    PREDICTABL_TEST_OTHER_KEY=pk_...      # a token of ANOTHER user
    uv run pytest -m integration

Point it at models on the instance (id prefixes are fine):
PREDICTABL_TEST_FLAT (a flat binary classifier, e.g. airline satisfaction,
scored with rows from PREDICTABL_TEST_FLAT_CSV), PREDICTABL_TEST_FORECAST (a
daily forecast with entity PREDICTABL_TEST_FORECAST_ENTITY and a date inside
its data, PREDICTABL_TEST_FORECAST_AS_OF) and PREDICTABL_TEST_EVENT.
Nothing here calls an LLM: scoring, scopes, isolation, uploads, and training
an existing plan with the fast preset.
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import pytest

import predictabl as pb
from predictabl import errors

URL = os.environ.get("PREDICTABL_TEST_URL")
KEY = os.environ.get("PREDICTABL_TEST_KEY")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (URL and KEY), reason="no PREDICTABL_TEST_URL/KEY"),
]

FLAT = os.environ.get("PREDICTABL_TEST_FLAT", "")
FORECAST = os.environ.get("PREDICTABL_TEST_FORECAST", "")
EVENT = os.environ.get("PREDICTABL_TEST_EVENT", "")
AIRLINE_CSV = Path(os.environ.get("PREDICTABL_TEST_FLAT_CSV", "/nonexistent"))
ENTITY = os.environ.get("PREDICTABL_TEST_FORECAST_ENTITY", "")
AS_OF = os.environ.get("PREDICTABL_TEST_FORECAST_AS_OF", "")


@pytest.fixture(scope="module")
def c():
    with pb.Client(api_key=KEY, base_url=URL, poll_interval=1) as client:
        yield client


@pytest.fixture(scope="module")
def ids(c):
    """Resolve the id prefixes to full ids through the API."""
    all_ids = [m.id for m in c.models(limit=1000)]

    def full(prefix):
        if not prefix:
            pytest.skip("model id not configured")
        hits = [i for i in all_ids if i.startswith(prefix)]
        if not hits:
            pytest.skip(f"model {prefix} not on this instance")
        return hits[0]

    return {"flat": full(FLAT), "forecast": full(FORECAST), "event": full(EVENT)}


def _airline_rows(n):
    if not AIRLINE_CSV.exists():
        pytest.skip("airline CSV not available")
    with AIRLINE_CSV.open() as fh:
        reader = csv.DictReader(fh)
        return [next(reader) for _ in range(n)]


# ---------------------------------------------------------------- account


def test_account_through_a_token(c):
    acct = c.account()
    assert acct["email"] and acct["plan"] in ("free", "unlimited")
    assert "builds_used" in acct


# ---------------------------------------------------------------- scoring


def test_flat_model_facts_and_report(c, ids):
    m = c.model(ids["flat"])
    assert m.kind == "flat" and m.meaning
    rep = m.report()
    assert rep["metric"] == "auc" and 0.5 < rep["score"] <= 1
    assert rep["rows_trained"] > 0 and rep["top_features"]


def test_flat_model_refuses_to_score_without_rows(c, ids):
    with pytest.raises(errors.InvalidRequest, match="no time axis"):
        c.model(ids["flat"]).predict()


def test_flat_model_scores_rows_and_ignores_the_outcome(c, ids):
    rows = _airline_rows(1200)  # more than one server batch
    m = c.model(ids["flat"])
    with_outcome = m.predict(rows=rows)
    without = m.predict(
        rows=[{k: v for k, v in r.items() if k != "satisfaction"} for r in rows]
    )
    assert len(with_outcome) == len(without) == 1200
    assert [p["prediction"] for p in with_outcome] == [p["prediction"] for p in without]
    assert all("actual" in p for p in with_outcome)
    assert not any("actual" in p for p in without)
    assert with_outcome.meaning == "probability that passenger is satisfied"
    assert with_outcome.warning is None


def test_flat_model_scores_a_dataframe(c, ids):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(_airline_rows(5)).drop(columns=["satisfaction"])
    out = c.model(ids["flat"]).predict(rows=df).to_pandas()
    assert len(out) == 5 and out["prediction"].between(0, 1).all()


def test_forecast_states_its_as_of(c, ids):
    preds = c.model(ids["forecast"]).predict(limit=5)
    assert preds.kind == "forecast" and preds.as_of
    assert len(preds) == 5


def test_forecast_for_one_entity_at_a_chosen_date(c, ids):
    if not (ENTITY and AS_OF):
        pytest.skip("forecast entity/as_of not configured")
    preds = c.model(ids["forecast"]).predict(ids=[ENTITY], as_of=AS_OF)
    assert preds.as_of == AS_OF
    assert len(preds) > 0
    assert all(p["entity_id"].split("·")[0] == ENTITY for p in preds)


def test_event_model_explains_an_empty_default_window(c, ids):
    preds = c.model(ids["event"]).predict()
    if len(preds) == 0:
        assert preds.note and "newest" in preds.note
    assert preds.kind == "event"


# ------------------------------------------------------- scopes, isolation


def test_an_inference_token_scores_but_cannot_build(ids):
    key = os.environ.get("PREDICTABL_TEST_INFERENCE_KEY")
    if not key:
        pytest.skip("no inference key")
    with pb.Client(api_key=key, base_url=URL) as inf:
        assert len(inf.model(ids["flat"]).predict(rows=_airline_rows(2))) == 2
        with pytest.raises(errors.PermissionDenied):
            inf.connect(_tiny_csv(), quiet=True)
        with pytest.raises(errors.PermissionDenied):
            inf.datasets()


def test_another_users_token_sees_nothing_of_ours(ids):
    key = os.environ.get("PREDICTABL_TEST_OTHER_KEY")
    if not key:
        pytest.skip("no other-user key")
    with pb.Client(api_key=key, base_url=URL) as other:
        assert ids["flat"] not in {m.id for m in other.models(limit=1000)}
        with pytest.raises(errors.NotFound):
            other.model(ids["flat"])
        with pytest.raises(errors.NotFound):
            other.model_unchecked = pb.Model(id=ids["flat"], state="", _client=other)
            other.model_unchecked.predict(rows=[{"Age": 30}])
        with pytest.raises(errors.NotFound):
            pb.Model(id=ids["forecast"], state="", _client=other).predict(limit=1)


def test_a_bad_key_is_an_authentication_error():
    with pb.Client(api_key="pk_not-a-real-key", base_url=URL) as bad:
        with pytest.raises(errors.AuthenticationError):
            bad.account()


# ------------------------------------------------------------ build path


def test_connect_creates_a_dataset_marked_with_the_token(c):
    """Introspection needs the LLM; on a server without a key it fails, and
    the SDK must say so instead of hanging."""
    try:
        data = c.connect(_tiny_csv(), name="sdk-integration", quiet=True, timeout=300)
        assert data.status == "ready"
    except errors.BuildFailed as exc:
        assert "Could not read this dataset" in str(exc)
        data = next(d for d in c.datasets() if d.name == "sdk-integration")
    finally:
        pass
    data.delete()
    assert all(d.id != data.id for d in c.datasets())


def test_retrain_an_existing_plan_from_code(c, ids):
    """A real training run, no LLM: the flat model's own plan, fast preset."""
    run = c._t.get(f"/runs/{ids['flat']}")
    plan = c.plan(run["opportunity_id"])
    assert plan.features
    updates = []
    model = plan.train(preset="fast_preview", quiet=True, on_progress=updates.append)
    try:
        assert model.ready and model.created_via == "sdk integration build"
        assert model.report()["score"] > 0.5
        assert ("Final model", "done") in [(u.step, u.status) for u in updates]
        assert len(model.predict(rows=_airline_rows(3))) == 3
    finally:
        c._t.delete(f"/runs/{model.id}")


# ------------------------------------------------------------------- MCP


def test_mcp_server_over_stdio(ids):
    """Launch `predictabl mcp` as a real subprocess and call tools over stdio."""
    pytest.importorskip("mcp")
    import asyncio
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def go():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "predictabl", "--base-url", URL, "mcp"],
            env={**os.environ, "PREDICTABL_API_KEY": KEY},
        )
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            names = {t.name for t in (await s.list_tools()).tools}
            assert {"list_models", "predict", "predict_rows"} <= names
            listed = await s.call_tool("list_models", {"limit": 500})
            assert ids["flat"] in str(listed.content)
            scored = await s.call_tool("predict", {"run_id": ids["forecast"], "top": 3})
            assert "as_of" in str(scored.content)

    asyncio.run(go())


def _tiny_csv():
    p = Path(tempfile.mkdtemp()) / "tiny.csv"
    p.write_text("id,amount,late\n1,10,0\n2,20,1\n3,30,0\n")
    return p
