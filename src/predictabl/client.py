"""Client and the objects it returns: Data, Plan, Model, Predictions.

The server does the work (understanding the data, planning features, training,
scoring); this module turns its endpoints into a few calls with progress, and
waits on background jobs for you.
"""

from __future__ import annotations

import io
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from . import errors
from ._http import Transport, resolve_base_url
from ._progress import Callback, Progress

# Run states, in order, with the name people see.
_STEPS = {
    "QUEUED": "Queued",
    "PLAN": "Preparing",
    "BUILD_DATASET": "Building data",
    "VALIDATE_DATASET": "Checking data",
    "TRAIN": "Training",
    "EVALUATE": "Testing",
    "SCORE": "Final model",
}
_TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}
_RECORDS_PER_CALL = 1000  # the server's cap for predict-records
_MAX_LIMIT = 2000  # the server's cap for `top`


class Client:
    """A connection to Predictabl.

    `api_key` defaults to the PREDICTABL_API_KEY environment variable and
    `base_url` to PREDICTABL_BASE_URL, else the hosted app. For a local
    instance use `base_url="http://localhost:8000/api/v1"`.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        timeout: float = 120.0,
        poll_interval: float = 2.0,
        client_tag: str | None = None,
        _http=None,
    ):
        self.api_key = api_key or os.environ.get("PREDICTABL_API_KEY")
        self.base_url = resolve_base_url(base_url)
        self.poll_interval = poll_interval
        self._t = Transport(
            self.api_key,
            self.base_url,
            timeout=timeout,
            client_tag=client_tag,
            http=_http,
        )

    def close(self) -> None:
        self._t.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<predictabl.Client {self.base_url}>"

    # --------------------------------------------------------------- account

    def account(self) -> dict:
        """Who you are, your plan, and builds used/left."""
        return self._t.get("/account")

    # ------------------------------------------------------------------ data

    def datasets(self) -> list[Data]:
        return [Data._from(self, row) for row in self._t.get("/connections") or []]

    def data(self, data_id: str) -> Data:
        for row in self._t.get("/connections") or []:
            if row["id"] == data_id:
                return Data._from(self, row)
        raise errors.NotFound(f"No dataset {data_id!r} on this account.")

    def upload(
        self,
        source: str | Path | Iterable[str | Path] | Any,
        name: str | None = None,
        *,
        wait: bool = True,
        quiet: bool = False,
        on_progress: Callback | None = None,
        timeout: float | None = 1800,
    ) -> Data:
        """Upload CSV, Excel or Parquet files (or a pandas DataFrame) as a new
        dataset. Several files become several tables Predictabl can join.

        With `wait` (default) returns once Predictabl has read the tables."""
        files, label = _files_of(source)
        progress = Progress(quiet=quiet, on_update=on_progress)
        size = sum(len(content) for _, content, _ in files)
        progress.running("Uploading", f"{label} · {_human_bytes(size)}")
        try:
            row = self._t.post(
                "/connections/from-csv",
                data={"name": name or label},
                files=[
                    ("files", (fname, content, ctype)) for fname, content, ctype in files
                ],
            )
        except errors.PredictablError:
            progress.failed("Uploading")
            raise
        progress.done("Uploading", f"{label} · {_human_bytes(size)}")
        data = Data._from(self, row)
        if wait:
            data.wait_ready(progress=progress, timeout=timeout)
        return data

    def plan(self, question_id: str) -> Plan:
        """The latest plan of a question (opportunity) planned earlier, e.g.
        in the web app, to retrain it from code without planning again."""
        got = self._t.get(f"/opportunities/{question_id}/spec") or {}
        spec = got.get("spec")
        if not spec:
            raise errors.NotFound(f"Question {question_id!r} has no plan yet.")
        return Plan(
            id=spec["id"],
            version=int(got.get("version") or 1),
            opportunity_id=question_id,
            spec=spec,
            _client=self,
        )

    # ---------------------------------------------------------------- models

    def models(self, limit: int = 50) -> list[Model]:
        rows = self._t.get("/runs", limit=limit) or []
        return [Model._from(self, row) for row in rows[:limit]]

    def model(self, model_id: str) -> Model:
        return Model._from(self, self._t.get(f"/runs/{model_id}"))

    # --------------------------------------------------------------- helpers

    def _sleep(self, progress: Progress | None) -> None:
        end = time.monotonic() + self.poll_interval
        while True:
            left = end - time.monotonic()
            if left <= 0:
                return
            if progress is not None:
                progress.tick()
            time.sleep(min(0.2, left))

    @property
    def web_url(self) -> str:
        """Where the web app for this instance lives (best effort)."""
        base = self.base_url
        for suffix in ("/pub/v1", "/api/v1"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        if base.startswith(("http://localhost:8000", "http://127.0.0.1:8000")):
            return base.replace(":8000", ":3000")
        return base


# ============================================================================


@dataclass
class Data:
    """A dataset on Predictabl: uploaded files or a connected database."""

    id: str
    name: str
    status: str
    kind: str | None = None
    table_count: int | None = None
    error: str | None = None
    _client: Client = field(repr=False, default=None)  # type: ignore[assignment]

    @classmethod
    def _from(cls, client: Client, row: Mapping) -> Data:
        return cls(
            id=row["id"],
            name=row.get("name") or "",
            status=row.get("status") or "",
            kind=row.get("kind"),
            table_count=row.get("table_count"),
            error=row.get("error"),
            _client=client,
        )

    def refresh(self) -> Data:
        fresh = self._client.data(self.id)
        self.status, self.table_count, self.error = (
            fresh.status,
            fresh.table_count,
            fresh.error,
        )
        return self

    def wait_ready(
        self, *, progress: Progress | None = None, timeout: float | None = 1800
    ) -> Data:
        """Wait until Predictabl has read the tables (status "ready")."""
        progress = progress or Progress(quiet=True)
        started = time.monotonic()
        progress.running("Reading data", "understanding the tables")
        while True:
            self.refresh()
            if self.status == "ready":
                n = self.table_count
                progress.done(
                    "Reading data", f"{n} table{'s' if n != 1 else ''}" if n else ""
                )
                return self
            if self.status in ("error", "failed"):
                progress.failed("Reading data", self.error or "")
                reason = self.error or "unknown error"
                raise errors.BuildFailed(f"Could not read this dataset: {reason}")
            _check_timeout(started, timeout, "reading the data")
            self._client._sleep(progress)

    def questions(self) -> list[dict]:
        """Questions (prediction opportunities) found or asked on this data."""
        return self._client._t.get(f"/connections/{self.id}/opportunities") or []

    def plan(
        self,
        question: str,
        *,
        quiet: bool = False,
        on_progress: Callback | None = None,
        timeout: float | None = 1800,
    ) -> Plan:
        """Turn a question into a reviewable plan: target, features and what
        was excluded as leakage. Spends one build on Predictabl's AI key."""
        progress = Progress(quiet=quiet, on_update=on_progress)
        opp = self._ask(question, progress, timeout)
        return Plan._generate(self._client, opp, progress, timeout)

    def predict(
        self,
        question: str,
        *,
        preset: str = "full_quality",
        engine: str = "auto",
        quiet: bool = False,
        on_progress: Callback | None = None,
        timeout: float | None = None,
    ) -> Model:
        """The whole path: question -> plan -> trained, tested model."""
        progress = Progress(quiet=quiet, on_update=on_progress)
        opp = self._ask(question, progress, 1800)
        plan = Plan._generate(self._client, opp, progress, 1800)
        return plan._train(preset, engine, progress, timeout)

    def delete(self) -> None:
        self._client._t.delete(f"/connections/{self.id}")

    # ------------------------------------------------------------- internals

    def _ask(self, question: str, progress: Progress, timeout: float | None) -> dict:
        t = self._client._t
        if self.status != "ready":
            self.wait_ready(progress=progress)
        progress.running("Understanding", question)
        before = {o["id"] for o in self.questions()}
        queued = (
            t.post(
                f"/connections/{self.id}/opportunities",
                json={"mode": "single", "user_context": question},
            )
            or {}
        )
        job_id = queued.get("job_id")
        started = time.monotonic()
        while True:
            job = t.get(f"/connections/{self.id}/opportunities/job") or {}
            state = job.get("state")
            # Only OUR job counts: an earlier one on this data may be "done".
            mine = job_id is None or job.get("id") == job_id
            if mine and state in ("done", "failed"):
                break
            _check_timeout(started, timeout, "understanding the question")
            self._client._sleep(progress)
        new = [o for o in self.questions() if o["id"] not in before]
        if state == "failed" or not new:
            msg = job.get("message") or "Predictabl could not frame this question."
            progress.failed("Understanding", msg)
            raise errors.BuildFailed(msg)
        opp = max(new, key=lambda o: o.get("created_at") or "")
        progress.done("Understanding", opp.get("title") or question)
        return opp


# ============================================================================


@dataclass
class Plan:
    """What Predictabl will train: the target, the features, and what it
    excluded as leakage. Review it, refine it in plain words, then train."""

    id: str
    version: int
    opportunity_id: str
    spec: dict = field(repr=False)
    _client: Client = field(repr=False, default=None)  # type: ignore[assignment]

    @property
    def task(self) -> str | None:
        return self.spec.get("task")

    @property
    def target(self) -> str:
        t = self.spec.get("target") or {}
        return t.get("definition") or t.get("name") or ""

    @property
    def features(self) -> list[str]:
        return [
            f["name"] for f in self.spec.get("features") or [] if f.get("enabled", True)
        ]

    @property
    def excluded(self) -> list[dict]:
        """Columns kept out of the model, with the reason (usually leakage)."""
        return list(self.spec.get("leakage_exclusions") or [])

    def summary(self) -> str:
        lines = [f"Predicting: {self.target}", f"Task: {self.task}"]
        lines.append(f"Features ({len(self.features)}): " + ", ".join(self.features[:12]))
        if len(self.features) > 12:
            lines[-1] += f", … (+{len(self.features) - 12})"
        for ex in self.excluded:
            col = ex.get("column") or ex.get("name") or "?"
            lines.append(f"Excluded {col}: {ex.get('reason') or ''}".rstrip(": "))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def refine(
        self,
        message: str,
        *,
        quiet: bool = False,
        on_progress: Callback | None = None,
        timeout: float | None = 1800,
    ) -> Plan:
        """Change the plan in plain words ("don't use the courier column").
        Returns the new version; this object stays the version it was."""
        t = self._client._t
        progress = Progress(quiet=quiet, on_update=on_progress)
        progress.running("Refining", message)
        t.post(
            f"/opportunities/{self.opportunity_id}/spec/refine", json={"message": message}
        )
        new = _wait_for_spec(
            self._client, self.opportunity_id, progress, timeout, newer_than=self.version
        )
        progress.done("Refining", f"version {new.version}")
        return new

    def train(
        self,
        *,
        preset: str = "full_quality",
        engine: str = "auto",
        wait: bool = True,
        quiet: bool = False,
        on_progress: Callback | None = None,
        timeout: float | None = None,
    ) -> Model:
        """Train and test this plan. `preset="fast_preview"` is a ~15-second
        sanity check on a sample; the default is the real model."""
        progress = Progress(quiet=quiet, on_update=on_progress)
        return self._train(preset, engine, progress, timeout, wait=wait)

    # ------------------------------------------------------------- internals

    @classmethod
    def _generate(
        cls, client: Client, opp: dict, progress: Progress, timeout: float | None
    ) -> Plan:
        progress.running("Planning", "target, features, leakage checks")
        client._t.post(f"/opportunities/{opp['id']}/spec", json={})
        plan = _wait_for_spec(client, opp["id"], progress, timeout)
        progress.done(
            "Planning",
            f"{len(plan.features)} features · {len(plan.excluded)} excluded",
        )
        return plan

    def _train(
        self,
        preset: str,
        engine: str,
        progress: Progress,
        timeout: float | None,
        *,
        wait: bool = True,
    ) -> Model:
        row = self._client._t.post(
            "/runs", json={"spec_id": self.id, "preset": preset, "engine": engine}
        )
        model = Model._from(self._client, row)
        if wait:
            model._wait(progress, timeout)
        return model


def _wait_for_spec(
    client: Client,
    opportunity_id: str,
    progress: Progress,
    timeout: float | None,
    *,
    newer_than: int | None = None,
) -> Plan:
    t = client._t
    started = time.monotonic()
    while True:
        got = t.get(f"/opportunities/{opportunity_id}/spec") or {}
        spec = got.get("spec")
        if spec and (newer_than is None or int(got.get("version") or 0) > newer_than):
            return Plan(
                id=spec["id"],
                version=int(got.get("version") or spec.get("version") or 1),
                opportunity_id=opportunity_id,
                spec=spec,
                _client=client,
            )
        opp = t.get(f"/opportunities/{opportunity_id}") or {}
        failure = opp.get("last_generation_error")
        if failure:
            msg = failure.get("message") if isinstance(failure, dict) else str(failure)
            progress.failed(detail=msg or "")
            raise errors.BuildFailed(msg or "Planning failed.")
        _check_timeout(started, timeout, "planning")
        client._sleep(progress)


# ============================================================================


@dataclass
class Model:
    """A trained (or training) model. `predict()` scores with it."""

    id: str
    state: str
    title: str | None = None
    kind: str | None = None  # forecast | event | flat
    meaning: str | None = None
    created_via: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    suggestions: list = field(default_factory=list)
    _client: Client = field(repr=False, default=None)  # type: ignore[assignment]

    @classmethod
    def _from(cls, client: Client, row: Mapping) -> Model:
        return cls(
            id=row["id"],
            state=row.get("state") or "",
            title=row.get("opportunity_title"),
            kind=row.get("kind"),
            meaning=row.get("meaning"),
            created_via=row.get("created_via"),
            error_code=row.get("error_code"),
            error_message=row.get("error_message"),
            suggestions=row.get("suggestions") or [],
            _client=client,
        )

    @property
    def url(self) -> str:
        """This model in the web app."""
        return f"{self._client.web_url}/runs/{self.id}"

    @property
    def ready(self) -> bool:
        return self.state == "SUCCEEDED"

    def refresh(self) -> Model:
        fresh = self._client.model(self.id)
        self.__dict__.update({k: v for k, v in fresh.__dict__.items() if k != "_client"})
        return self

    def wait(
        self,
        *,
        quiet: bool = False,
        on_progress: Callback | None = None,
        timeout: float | None = None,
    ) -> Model:
        """Block until training ends. Raises BuildFailed if it failed."""
        return self._wait(Progress(quiet=quiet, on_update=on_progress), timeout)

    def events(self) -> list[dict]:
        """Everything the run reported, step by step."""
        return self._client._t.get(f"/runs/{self.id}/events.json") or []

    def cancel(self) -> None:
        self._client._t.post(f"/runs/{self.id}/cancel")

    def report(self) -> dict:
        """The tested result: metric, rows, top features, what it predicts."""
        res = self._client._t.get(f"/runs/{self.id}/results") or {}
        winner = res.get("winner")
        scores = (res.get("engines") or {}).get(winner) or {}
        metric = res.get("primary_metric") or _default_metric(res.get("task"))
        imp = res.get("feature_importance") or []
        if isinstance(imp, dict):
            imp = imp.get(winner) or next(iter(imp.values()), [])
        top = [
            (f.get("feature") or f.get("name")) if isinstance(f, dict) else f
            for f in (imp or [])[:10]
        ]
        if self.kind is None:
            self.refresh()
        return {
            "model": self.id,
            "predicts": self.title,
            "meaning": self.meaning,
            "kind": self.kind,
            "task": res.get("task"),
            "engine": winner,
            "metric": metric,
            "score": scores.get(metric) if metric else None,
            "score_std": (scores.get("metric_std") or {}).get(metric) if metric else None,
            "rows_trained": res.get("n_train"),
            "rows_tested": res.get("n_test"),
            "top_features": [t for t in top if t],
            "url": self.url,
        }

    def predict(
        self,
        rows: Any = None,
        *,
        ids: Iterable[str] | str | None = None,
        since: str | date | datetime | None = None,
        until: str | date | datetime | None = None,
        since_days: int | None = None,
        as_of: str | date | datetime | None = None,
        limit: int | None = None,
        filters: list[dict] | None = None,
        include: list[str] | None = None,
        calibrated: bool = False,
    ) -> Predictions:
        """Score with this model.

        * `rows` (DataFrame, list of dicts or one dict): score these records.
          Predictabl builds the features from them with the model's own SQL.
        * otherwise Predictabl scores from the data it was trained on:
          forecasts from the latest data (or `as_of`), event models for rows
          dated today (or `since`/`until`/`since_days`), optionally only
          `ids`, highest first, at most `limit` (server cap 2000).
        """
        if rows is not None:
            return self._predict_rows(rows, calibrated)
        body: dict[str, Any] = {
            "top": min(int(limit), _MAX_LIMIT) if limit else _MAX_LIMIT,
            "calibrated": calibrated,
        }
        if since is not None:
            body["since"] = _iso(since)
        if until is not None:
            body["until"] = _iso(until)
        if since_days is not None:
            body["since_days"] = int(since_days)
        if as_of is not None:
            body["cutoff"] = _iso(as_of)
        if filters:
            body["filters"] = filters
        if include:
            body["include"] = include
        if ids is None:
            return Predictions._from(
                self._client._t.post(f"/runs/{self.id}/predict", json=body)
            )
        wanted = [ids] if isinstance(ids, str) else [str(i) for i in ids]
        merged: Predictions | None = None
        for one in wanted:
            got = self._client._t.post(
                f"/runs/{self.id}/predict", json={**body, "entity_contains": one}
            )
            part = Predictions._from(got)
            # entity_contains is a substring match; keep exact ids only.
            keep = [p for p in part if _is_entity(p.get("entity_id"), one)]
            if merged is None:
                merged = part
                merged[:] = keep
            else:
                merged.extend(keep)
        assert merged is not None
        merged.count = len(merged)
        return merged

    # ------------------------------------------------------------- internals

    def _predict_rows(self, rows: Any, calibrated: bool) -> Predictions:
        records = _records_of(rows)
        if not records:
            raise errors.InvalidRequest("No rows to score.")
        out: Predictions | None = None
        for i in range(0, len(records), _RECORDS_PER_CALL):
            got = self._client._t.post(
                f"/runs/{self.id}/predict-records",
                json={
                    "rows": records[i : i + _RECORDS_PER_CALL],
                    "calibrated": calibrated,
                },
            )
            part = Predictions._from(got)
            if out is None:
                out = part
            else:
                out.extend(part)
        assert out is not None
        out.count = len(out)
        return out

    def _wait(self, progress: Progress, timeout: float | None) -> Model:
        started = time.monotonic()
        seen = 0
        while True:
            self.refresh()
            if self.state in _STEPS:
                detail = ""
                events = self.events()
                for ev in events[seen:]:
                    if ev.get("state") == self.state and ev.get("message"):
                        detail = _short(ev["message"])
                seen = len(events)
                progress.running(_STEPS[self.state], detail)
            if self.state in _TERMINAL:
                break
            _check_timeout(started, timeout, "training")
            self._client._sleep(progress)
        if self.state == "SUCCEEDED":
            rep = self.report()
            score = rep.get("score")
            detail = (
                f"{rep['metric']} {score:.4f}" if isinstance(score, float) else "done"
            )
            progress.done("Final model", detail)
            return self
        progress.failed(detail=self.error_message or self.state.lower())
        raise errors.BuildFailed(
            self.error_message or f"Run {self.state.lower()}.",
            code=self.error_code,
            suggestions=self.suggestions,
        )


# ============================================================================


class Predictions(list):
    """A list of predictions (dicts with `entity_id`, `prediction`, ...),
    plus what they mean: `meaning`, `kind`, `as_of`, and any `note` or
    `warning` the server attached."""

    meaning: str | None
    kind: str | None
    as_of: str | None
    note: str | None
    warning: str | None
    where_used: str | None
    count: int
    raw: dict

    @classmethod
    def _from(cls, data: Mapping) -> Predictions:
        p = cls(data.get("predictions") or [])
        p.meaning = data.get("meaning")
        p.kind = data.get("kind")
        p.as_of = data.get("as_of")
        p.note = data.get("note")
        p.warning = data.get("warning")
        p.where_used = data.get("where_used")
        p.count = int(data.get("count") or len(p))
        p.raw = {k: v for k, v in data.items() if k != "predictions"}
        return p

    def to_pandas(self):
        """The predictions as a DataFrame (needs pandas). Context columns
        requested with `include` become their own columns."""
        import pandas as pd

        flat = []
        for p in self:
            row = {k: v for k, v in p.items() if k not in ("context", "probabilities")}
            for k, v in (p.get("context") or {}).items():
                row[k] = v
            for k, v in (p.get("probabilities") or {}).items():
                row[f"p_{k}"] = v
            flat.append(row)
        return pd.DataFrame(flat)

    def __repr__(self) -> str:
        head = f"<Predictions {len(self)}"
        if self.count and self.count > len(self):
            head += f" of {self.count}"
        if self.meaning:
            head += f" · {self.meaning}"
        if self.as_of:
            head += f" · as of {self.as_of}"
        return head + ">"


# ============================================================================


def _files_of(source: Any) -> tuple[list[tuple[str, bytes, str]], str]:
    """(files, label) for an upload: paths, a list of paths, or a DataFrame."""
    if hasattr(source, "to_csv") and hasattr(source, "columns"):  # DataFrame
        buf = io.StringIO()
        source.to_csv(buf, index=False)
        return [("data.csv", buf.getvalue().encode(), "text/csv")], "data.csv"
    paths = [source] if isinstance(source, str | Path) else list(source)
    files = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise errors.InvalidRequest(f"No such file: {path}")
        files.append((path.name, path.read_bytes(), _content_type(path)))
    label = files[0][0] if len(files) == 1 else f"{len(files)} files"
    return files, label


def _content_type(path: Path) -> str:
    return {
        ".csv": "text/csv",
        ".parquet": "application/octet-stream",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
    }.get(path.suffix.lower(), "application/octet-stream")


def _records_of(rows: Any) -> list[dict]:
    if isinstance(rows, Mapping):
        return [_jsonable(dict(rows))]
    if hasattr(rows, "to_dict") and hasattr(rows, "columns"):  # DataFrame
        import json

        return json.loads(rows.to_json(orient="records", date_format="iso"))
    return [_jsonable(dict(r)) for r in rows]


def _jsonable(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, datetime | date):
            out[k] = v.isoformat()
        elif hasattr(v, "item") and callable(v.item):  # numpy scalar
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _iso(v: str | date | datetime) -> str:
    if isinstance(v, datetime):
        return v.isoformat(sep=" ", timespec="seconds")
    if isinstance(v, date):
        return v.isoformat()
    return str(v)


def _is_entity(entity_id: str | None, wanted: str) -> bool:
    if entity_id is None:
        return False
    return entity_id == wanted or entity_id.split("·")[0] == wanted


def _default_metric(task: str | None) -> str | None:
    return {
        "binary_classification": "auc",
        "multiclass": "accuracy",
        "regression": "rmse",
    }.get(task or "")


def _short(message: str, width: int = 70) -> str:
    message = " ".join(message.split())
    return message if len(message) <= width else message[: width - 1] + "…"


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"  # pragma: no cover


def _check_timeout(started: float, timeout: float | None, what: str) -> None:
    if timeout is not None and time.monotonic() - started > timeout:
        raise errors.PredictablError(
            f"Timed out after {timeout:.0f}s while {what}. It may still be running "
            "on the server; check the web app."
        )
