"""An in-memory stand-in for the Predictabl API, for unit tests.

It mimics the real contracts the client relies on: the `{data, error}`
envelope, FastAPI's `{detail}` refusals, background work that finishes over
several polls, and the caps (1000 records per call, top <= 2000).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import httpx


def ok(data, status=200):
    return httpx.Response(status, json={"data": data, "error": None})


def err(code, message, status):
    return httpx.Response(
        status, json={"data": None, "error": {"code": code, "message": message}}
    )


@dataclass
class FakeApi:
    polls_until_ready: int = 2
    polls_until_spec: int = 2
    run_states: list = field(
        default_factory=lambda: [
            "QUEUED",
            "PLAN",
            "BUILD_DATASET",
            "TRAIN",
            "EVALUATE",
            "SUCCEEDED",
        ]
    )
    fail_run: bool = False
    fail_plan: bool = False
    older_job_done: bool = True
    calls: list = field(default_factory=list)
    _conn_polls: int = 0
    _spec_polls: int = 0
    _run_i: int = 0
    _opps: list = field(default_factory=list)
    _spec_version: int = 0
    _job_polls: int = 0
    _pending: bool = False

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, req: httpx.Request) -> httpx.Response:
        path = req.url.path.split("/api/v1", 1)[-1]
        body = None
        if req.content and req.headers.get("content-type", "").startswith(
            "application/json"
        ):
            body = json.loads(req.content)
        self.calls.append((req.method, path, body, dict(req.headers)))
        m = req.method

        if path == "/account" and m == "GET":
            return ok({"email": "a@b.c", "plan": "free", "builds_left": 1})
        if path == "/connections/from-csv" and m == "POST":
            return ok(
                {
                    "id": "c1",
                    "name": "orders",
                    "status": "introspecting",
                    "kind": "duckdb",
                }
            )
        if path == "/connections" and m == "GET":
            self._conn_polls += 1
            status = (
                "ready" if self._conn_polls > self.polls_until_ready else "introspecting"
            )
            return ok(
                [{"id": "c1", "name": "orders", "status": status, "table_count": 1}]
            )
        if path == "/connections/c1/opportunities" and m == "GET":
            return ok(list(self._opps))
        if path == "/connections/c1/opportunities" and m == "POST":
            self._job_polls = 0
            return ok({"job": "generate_opportunities", "job_id": "job-new"}, 202)
        if path == "/connections/c1/opportunities/job":
            self._job_polls += 1
            # First poll: the server still shows an OLDER, finished job.
            if self.older_job_done and self._job_polls == 1:
                return ok({"id": "job-old", "state": "done", "added": 1})
            if self._job_polls < 3:
                return ok({"id": "job-new", "state": "running"})
            if not self._opps:
                self._opps.append(
                    {
                        "id": "o1",
                        "title": "Late orders",
                        "created_at": "2026-11-14T10:00:00",
                    }
                )
            return ok({"id": "job-new", "state": "done", "added": 1})
        if path == "/opportunities/o1/spec" and m == "POST":
            self._spec_polls, self._pending = 0, True
            return ok({"job": "generate_spec"}, 202)
        if path == "/opportunities/o1/spec/refine" and m == "POST":
            self._spec_polls, self._pending = 0, True
            return ok({"job": "refine_spec"}, 202)
        if path == "/opportunities/o1/spec" and m == "GET":
            self._spec_polls += 1
            if (
                self._pending
                and self._spec_polls > self.polls_until_spec
                and not self.fail_plan
            ):
                self._spec_version += 1  # the background generation lands
                self._pending = False
            if self._spec_version == 0:
                return ok({"spec": None, "version": None, "history": []})
            return ok({"spec": self._spec(), "version": self._spec_version})
        if path == "/opportunities/o1" and m == "GET":
            payload = {"id": "o1", "title": "Late orders"}
            if self.fail_plan:
                payload["last_generation_error"] = {"message": "Planning failed: bad SQL"}
            return ok(payload)
        if path == "/runs" and m == "POST":
            self._run_i = 0
            return ok(self._run(), 201)
        if path == "/runs" and m == "GET":
            return ok([self._run(state="SUCCEEDED")])
        mrun = re.fullmatch(r"/runs/(r1)(/.*)?", path)
        if mrun:
            sub = mrun.group(2) or ""
            if sub == "" and m == "GET":
                self._run_i = min(self._run_i + 1, len(self.run_states) - 1)
                return ok(self._run())
            if sub == "/events.json":
                return ok([{"state": "TRAIN", "message": "lightgbm: tree 840/2000"}])
            if sub == "/results":
                return ok(
                    {
                        "task": "binary_classification",
                        "winner": "lightgbm",
                        "engines": {
                            "lightgbm": {"auc": 0.8712, "metric_std": {"auc": 0.004}}
                        },
                        "n_train": 900,
                        "n_test": 100,
                        "feature_importance": [{"feature": "days_late_before"}],
                    }
                )
            if sub == "/predict-records":
                rows = body["rows"]
                if len(rows) > 1000:
                    return err("TOO_MANY", "max 1000", 422)
                preds = [
                    {"entity_id": str(i), "prediction": 0.5} for i, _ in enumerate(rows)
                ]
                return ok(
                    {
                        "meaning": "probability that the order is late",
                        "predictions": preds,
                    }
                )
            if sub == "/predict":
                ids = ["1", "10", "11", "2"]
                needle = body.get("entity_contains")
                hits = [i for i in ids if needle is None or needle in i]
                preds = [
                    {
                        "entity_id": i,
                        "prediction": 1 / (n + 1),
                        "context": {"city": "Milano"},
                    }
                    for n, i in enumerate(hits)
                ][: body["top"]]
                return ok(
                    {
                        "kind": "event",
                        "meaning": "probability that the order is late",
                        "count": len(hits),
                        "predictions": preds,
                    }
                )
            if sub == "/cancel":
                return ok({"cancelled": True})
        return err("NOT_FOUND", f"no route {m} {path}", 404)

    def _spec(self):
        return {
            "id": f"s{self._spec_version}",
            "task": "binary_classification",
            "target": {
                "name": "late",
                "definition": "Order delivered after promised date",
            },
            "features": [
                {"name": "f1"},
                {"name": "f2", "enabled": False},
                {"name": "f3"},
            ],
            "leakage_exclusions": [
                {"column": "delivered_at", "reason": "dated after the outcome"}
            ],
        }

    def _run(self, state=None):
        st = state or self.run_states[self._run_i]
        if self.fail_run and st == "SUCCEEDED":
            return {
                "id": "r1",
                "state": "FAILED",
                "error_code": "BUILD_LIMIT_REACHED",
                "error_message": "Free plan: 2 of 2 model builds used.",
                "suggestions": ["add_own_key", "retry"],
            }
        return {
            "id": "r1",
            "state": st,
            "opportunity_title": "Late orders",
            "kind": "event",
            "meaning": "probability that the order is late",
        }
