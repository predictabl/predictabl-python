"""Predictabl as an MCP server, so an AI agent (Claude Code, Claude Desktop, any
MCP client) can discover your models and score with them.

    claude mcp add predictabl --env PREDICTABL_API_KEY=pk_... -- uvx "predictabl[mcp]" mcp

It is a thin layer over the same client as the Python API: every tool is one
or two API calls, so what an agent sees is exactly what your code would get,
and a token reaches only its owner's models. The tools read and score; they
never upload or train.
"""

from __future__ import annotations

import base64
from pathlib import Path

from .client import Client

INSTRUCTIONS = (
    "Predictabl hosts this user's trained prediction models. Start with "
    "list_models, check get_model before relying on a model, then use predict "
    "for anything about what happens next. Quote `meaning` with numbers so the "
    "user knows what a score is, and state `as_of` for forecasts."
)


def _icons():
    try:
        from mcp.types import Icon

        raw = (Path(__file__).parent / "assets" / "logo-128.png").read_bytes()
        uri = "data:image/png;base64," + base64.b64encode(raw).decode()
        return [Icon(src=uri, mimeType="image/png", sizes=["128x128"])]
    except Exception:  # noqa: BLE001 - branding must never block startup
        return None


def _server_class():
    """mcp 2.x calls the high-level server MCPServer; 1.x called it FastMCP.
    Same decorator API for what this module uses (tool(), run())."""
    try:
        from mcp.server.mcpserver import MCPServer

        return MCPServer
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP

        return FastMCP
    except ImportError as exc:  # pragma: no cover - exercised by users only
        raise SystemExit(
            "The MCP server needs the `mcp` extra: pip install 'predictabl[mcp]'"
        ) from exc


def build_server(client: Client):
    """The FastMCP server with every tool bound to `client`."""
    Server = _server_class()
    t = client._t
    try:
        mcp = Server("predictabl", instructions=INSTRUCTIONS, icons=_icons())
    except TypeError:  # an mcp release without server icons
        mcp = Server("predictabl", instructions=INSTRUCTIONS)

    @mcp.tool()
    def list_models(limit: int = 50) -> list[dict]:
        """List trained models (successful training runs), newest first. Each
        entry is one model you can inspect or score with; use its `id` with the
        other tools. Returns: id, predicting (what it predicts), connection,
        engine, headline metric (+ value), starred, started_at."""
        out = []
        for r in t.get("/runs") or []:
            if r.get("state") != "SUCCEEDED":
                continue
            hm = r.get("headline_metric") or {}
            out.append(
                {
                    "id": r["id"],
                    "predicting": r.get("opportunity_title"),
                    "connection": r.get("connection_name"),
                    "engine": r.get("engine_used") or r.get("engine_requested"),
                    "metric": hm.get("name"),
                    "metric_value": hm.get("value"),
                    "starred": bool(r.get("starred")),
                    "started_at": r.get("started_at"),
                }
            )
            if len(out) >= limit:
                break
        return out

    @mcp.tool()
    def get_model(run_id: str) -> dict:
        """Detailed card for one model: what it predicts and what its number
        MEANS, how it picks what to score (`kind`: forecast | event | flat),
        the winning engine and its tested metric, rows trained/tested and the
        top features. Check this before quoting a model's predictions."""
        return client.model(run_id).report()

    @mcp.tool()
    def get_forecast(run_id: str, limit: int = 100) -> dict:
        """The forecasts this model produced AT TRAINING TIME: one prediction
        per entity at the most recent training cutoff (a validated backtest
        window, usually in the past). Returns as_of, total count and up to
        `limit` items. For the genuine future ("next 7 days"), use `predict`."""
        fc = t.get(f"/runs/{run_id}/forecasts")
        if not fc:
            raise RuntimeError(
                f"no forecasts for run {run_id} (not a forecasting model?)"
            )
        items = (fc.get("items") or [])[:limit]
        return {
            "as_of": fc.get("score_at"),
            "count": fc.get("count"),
            "showing": len(items),
            "items": items,
        }

    @mcp.tool()
    def top_risks(run_id: str, n: int = 10, order: str = "desc") -> dict:
        """The N entities a CLASSIFIER scored highest (or lowest) on its
        held-out test: "the top 10 customers at risk of churn". Each item has
        `score` and `actual` (the held-out truth), so you can see whether the
        top of the ranking was right. For forecasting models use
        top_forecasts."""
        return t.get(f"/runs/{run_id}/top-scored", top=n, order=order)

    @mcp.tool()
    def top_forecasts(run_id: str, n: int = 10, order: str = "desc") -> dict:
        """The top (or bottom) N entities by predicted value, ranked
        server-side across the ENTIRE roster: exact "which store is highest".
        `order`: "desc" (default) or "asc". Forecasting models only."""
        fc = t.get(f"/runs/{run_id}/forecasts", top=n, order=order)
        if not fc:
            raise RuntimeError(
                f"no forecasts to rank for run {run_id}: only per-step forecasting "
                "models have them. For a classifier use top_risks."
            )
        return fc

    @mcp.tool()
    def get_backtest(run_id: str, entity: str | None = None) -> dict:
        """A forecasting model's held-out backtest for one entity: predicted vs
        ACTUAL over the horizon, on data it did not train on. Omit `entity` for
        the default one. Use it as evidence the model works."""
        meta = t.get(f"/runs/{run_id}/forecast/meta")
        if not meta:
            raise RuntimeError(
                f"no backtest panel for run {run_id} (not a per-step forecast?)"
            )
        ent = entity or meta.get("default_entity")
        series = t.get(f"/runs/{run_id}/forecast", entity=ent) if ent else None
        points = []
        for p in (series or {}).get("points", []):
            actual = (
                p.get("actual_bt")
                if p.get("actual_bt") is not None
                else p.get("actual_hist")
            )
            if p.get("pred_bt") is None and actual is None:
                continue
            points.append(
                {"date": p.get("date"), "actual": actual, "predicted": p.get("pred_bt")}
            )
        return {
            "model": run_id,
            "entity": ent,
            "cadence": meta.get("cadence"),
            "horizon": meta.get("horizon"),
            "origin_date": (series or {}).get("origin_date"),
            "points": points,
        }

    @mcp.tool()
    def predict(
        run_id: str,
        since_days: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        as_of: str | None = None,
        filters: list[dict] | None = None,
        include: list[str] | None = None,
        top: int = 20,
        entity_contains: str | None = None,
        calibrated: bool = False,
    ) -> dict:
        """Score with a trained model ON FRESH DATA: features are rebuilt from
        the data as of now, by the model itself. THE tool for "what happens
        next".

        - Forecast models: project from the latest data (or `as_of`,
          YYYY-MM-DD). `entity_id` is `entity·cutoff·step`; step 1 is the
          first future period. The response's `as_of` is the date forecast
          FROM; say it, so stale data is visible.
        - Event/risk models: score rows dated TODAY by default; widen with
          `since_days`, or pick a window with `date_from`/`date_to`. If nothing
          matched, the response has a `note` with the newest date in the data.
        - Flat models (no time axis) cannot be scored this way: use
          predict_rows with the records.

        `filters`: [{"column","op","value"}], op eq|ne|in|contains|gt|gte|lt|lte,
        on the model's table or a table it joins (e.g. city). `include`: extra
        columns to return per prediction (in `context`). `entity_contains`:
        only entities whose id contains this text. Results are ranked by value
        and capped at `top`. Always read `meaning`: it says what the number is.
        For MULTICLASS models read `predicted_class`, not `prediction`."""
        body: dict = {"top": max(1, min(int(top), 2000)), "calibrated": calibrated}
        if since_days is not None:
            body["since_days"] = since_days
        if date_from:
            body["since"] = date_from
        if date_to:
            body["until"] = date_to
        if as_of:
            body["cutoff"] = as_of
        if filters:
            body["filters"] = filters
        if include:
            body["include"] = include
        if entity_contains:
            body["entity_contains"] = entity_contains
        return t.post(f"/runs/{run_id}/predict", json=body)

    @mcp.tool()
    def predict_rows(run_id: str, rows: list[dict]) -> dict:
        """Score records you HAND IN: "would a machine with this torque and
        temperature fail?". Each row is a dict keyed by the model's SOURCE
        columns (the raw columns of its table), without the outcome. The
        model's own feature SQL is applied, so there is no train/serve skew.

        Best for flat models (one row in, one prediction). If the response has
        a `warning`, the model summarises history and only your rows were
        available: prefer `predict` for those. At most 1000 rows per call."""
        return t.post(f"/runs/{run_id}/predict-records", json={"rows": rows[:1000]})

    @mcp.tool()
    def list_connections() -> list[dict]:
        """List the datasets (uploads and connected databases) on this account."""
        return [
            {
                "id": c["id"],
                "name": c.get("name"),
                "kind": c.get("kind"),
                "tables": c.get("table_count"),
            }
            for c in t.get("/connections") or []
        ]

    @mcp.tool()
    def account() -> dict:
        """The account behind this key: email, plan, builds used and left."""
        return client.account()

    return mcp


def serve(base_url: str | None = None) -> None:
    """Run over stdio until the client disconnects."""
    client = Client(base_url=base_url, client_tag="mcp")
    build_server(client).run()
