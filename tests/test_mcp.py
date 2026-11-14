"""The MCP server: every tool is registered and calls the API it says."""

from __future__ import annotations

import asyncio

import httpx
import pytest

pytest.importorskip("mcp")

import predictabl as pb  # noqa: E402
from predictabl.mcp import build_server  # noqa: E402

from .fake_api import FakeApi  # noqa: E402


def _server():
    api = FakeApi()
    client = pb.Client(
        api_key="pk",
        base_url="http://test/api/v1",
        poll_interval=0,
        _http=httpx.Client(transport=api.transport()),
        client_tag="mcp",
    )
    return build_server(client), api


def test_all_tools_are_registered():
    srv, _ = _server()
    names = {t.name for t in asyncio.run(srv.list_tools())}
    assert names == {
        "list_models",
        "get_model",
        "get_forecast",
        "top_risks",
        "top_forecasts",
        "get_backtest",
        "predict",
        "predict_rows",
        "list_connections",
        "account",
    }


def test_predict_tool_maps_its_arguments_to_the_api():
    srv, api = _server()
    asyncio.run(
        srv.call_tool(
            "predict",
            {
                "run_id": "r1",
                "date_from": "2026-01-01",
                "date_to": "2026-01-31",
                "as_of": "2026-02-01",
                "top": 5000,
                "entity_contains": "1",
            },
        )
    )
    method, path, body, headers = api.calls[-1]
    assert (method, path) == ("POST", "/runs/r1/predict")
    assert body["since"] == "2026-01-01" and body["until"] == "2026-01-31"
    assert body["cutoff"] == "2026-02-01" and body["top"] == 2000
    assert "(mcp)" in headers["user-agent"]


def test_predict_rows_tool_respects_the_server_cap():
    srv, api = _server()
    asyncio.run(
        srv.call_tool("predict_rows", {"run_id": "r1", "rows": [{"x": 1}] * 1500})
    )
    body = api.calls[-1][2]
    assert len(body["rows"]) == 1000


def test_list_models_only_lists_trained_ones():
    srv, _ = _server()
    out = asyncio.run(srv.call_tool("list_models", {}))
    text = str(out)
    assert "r1" in text and "Late orders" in text
