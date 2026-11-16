"""The one place that talks HTTP.

Unwraps the API's `{data, error}` envelope, turns refusals into typed errors,
and retries what is safe to retry (connection failures and 502/503/504 on
reads). Everything else in the package goes through `Transport`.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from . import errors
from ._version import __version__

HOSTED_URL = "https://app.predictabl.tech/pub/v1"
LOCAL_URL = "http://localhost:8000/api/v1"

_RETRY_STATUS = {502, 503, 504}


def resolve_base_url(base_url: str | None) -> str:
    url = base_url or os.environ.get("PREDICTABL_BASE_URL") or HOSTED_URL
    return url.rstrip("/")


class Transport:
    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        *,
        timeout: float = 120.0,
        client_tag: str | None = None,
        http: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        agent = f"predictabl-python/{__version__}"
        if client_tag:
            agent += f" ({client_tag})"
        headers = {"User-Agent": agent, "Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = http or httpx.Client(timeout=timeout)
        self._headers = headers

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------ calls

    def get(self, path: str, **params: Any) -> Any:
        params = {k: v for k, v in params.items() if v is not None}
        return self._request("GET", path, params=params or None, retry=True)

    def post(self, path: str, json: Any = None, **kw: Any) -> Any:
        return self._request("POST", path, json=json, **kw)

    def put(self, path: str, json: Any = None) -> Any:
        return self._request("PUT", path, json=json)

    def patch(self, path: str, json: Any = None) -> Any:
        return self._request("PATCH", path, json=json)

    def delete(self, path: str) -> Any:
        return self._request("DELETE", path)

    def _request(self, method: str, path: str, *, retry: bool = False, **kw: Any) -> Any:
        url = f"{self.base_url}{path}"
        attempts = 4 if retry else 1
        last: Exception | None = None
        for i in range(attempts):
            try:
                resp = self._http.request(method, url, headers=self._headers, **kw)
            except httpx.TransportError as exc:
                last = exc
                if i + 1 < attempts:
                    time.sleep(min(2**i, 8))
                    continue
                raise errors.ApiError(
                    f"Could not reach Predictabl at {self.base_url} ({exc})."
                ) from exc
            if resp.status_code in _RETRY_STATUS and i + 1 < attempts:
                time.sleep(min(2**i, 8))
                continue
            try:
                return _unwrap(resp)
            except errors.AuthenticationError as exc:
                raise errors.AuthenticationError(
                    _auth_hint(bool(self.api_key), exc.message),
                    code=exc.code,
                    status=exc.status,
                ) from None
        raise errors.ApiError(str(last))  # pragma: no cover - loop always returns


def _unwrap(resp: httpx.Response) -> Any:
    try:
        body = resp.json()
    except ValueError:
        body = None
    if resp.is_success and isinstance(body, dict) and body.get("error") is None:
        return body.get("data") if "data" in body else body
    code, message = _error_of(body, resp)
    status = resp.status_code
    cls = {
        401: errors.AuthenticationError,
        402: errors.LimitReached,
        403: errors.PermissionDenied,
        404: errors.NotFound,
    }.get(status)
    if cls is None:
        cls = errors.InvalidRequest if 400 <= status < 500 else errors.ApiError
    raise cls(message or f"HTTP {status}", code=code, status=status)


def _error_of(body: Any, resp: httpx.Response) -> tuple[str | None, str]:
    """The API answers `{data: null, error: {code, message}}`; FastAPI's own
    refusals (auth, scope) answer `{detail: ...}`. Accept both."""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return err.get("code"), str(err.get("message") or "")
        detail = body.get("detail")
        if isinstance(detail, str):
            return None, detail
        if detail is not None:
            return None, str(detail)
    text = (resp.text or "").strip()
    return None, text[:500]


def _auth_hint(key_sent: bool, server_said: str) -> str:
    """A 401 always says what to do: the server's own words are terse."""
    said = f" (server: {server_said})" if server_said else ""
    if not key_sent:
        return (
            "No API key. Create one in the web app under Settings → API & MCP, "
            "then `export PREDICTABL_API_KEY=pk_...` or call "
            'pb.login("pk_...").' + said
        )
    return (
        "The API key was refused" + said + ". Check it was copied in full "
        "(it starts with pk_) and has not been revoked under "
        "Settings → API & MCP."
    )
