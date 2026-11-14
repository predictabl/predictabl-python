"""Predictabl from Python.

    import predictabl as pb

    pb.login("pk_...")                      # or set PREDICTABL_API_KEY
    data  = pb.upload("orders.csv")
    model = data.predict("Will an order arrive late?")
    model.report()
    model.predict(rows=new_orders)

Module-level functions use one shared client, configured by `login()` or the
PREDICTABL_API_KEY / PREDICTABL_BASE_URL environment variables. For several
accounts or instances at once, create `predictabl.Client` objects instead.
"""

from __future__ import annotations

from ._progress import Update
from ._version import __version__
from .client import Client, Data, Model, Plan, Predictions
from .errors import (
    ApiError,
    AuthenticationError,
    BuildFailed,
    InvalidRequest,
    LimitReached,
    NotFound,
    PermissionDenied,
    PredictablError,
)

__all__ = [
    "__version__",
    "login",
    "client",
    "account",
    "upload",
    "data",
    "datasets",
    "model",
    "models",
    "Client",
    "Data",
    "Plan",
    "Model",
    "Predictions",
    "Update",
    "PredictablError",
    "AuthenticationError",
    "PermissionDenied",
    "NotFound",
    "LimitReached",
    "InvalidRequest",
    "BuildFailed",
    "ApiError",
]

_default: Client | None = None


def login(api_key: str | None = None, base_url: str | None = None, **kw) -> Client:
    """Configure the shared client. Without arguments, reads
    PREDICTABL_API_KEY (and PREDICTABL_BASE_URL) from the environment."""
    global _default
    if _default is not None:
        _default.close()
    _default = Client(api_key=api_key, base_url=base_url, **kw)
    return _default


def client() -> Client:
    """The shared client, created from the environment on first use."""
    global _default
    if _default is None:
        _default = Client()
    return _default


def account() -> dict:
    return client().account()


def upload(source, name: str | None = None, **kw) -> Data:
    return client().upload(source, name, **kw)


def data(data_id: str) -> Data:
    return client().data(data_id)


def datasets() -> list[Data]:
    return client().datasets()


def model(model_id: str) -> Model:
    return client().model(model_id)


def models(limit: int = 50) -> list[Model]:
    return client().models(limit)
