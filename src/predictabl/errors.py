"""Errors raised by the Predictabl client.

Every error carries the server's machine `code` (when it sent one) and a
message meant for a person: the API phrases refusals as "what happened and
what to do", and these classes pass that through untouched.
"""

from __future__ import annotations


class PredictablError(Exception):
    """Base class for everything this package raises."""

    def __init__(
        self, message: str, *, code: str | None = None, status: int | None = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}" if self.code else self.message


class AuthenticationError(PredictablError):
    """The API key is missing, wrong or revoked (HTTP 401)."""


class PermissionDenied(PredictablError):
    """The API key is valid but its scope does not allow this (HTTP 403).
    Scoring-only (`inference`) keys cannot upload, plan or train."""


class NotFound(PredictablError):
    """No such object for this account (HTTP 404). Someone else's objects
    look exactly like missing ones, by design."""


class LimitReached(PredictablError):
    """A free-plan allowance on Predictabl's AI key is used up (HTTP 402).
    The message says how to continue (your own AI key, or an upgrade)."""


class InvalidRequest(PredictablError):
    """The request was understood but refused (HTTP 4xx), e.g. a model that
    needs rows was asked to score without them."""


class BuildFailed(PredictablError):
    """A step of building a model (reading the data, finding the question,
    planning, training) ended in failure. `suggestions` lists the server's
    recovery options when it gave any."""

    def __init__(self, message: str, *, code: str | None = None, suggestions=None):
        super().__init__(message, code=code)
        self.suggestions = list(suggestions or [])


class ApiError(PredictablError):
    """Anything else the server answered with an error."""
