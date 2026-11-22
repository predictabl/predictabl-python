"""Command line: `predictabl whoami`, `predictabl models`."""

from __future__ import annotations

import argparse
import json
import sys

from . import errors
from ._version import __version__
from .client import Client

MCP_MOVED = """\
Predictabl's MCP server is hosted: https://app.predictabl.tech/mcp
(this local command was removed in predictabl 0.2.0).

  Claude Code:  claude mcp add --transport http predictabl https://app.predictabl.tech/mcp
  Any other app (ChatGPT, Claude, ...): add the address and sign in.

Setup for each app: https://predictabl.tech/docs/agents/"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="predictabl", description="Predictabl from the shell."
    )
    parser.add_argument(
        "--version", action="version", version=f"predictabl {__version__}"
    )
    parser.add_argument(
        "--base-url", help="default: PREDICTABL_BASE_URL or the hosted app"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("whoami", help="your account, plan and what is left")
    sub.add_parser("models", help="your models")
    sub.add_parser("mcp", help="how to connect an AI agent (MCP)")
    args = parser.parse_args(argv)

    if args.cmd == "mcp":
        print(MCP_MOVED, file=sys.stderr)
        return 2

    try:
        with Client(base_url=args.base_url) as c:
            if args.cmd == "whoami":
                print(json.dumps(c.account(), indent=2))
            elif args.cmd == "models":
                for m in c.models():
                    print(f"{m.id}  {m.state:<10} {m.title or ''}")
    except errors.PredictablError as exc:
        print(f"predictabl: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
