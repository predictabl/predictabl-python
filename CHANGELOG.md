# Changelog

## 0.2.1

- `predictabl mcp` message: any MCP app connects by adding the address and
  signing in (the token-header example is gone).

## 0.2.0

- Removed the local MCP server (`predictabl mcp`). Predictabl's MCP server is
  hosted at https://app.predictabl.tech/mcp and does everything from the chat:
  upload data, ask a question, train, get predictions. Connect it from
  ChatGPT, Claude, Claude Code, Cursor or any MCP client
  (https://predictabl.tech/docs/agents/). `predictabl mcp` now prints how.
  The `mcp` extra is empty and kept only so old install commands still work.

## 0.1.5

- README and the local MCP server's instructions point to the hosted MCP
  server at https://app.predictabl.tech/mcp: connect it from ChatGPT, Claude or
  any MCP client to upload data, ask a question, train and get predictions
  from the chat. The local server (`predictabl mcp`) still reads and predicts
  only; no other code change.

## 0.1.4

- Package metadata only, no code change: clearer PyPI description, more
  keywords and classifiers, the author named, the Documentation link goes to
  predictabl.tech/docs and a Changelog link is added.
- README: one opening line on what Predictabl is; `to_pandas()` notes that it
  needs `pip install "predictabl[pandas]"`.

## 0.1.3

- A refused API key now says what to do: with no key set, how to create one and
  set `PREDICTABL_API_KEY`; with a wrong or revoked key, how to check it. (It
  used to say only `unauthenticated`.)
- README: plans and limits described in credits, as on the server.

## 0.1.2

- Fix: `model.report()` (and so every `train()` or `predict()` that waits for
  the result) crashed with `TypeError: unhashable type: 'dict'` against the
  current server, which sends the headline metric as an object. It now reads
  it, and the report includes `test_score`.
- Docstrings speak of credits and allowances.

## 0.1.1

- `connect()` replaces `upload()`: pass files or a DataFrame as before, or a
  Postgres address (`"postgresql://user:pw@host/db"`), which Predictabl reads
  in place, read-only. The password never appears in progress or errors.
  `upload()` still works and warns; it will be removed in a later version.
- Predictions print readably: what the number means, then the numbers.
  One prediction prints on one line, e.g.
  `Probability that customer churned: 0.82`.
- README: churn example with its output; token access is *Full access* or
  *Predictions only*.

## 0.1.0

First release.

- `upload()` files or a DataFrame; `Data.predict(question)` builds a tested
  model in one call with step-by-step progress; `Data.plan()`,
  `Plan.refine()`, `Plan.train()` to review first; `pb.plan()` to retrain an
  existing plan.
- `Model.predict()` scores handed-in rows (batched) or the model's own data
  (`ids`, `since`/`until`, `since_days`, `as_of`, `limit`, `filters`,
  `include`). Predictions say what they mean and, for forecasts, the date
  they are made from.
- Typed errors with the server's code and message.
- `predictabl mcp`: an MCP server for AI agents over the same client.
