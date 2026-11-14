# Changelog

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
