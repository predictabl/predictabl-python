# Changelog

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
