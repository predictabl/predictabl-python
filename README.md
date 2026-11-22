# Predictabl Python API library

[Predictabl](https://predictabl.tech) is your agentic data scientist for tabular
data: it reads your tables, plans the features, trains and tests the model, and
serves the predictions.

Turn a question about your data into a trained, tested model, then score with
it. The features are planned in SQL you can review, and checked for leakage
before training.

**Documentation: [predictabl.tech/docs](https://predictabl.tech/docs/)** · for AI
agents, the whole documentation as one Markdown file:
[predictabl.tech/docs/llms-full.txt](https://predictabl.tech/docs/llms-full.txt)

> **Status: alpha (0.1).** The API may still change between minor versions.

```bash
pip install predictabl
```

```python
import predictabl as pb

pb.login("pk_...")          # or set PREDICTABL_API_KEY

data = pb.connect("customers.csv")      # a file, a DataFrame, or a database
churn_model = data.predict("Which customers will churn?")
```

```
✓ Uploading      customers.csv · 18.2 MB
✓ Reading data   1 table
✓ Understanding  Predict which customers will churn
✓ Planning       31 features · 2 excluded
✓ Building data
✓ Training       lightgbm: kept 840 trees
✓ Final model    auc 0.8712
```

```python
churn_model.predict(customer)        # one customer (a dict), a list, or a DataFrame
```

```text
Probability that customer churned: 0.82
```

```python
churn_model.report()                 # what it predicts, tested score, top features
churn_model.url                      # the same model in the web app
```

Your data can be files, a pandas DataFrame, or a Postgres database, which
Predictabl reads in place, read-only:

```python
data = pb.connect("postgresql://readonly:pw@db.acme.com/sales")
```

Everything you build from code also appears in the web app, labelled with the
token that created it.

## Get an API key

In the web app: **Settings → API & MCP → Create token**.

- By default a token has **full access**: it can do from code what you do in the
  web app (upload, ask, plan, train and get predictions).
- Tick **Predictions only** for a read-only token: it can get predictions from
  your trained models and change nothing.

A token only ever reaches its owner's data and models. It cannot create other
tokens. Revoking it stops it immediately.

## Review the plan before training

`data.predict(question)` goes straight from question to model. To look first:

```python
plan = data.plan("Which customers will stop buying in the next 30 days?")
print(plan)                     # target, features, and what was excluded and why
plan = plan.refine("Don't use the support_tier column")
model = plan.train()            # or plan.train(preset="fast_preview") for a 15-second check
```

## Scoring

How a model picks what to score depends on its kind (`model.kind`):

| kind | what `model.predict()` does | typical model |
|---|---|---|
| `flat` | needs `rows=`: one record in, one prediction out | survey, sensor reading, form |
| `event` | scores rows dated **today** in the data it trained on; pick a window with `since=`/`until=` or `since_days=` | orders, sign-ups, bookings |
| `forecast` | projects from the latest data, or `as_of="2026-10-01"` | sales per store, demand per item |

```python
preds = model.predict(rows=df)                     # your records
preds = model.predict(since_days=7)                # event model: last week's rows
preds = model.predict(ids=["store_12"], limit=48)  # one entity
preds = model.predict(filters=[{"column": "city", "op": "eq", "value": "Milano"}],
                      include=["city"])

preds.meaning    # e.g. "probability that customer churned"
preds.as_of      # forecasts: the date they are made from
preds.note       # e.g. why nothing matched, and the newest date in the data
preds.to_pandas()     # needs: pip install "predictabl[pandas]"
```

A record you score can still contain its outcome column (e.g. a row copied from
the training file): it never reaches the model, and comes back as `actual` so
the call doubles as a check.

## Plans and limits

On Predictabl's AI key, building uses credits: turning a question into a plan,
and changing a plan. Training, retraining and predictions never use credits, and
a plan or model that fails to build is refunded. With your own OpenAI or
Anthropic key (Settings → AI provider) building uses no credits. Your plan also
sets how many models you keep and predictions a month. `pb.account()` shows
what is left; running out raises `predictabl.LimitReached` with what to do next.
Details: [Errors and limits](https://predictabl.tech/docs/errors-and-limits/).

## Use it from an AI agent (MCP)

**From ChatGPT or Claude, no install:** add `https://app.predictabl.tech/mcp` as
a connector and sign in. Then upload your data, ask your question, and get
your predictions, all in the chat. Setup for each app:
[AI agents and MCP](https://predictabl.tech/docs/agents/).

**On your machine:** this library also includes a local MCP server that lists
your models, shows how good they are, and gets predictions from them, using a token:

```bash
claude mcp add predictabl-local --env PREDICTABL_API_KEY=pk_... -- uvx --from "predictabl[mcp]" predictabl mcp
```

Then ask: *"Which of my customers are most likely to churn this month, and how
good is that model?"* To upload or train from a chat, use the hosted server.

## Configuration

| | |
|---|---|
| `PREDICTABL_API_KEY` | your token (`pk_...`) |
| `PREDICTABL_BASE_URL` | default `https://app.predictabl.tech/pub/v1`; a local instance is `http://localhost:8000/api/v1` |

For several accounts or instances, use clients directly:
`pb.Client(api_key=..., base_url=...)` has the same methods.

## Errors

All errors derive from `predictabl.PredictablError` and carry the server's
`code` and a message that says what to do: `AuthenticationError` (bad or
revoked key), `PermissionDenied` (the key's scope does not allow it),
`NotFound`, `LimitReached`, `InvalidRequest`, `BuildFailed` (with the server's
recovery `suggestions`).

## License

MIT
