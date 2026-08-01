# A2A Invoice Action Agent

A FastAPI-based A2A 1.0 invoice reconciliation agent.

The agent:

- receives invoice batches,
- uses Gemini to select one action per invoice package,
- returns evidence-backed proposals,
- waits for result continuations,
- executes only accepted proposals,
- stores tasks in SQLite,
- supports retries and idempotency,
- isolates tasks by Bearer token,
- supports cancellation.

## Supported actions

- `settle_invoice`
- `request_approval`
- `hold_invoice`
- `reject_duplicate`
- `open_exception`

## Project files

```text
app.py
ai.py
database.py
models.py
storage.py
agent_card.json
requirements.txt
runtime.txt
render.yaml
README.md