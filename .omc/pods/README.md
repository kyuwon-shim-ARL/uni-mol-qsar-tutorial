# Pod incident log

Every GPU pod launch attempt — successful, failed, or stockout — must
result in exactly one YAML file in this directory.

## File naming

```
{YYYYMMDD}_{purpose-slug}_{outcome}.yaml
```

Examples:
- `20260526_qsar-tut-a40.yaml` — first successful CO-ADD reproduction
- `20260529_qsar-braf-a40_stockout.yaml` — failed to acquire any GPU

## Minimum fields per entry

```yaml
target:           # what was the GPU supposed to run
intended_gpu:     # which GPU class was requested
approved_budget:  # $ cap the user explicitly approved
actual_cost_incurred: # $ actually billed (0 if pod creation failed)
attempts_total:   # how many create_pod calls were issued
result_summary:   # one line — success/partial/stockout/abort
```

## Why this matters

The EXPECTED_OUTPUTS.md preflight checklist requires explicit cost
confirmation before pod creation. The incident log is the proof that
the protocol was followed (or where it failed). Auditors and the next
operator both rely on this file.

## Cost-log linkage

The cumulative `.omc/cost-log.csv` should have one row per pod attempt
(success or failure). Cross-reference:

- pod file → fine-grained context, failure matrix, decision rationale
- cost log → cumulative spend, easy aggregation

Never delete pod files. They are the most reliable record of what
actually happened on shared infrastructure.
