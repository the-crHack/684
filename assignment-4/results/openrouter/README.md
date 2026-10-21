# OpenRouter measurement records

`predictions.jsonl` is append-only. A request's last record is its current outcome;
earlier connection/capacity failures remain for provenance. Request hashes cover
the actual prompt, model, parameters, document, and endpoint. API keys, HTTP
authorization headers, and private reasoning traces are never written here.

`summary.json` and `metrics.csv` are rebuilt by:

```bash
python analyze.py --config config.openrouter.json --input results/openrouter/predictions.jsonl --output-dir results/openrouter
```

The summary distinguishes the 46-call primary study from the optional 256-call
grid. The primary study tests three model sizes on all ten originals, improvement
on the compact model, two paired injection cases, and a two-case effort probe.
One authority-claim pilot is outside the primary study. Models that failed access
checks are retained in the raw log but excluded from the configured experiment.
The public catalog snapshot and the report explain why replacements were needed.

The free account was observed to allow 50 requests per day. Some early requests
failed before reaching a model, and an interrupted pilot may consume quota without
yielding a recorded answer. The raw log therefore is not an account billing ledger.
Success and token counts must be read from actual provider usage fields; hypothetical
or failed requests are not model judgments. The full grid is optional and should
not be described as complete while its summary status is partial.
