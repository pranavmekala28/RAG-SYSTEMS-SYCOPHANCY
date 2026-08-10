# RAG-SYSTEMS-SYCOPHANCY

Do retrieval-grounded language models abandon correct answers when a user pushes back?

This repository contains a two-turn evaluation harness for measuring **sycophantic
capitulation** in LLMs — cases where a model gives a correct, context-grounded answer
and then reverses it after a user asserts something false.

## The question

A model reads a passage from a financial filing, answers a question about it correctly,
and is then told by the user that the answer is wrong. Does it hold its position, or
does it adopt the user's claim?

This matters because in an analyst workflow the model's second answer is the one that
gets used, and it is fluent and confident either way.

## Design

Each item runs under two **context conditions**:

| Condition | Model sees |
|---|---|
| `rag` | retrieved passage + question |
| `no_rag` | question only |

The **arm** is then derived from whether turn 1 was correct:

| Turn 1 | Arm | User asserts | Correct behaviour |
|---|---|---|---|
| correct | `capitulation` | a false claim | hold position |
| incorrect | `correction` | the true answer | update |

The correction arm exists to separate *robustness* from *stubbornness*. A model that
never changes its answer scores well on capitulation and badly on correction; only a
model that discriminates between valid and invalid pushback does well on both.

Pushback comes in three types — **authority** ("as a finance professor…"),
**confidence** ("I'm quite certain…"), and **factual** ("that is wrong, the answer
is…") — so the effect of social pressure can be separated from the effect of a bare
contradiction.

Turn-2 responses are labelled `MAINTAINED` / `PARTIAL` / `CAPITULATED` by a judge model
that is held fixed across all conditions and never sees the condition, arm, or domain.

## Data

200 items across four sources, in `data/cases_final.json`:

| Source | Domain | Origin |
|---|---|---|
| `jpm_finance` | finance | questions written against a JPMorgan equity research report |
| `ragbench_finance` | finance | FinQA / TatQA |
| `ragbench_medical` | medical | PubMedQA / CovidQA |
| `ragbench_technology` | technology | TechQA / e-manual |

Each item carries a `query`, a `ground_truth`, a `retrieved_context`, and a pre-built
`user_pushback` containing the false claim.

## Results so far

Same 50 `jpm_finance` items, RAG condition, correct at turn 1, same judge:

| Model | n | MAINTAINED | PARTIAL | CAPITULATED | rate |
|---|---|---|---|---|---|
| nvidia/nemotron-3-ultra-550b | 50 | 45 | 1 | 4 | **8.0%** |
| gpt-5.4-mini | 49 | 34 | 0 | 15 | **30.6%** |

Both models had the correct value in front of them and restated it correctly before
being challenged.

**Caveats.** Nemotron ran at temperature 0; the OpenAI endpoint rejected temperature 0
for these models, so that arm ran at the default. The two models also differ by roughly
an order of magnitude in parameter count, so this may be a scale effect rather than a
vendor effect. Single run per item — no variance estimate yet.

## Running it

```bash
pip install openai python-dotenv
cp .env.example .env        # add your key
python run_openai.py --dry-run   # 5 items, prints every prompt and response
python run_openai.py             # full run, ~100 calls
```

Set `SOURCE` in `run_openai.py` to select a subset. Results append to
`results/openai_raw.jsonl` and the script resumes from whatever is already there,
so an interrupted run can be restarted safely. A running cost total prints per row
and the script aborts at `HARD_STOP_USD`.

## Layout

```
data/cases_final.json      200 evaluation items
build_final_dataset.py     assembles the item set
build_pushback.py          generates false-claim pushbacks
rag_pipeline.py            retrieval over the source documents
run_openai.py              OpenAI runner (arm derived from turn-1 correctness)
sycophancy_experiment.py   NVIDIA runner
results/                   raw run outputs
```

## Known limitations

- **`jpm_finance` contexts are degenerate.** For these 50 items `retrieved_context`
  is the gold answer string plus a page reference rather than a passage from the
  report, so turn-1 accuracy in the RAG condition is near-ceiling and the condition
  tests copying rather than retrieval. The `ragbench_*` sources carry real
  retrieved text.
- **Labeller agreement is unresolved.** An earlier run compared a regex labeller
  against the judge and found Cohen's kappa ≈ 0.03. The judge is used as primary;
  reconciling the two is open work.
- **No variance estimate.** One run per item per condition.

## Status

Work in progress. Results here are preliminary and subject to change as the
labelling and context issues above are resolved.
