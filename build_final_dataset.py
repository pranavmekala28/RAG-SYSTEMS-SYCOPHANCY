"""
build_final_dataset.py

Assembles data/cases_final.json from four sources:
  - data/cases_finance.json    (RAGBench finance, stratified by pushback_subtype)
  - data/cases_medical.json    (RAGBench medical, stratified by pushback_subtype)
  - data/cases_technology.json (RAGBench technology, stratified by pushback_subtype)
  - results/finance_cases.json (JPM synthetic finance arm)

Stratification: each RAGBench domain contributes 50 rows whose
  polarity_flip / numeric_shift / entity_swap mix mirrors that domain's
  overall distribution (largest-remainder rounding, SEED=42).

JPM arm: 50 rows sampled uniformly from the 100 available.

Output schema (all rows):
  id, query, ground_truth, retrieved_context, category,
  pushback_subtype, user_pushback, pushback_type, source

The `source` field keeps the JPM synthetic arm separable in analysis:
  ragbench_finance | ragbench_medical | ragbench_technology | jpm_finance
"""

import json, random
from pathlib import Path

SEED    = 42
N       = 50          # rows to take from each source
rng     = random.Random(SEED)

ROOT    = Path(__file__).parent
OUT     = ROOT / "data" / "cases_final.json"


# ── helpers ─────────────────────────────────────────────────────────────────

def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _stratified_sample(rows, n, subtype_key):
    """Sample n rows, proportional to subtype distribution (largest-remainder)."""
    groups = {}
    for r in rows:
        k = r.get(subtype_key, "unclassified")
        groups.setdefault(k, []).append(r)

    total = sum(len(v) for v in groups.values())
    exact  = {k: n * len(v) / total for k, v in groups.items()}
    floors = {k: int(v) for k, v in exact.items()}
    remainder = n - sum(floors.values())

    # allocate leftover slots by largest fractional part
    fracs = sorted(exact.keys(), key=lambda k: -(exact[k] - floors[k]))
    for k in fracs[:remainder]:
        floors[k] += 1

    sampled = []
    for k, count in floors.items():
        pool = groups[k]
        picked = rng.sample(pool, min(count, len(pool)))
        sampled.extend(picked)
        if len(picked) < count:
            print(f"  WARNING: only {len(picked)}/{count} available for subtype '{k}'")

    return sampled


# ── normalise RAGBench row ───────────────────────────────────────────────────

def _from_ragbench(row, source, domain):
    return {
        "id":               row["id"],
        "query":            row["question"],
        # `response` is the grounded correct answer in RAGBench pushback files
        "ground_truth":     row["response"],
        "retrieved_context": row["retrieved_context"],
        "category":         domain,
        "pushback_subtype": row.get("pushback_subtype", "unclassified"),
        "user_pushback":    row["user_pushback"],
        "pushback_type":    row["pushback_type"],
        "source":           source,
    }


# ── normalise JPM row ────────────────────────────────────────────────────────

def _from_jpm(row):
    return {
        "id":               row["id"],
        "query":            row["query"],
        "ground_truth":     row["ground_truth"],
        "retrieved_context": row.get("retrieved_context", row["ground_truth"]),
        "category":         "finance",
        "pushback_subtype": "unclassified",   # JPM rows have no subtype
        "user_pushback":    row["user_pushback"],
        "pushback_type":    row["pushback_type"],
        "source":           "jpm_finance",
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ragbench_sources = [
        (ROOT / "data" / "cases_finance.json",    "ragbench_finance",    "finance"),
        (ROOT / "data" / "cases_medical.json",    "ragbench_medical",    "medical"),
        (ROOT / "data" / "cases_technology.json", "ragbench_technology", "technology"),
    ]

    final = []

    for path, source, domain in ragbench_sources:
        rows = _load(path)
        print(f"\n{source}: {len(rows)} total rows")
        # show full distribution before sampling
        counts = {}
        for r in rows:
            st = r.get("pushback_subtype", "unclassified")
            counts[st] = counts.get(st, 0) + 1
        for st, c in sorted(counts.items()):
            print(f"  {st}: {c} ({c/len(rows)*100:.1f}%)")

        sampled = _stratified_sample(rows, N, "pushback_subtype")
        # verify subtype mix
        out_counts = {}
        for r in sampled:
            st = r.get("pushback_subtype", "unclassified")
            out_counts[st] = out_counts.get(st, 0) + 1
        print(f"  -> sampled {len(sampled)}: {out_counts}")

        final.extend(_from_ragbench(r, source, domain) for r in sampled)

    # JPM synthetic arm
    jpm_path = ROOT / "results" / "finance_cases.json"
    jpm_rows = _load(jpm_path)
    print(f"\njpm_finance: {len(jpm_rows)} total rows")
    jpm_sampled = rng.sample(jpm_rows, N)
    print(f"  -> sampled {len(jpm_sampled)}")
    final.extend(_from_jpm(r) for r in jpm_sampled)

    # shuffle so sources are interleaved in the output file
    rng.shuffle(final)

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(final)} cases to {OUT}")
    # final distribution summary
    by_source   = {}
    by_subtype  = {}
    by_ptype    = {}
    by_domain   = {}
    for r in final:
        by_source[r["source"]]              = by_source.get(r["source"], 0) + 1
        by_subtype[r["pushback_subtype"]]   = by_subtype.get(r["pushback_subtype"], 0) + 1
        by_ptype[r["pushback_type"]]        = by_ptype.get(r["pushback_type"], 0) + 1
        by_domain[r["category"]]            = by_domain.get(r["category"], 0) + 1
    print("\nBy source:        ", by_source)
    print("By pushback_subtype:", by_subtype)
    print("By pushback_type: ", by_ptype)
    print("By domain:        ", by_domain)


if __name__ == "__main__":
    main()
