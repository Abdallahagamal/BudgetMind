from __future__ import annotations

import argparse
import csv
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELED = REPO_ROOT / "Sprint1" / "Label" / "labeled_dataset.csv"
DEFAULT_CLEAN = REPO_ROOT / "Sprint1" / "classification_dataset" / "clean_dataset.csv"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent

REQUIRED_FIELDS = (
    "task_id",
    "task_text",
    "source",
    "date_collected",
    "type",
    "complexity",
    "domain",
)

# Canonical labels from Taxonomy_Labeling_Guidelines.md, plus the
# compact spellings already used in labeled_dataset.csv.
TYPE_CANONICAL = {
    "factual q&a": "Factual Q&A",
    "translation": "Translation",
    "summarization": "Summarization",
    "coding & debugging": "Coding & Debugging",
    "mathematical reasoning": "Mathematical Reasoning",
    "analytical / logical reasoning": "Analytical/Logical Reasoning",
    "analytical/logical reasoning": "Analytical/Logical Reasoning",
    "architecture & system design": "Architecture & System Design",
    "creative & open-ended generation": "Creative & Open-Ended Generation",
}

COMPLEXITY_CANONICAL = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
}

DOMAIN_CANONICAL = {
    "software engineering & technology": "Software Engineering & Technology",
    "mathematics & quantitative": "Mathematics & Quantitative",
    "language & communication": "Language & Communication",
    "general knowledge / cross-domain": "General Knowledge/Cross-Domain",
    "general knowledge/cross-domain": "General Knowledge/Cross-Domain",
}

# Majority TYPE expected from how Member 2 built the dataset.
# A mismatch is a warning, not an automatic error.
SOURCE_EXPECTED_TYPE = {
    "truthfulqa": {"Factual Q&A"},
    "natural questions (nq)": {"Factual Q&A"},
    "wmt translation": {"Translation"},
    "cnn_dailymail": {"Summarization"},
    "xsum": {"Summarization"},
    "google-research-datasets (mbpp)": {"Coding & Debugging"},
    "swe-bench": {"Coding & Debugging"},
    "openai/gsm8k": {"Mathematical Reasoning"},
    "mmlu": {"Analytical/Logical Reasoning", "Factual Q&A"},
    "big-bench-hard": {"Analytical/Logical Reasoning"},
    "devpkprajapati/ai-system-design-instruct": {"Architecture & System Design"},
    "hld-bench": {"Architecture & System Design", "Analytical/Logical Reasoning"},
    "hugging face creative writing datasets": {"Creative & Open-Ended Generation"},
}

TYPE_HINTS = (
    ("Translation", re.compile(r"\b(translate|translation|into (french|german|spanish|arabic|japanese))\b", re.I)),
    ("Summarization", re.compile(r"\b(summarize|summarise|summary|tl;dr|tldr)\b", re.I)),
    ("Coding & Debugging", re.compile(r"\b(python|javascript|sql|function|debug|refactor|nullpointer|indexerror)\b", re.I)),
    ("Mathematical Reasoning", re.compile(r"\b(solve for|how many|how much did|compute|algebra)\b", re.I)),
    ("Architecture & System Design", re.compile(r"\b(design a|architecture|microservices|high availability|sharding|distributed)\b", re.I)),
    ("Creative & Open-Ended Generation", re.compile(r"\b(brainstorm|write a (haiku|limerick|tagline|toast|launch email)|opening paragraph)\b", re.I)),
)

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def normalize_space(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def canon(mapping: dict[str, str], value: str) -> str | None:
    key = normalize_space(value).lower()
    return mapping.get(key)


def tokenize(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text) if len(t) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({k: (v or "").strip() if isinstance(v, str) else v for k, v in row.items()})
        return rows


def add_issue(issues: list[dict[str, str]], task_id: str, severity: str, check: str, detail: str) -> None:
    issues.append(
        {
            "task_id": task_id,
            "severity": severity,
            "check": check,
            "detail": detail,
        }
    )


def validate(
    labeled_path: Path,
    clean_path: Path,
    near_dup_threshold: float,
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    labeled = load_csv(labeled_path)
    issues: list[dict[str, str]] = []

    missing_cols = [c for c in REQUIRED_FIELDS if labeled and c not in labeled[0]]
    if missing_cols:
        add_issue(issues, "", "error", "schema", f"Missing columns: {missing_cols}")

    ids = [r.get("task_id", "") for r in labeled]
    blank_ids = [i for i, t in enumerate(ids, start=2) if not t]
    for _ in blank_ids:
        add_issue(issues, "", "error", "missing_task_id", "Row has empty task_id")

    dup_ids = [i for i, n in Counter(ids).items() if i and n > 1]
    for task_id in dup_ids:
        add_issue(issues, task_id, "error", "duplicate_task_id", "task_id appears more than once")

    type_counts: Counter[str] = Counter()
    complexity_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    source_type: dict[str, Counter[str]] = defaultdict(Counter)

    texts_norm: dict[str, list[str]] = defaultdict(list)
    tokens_by_id: dict[str, set[str]] = {}

    for row in labeled:
        task_id = row.get("task_id", "")
        text = row.get("task_text", "")
        source = row.get("source", "")
        type_raw = row.get("type", "")
        complexity_raw = row.get("complexity", "")
        domain_raw = row.get("domain", "")

        source_counts[source or "<empty>"] += 1

        if not text:
            add_issue(issues, task_id, "error", "missing_text", "task_text is empty")

        type_c = canon(TYPE_CANONICAL, type_raw) if type_raw else None
        complexity_c = canon(COMPLEXITY_CANONICAL, complexity_raw) if complexity_raw else None
        domain_c = canon(DOMAIN_CANONICAL, domain_raw) if domain_raw else None

        if not type_raw:
            add_issue(issues, task_id, "error", "missing_label", "type is empty")
        elif type_c is None:
            add_issue(issues, task_id, "error", "invalid_type", f"Not in taxonomy: {type_raw!r}")
        else:
            type_counts[type_c] += 1
            if type_raw != type_c:
                add_issue(
                    issues,
                    task_id,
                    "warning",
                    "type_spelling",
                    f"{type_raw!r} should be written {type_c!r} for file consistency",
                )

        if not complexity_raw:
            add_issue(issues, task_id, "error", "missing_label", "complexity is empty")
        elif complexity_c is None:
            add_issue(issues, task_id, "error", "invalid_complexity", f"Not in taxonomy: {complexity_raw!r}")
        else:
            complexity_counts[complexity_c] += 1

        if not domain_raw:
            add_issue(issues, task_id, "error", "missing_label", "domain is empty")
        elif domain_c is None:
            add_issue(issues, task_id, "error", "invalid_domain", f"Not in taxonomy: {domain_raw!r}")
        else:
            domain_counts[domain_c] += 1
            if domain_raw != domain_c:
                add_issue(
                    issues,
                    task_id,
                    "warning",
                    "domain_spelling",
                    f"{domain_raw!r} should be written {domain_c!r} for file consistency",
                )

        if type_c:
            source_type[source][type_c] += 1
            expected = SOURCE_EXPECTED_TYPE.get(source.lower())
            if expected and type_c not in expected:
                add_issue(
                    issues,
                    task_id,
                    "warning",
                    "source_type_mismatch",
                    f"source={source!r} is usually {sorted(expected)}; labeled {type_c}",
                )

            for hinted_type, pattern in TYPE_HINTS:
                if pattern.search(text) and type_c != hinted_type:
                    add_issue(
                        issues,
                        task_id,
                        "warning",
                        "keyword_type_mismatch",
                        f"text looks like {hinted_type} but labeled {type_c}",
                    )
                    break

            if type_c == "Architecture & System Design" and complexity_c == "Low":
                add_issue(
                    issues,
                    task_id,
                    "warning",
                    "complexity_unlikely",
                    "Architecture tasks are rarely Low",
                )
            if type_c == "Translation" and complexity_c == "High":
                add_issue(
                    issues,
                    task_id,
                    "warning",
                    "complexity_unlikely",
                    "Straight translation is rarely High unless extra judgment is required",
                )

        key = normalize_space(text).lower()
        if key:
            texts_norm[key].append(task_id)
        tokens_by_id[task_id] = tokenize(text)

    for text, group in texts_norm.items():
        unique_ids = sorted(set(group))
        if len(unique_ids) > 1:
            for task_id in unique_ids:
                add_issue(
                    issues,
                    task_id,
                    "error",
                    "exact_duplicate_text",
                    f"Same task_text as {', '.join(unique_ids)}",
                )

    # Near-duplicates: 540 rows, pairwise Jaccard is cheap.
    by_id = {r.get("task_id", ""): r for r in labeled}
    items = [(tid, toks) for tid, toks in tokens_by_id.items() if toks]
    near_pairs: list[tuple[str, str, float]] = []
    for i, (id_a, toks_a) in enumerate(items):
        for id_b, toks_b in items[i + 1 :]:
            score = jaccard(toks_a, toks_b)
            if score >= near_dup_threshold:
                near_pairs.append((id_a, id_b, score))
                labels_a = by_id[id_a]
                labels_b = by_id[id_b]
                same_labels = (
                    labels_a.get("type") == labels_b.get("type")
                    and labels_a.get("complexity") == labels_b.get("complexity")
                    and labels_a.get("domain") == labels_b.get("domain")
                )
                severity = "info" if same_labels else "warning"
                check = "near_duplicate" if same_labels else "inconsistent_near_duplicate"
                add_issue(
                    issues,
                    id_a,
                    severity,
                    check,
                    f"Similar to {id_b} (Jaccard={score:.2f}); "
                    f"{id_a}=({labels_a.get('type')}/{labels_a.get('complexity')}/{labels_a.get('domain')}) "
                    f"{id_b}=({labels_b.get('type')}/{labels_b.get('complexity')}/{labels_b.get('domain')})",
                )

    if clean_path.exists():
        clean = load_csv(clean_path)
        labeled_ids = {r.get("task_id") for r in labeled if r.get("task_id")}
        clean_ids = {r.get("task_id") for r in clean if r.get("task_id")}
        only_labeled = sorted(labeled_ids - clean_ids)
        only_clean = sorted(clean_ids - labeled_ids)
        for task_id in only_labeled:
            add_issue(issues, task_id, "warning", "id_not_in_clean", "Present in labeled file but not in clean_dataset.csv")
        for task_id in only_clean:
            add_issue(issues, task_id, "error", "id_missing_from_labeled", "Present in clean_dataset.csv but not labeled")

        clean_text = {r["task_id"]: normalize_space(r.get("task_text", "")) for r in clean if r.get("task_id")}
        for row in labeled:
            tid = row.get("task_id", "")
            if tid in clean_text:
                if normalize_space(row.get("task_text", "")) != clean_text[tid]:
                    add_issue(
                        issues,
                        tid,
                        "warning",
                        "text_changed_from_clean",
                        "task_text differs from Member 2 clean_dataset.csv",
                    )
    else:
        add_issue(issues, "", "warning", "clean_dataset_missing", f"Not found: {clean_path}")

    # Reproducible stratified sample for the part a script cannot do.
    by_type: dict[str, list[str]] = defaultdict(list)
    for row in labeled:
        t = canon(TYPE_CANONICAL, row.get("type", "")) or "<unlabeled>"
        if row.get("task_id"):
            by_type[t].append(row["task_id"])
    rng = random.Random(seed)
    sample: list[str] = []
    types = sorted(by_type)
    if types:
        base = max(1, sample_size // len(types))
        for t in types:
            pool = list(by_type[t])
            rng.shuffle(pool)
            take = min(len(pool), base)
            sample.extend(pool[:take])
        leftover = [tid for t in types for tid in by_type[t] if tid not in sample]
        rng.shuffle(leftover)
        sample.extend(leftover[: max(0, sample_size - len(sample))])
        sample = sample[:sample_size]

    stats = {
        "n_labeled": len(labeled),
        "type_counts": dict(type_counts),
        "complexity_counts": dict(complexity_counts),
        "domain_counts": dict(domain_counts),
        "source_counts": dict(source_counts),
        "source_type": {s: dict(c) for s, c in source_type.items()},
        "near_pairs": len(near_pairs),
        "sample": sample,
        "error_count": sum(1 for i in issues if i["severity"] == "error"),
        "warning_count": sum(1 for i in issues if i["severity"] == "warning"),
        "info_count": sum(1 for i in issues if i["severity"] == "info"),
    }
    return labeled, issues, stats


def counts_table(counts: dict[str, int]) -> str:
    if not counts:
        return "_none_"
    lines = ["| Label | Count |", "|---|---:|"]
    for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {k} | {n} |")
    return "\n".join(lines)


def write_outputs(
    out_dir: Path,
    labeled: list[dict[str, str]],
    issues: list[dict[str, str]],
    stats: dict,
    labeled_path: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    issues_path = out_dir / "validation_issues.csv"
    sample_path = out_dir / "manual_review_sample.csv"
    report_path = out_dir / "validation_report.md"

    with issues_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "severity", "check", "detail"])
        writer.writeheader()
        writer.writerows(issues)

    by_id = {r["task_id"]: r for r in labeled}
    with sample_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(REQUIRED_FIELDS) + ["review_note"])
        writer.writeheader()
        for task_id in stats["sample"]:
            row = dict(by_id[task_id])
            row["review_note"] = ""
            writer.writerow({k: row.get(k, "") for k in list(REQUIRED_FIELDS) + ["review_note"]})

    n = stats["n_labeled"]
    min_type = min(stats["type_counts"].values()) if stats["type_counts"] else 0
    report = f"""# Labeled dataset validation report

Source file: `{labeled_path}`
Rows: **{n}** (plus header). Member 4 should treat this as the full labeled set to check, not 541 separate manual reads.

## What this script can and cannot do

| Automatable | Still needs a human |
|---|---|
| Missing labels | Subtle wrong TYPE/COMPLEXITY on an otherwise well-formed row |
| Values outside the taxonomy | Tie-break cases (code review vs architecture) |
| Exact duplicates | Whether a near-duplicate *should* be kept (WMT / architecture variants) |
| Near-duplicates with different labels | Gold-example style judgment |
| Source vs typical TYPE | Final accept/reject of a warning |
| Class imbalance counts | Whether to collect more data |

## Status

- Errors: **{stats['error_count']}** (must fix before splitting)
- Warnings: **{stats['warning_count']}** (send to Member 1 / 3)
- Info: **{stats['info_count']}**
- Near-duplicate pairs (Jaccard ≥ threshold): **{stats['near_pairs']}**

See `validation_issues.csv` for every flagged `task_id`.

## TYPE

{counts_table(stats['type_counts'])}

Smallest TYPE class: **{min_type}**. If this is below ~15, stratified train/val/test is required and that class will be weak.

## COMPLEXITY

{counts_table(stats['complexity_counts'])}

## DOMAIN

{counts_table(stats['domain_counts'])}

## SOURCE

{counts_table(stats['source_counts'])}

## Manual review sample

`manual_review_sample.csv` is a **seeded stratified sample** of {len(stats['sample'])} rows (by TYPE).
Open it, apply the taxonomy gold examples / tie-breaks, and mark `review_note`.
That is the only way to catch labels that look legal but are wrong.
"""
    report_path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate BudgetMind labeled classification dataset (Member 4).")
    p.add_argument("--labeled", type=Path, default=DEFAULT_LABELED)
    p.add_argument("--clean", type=Path, default=DEFAULT_CLEAN)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--near-dup-threshold", type=float, default=0.75)
    p.add_argument("--sample-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    labeled, issues, stats = validate(
        args.labeled,
        args.clean,
        args.near_dup_threshold,
        args.sample_size,
        args.seed,
    )
    write_outputs(args.out_dir, labeled, issues, stats, args.labeled)
    print(f"Rows: {stats['n_labeled']}")
    print(f"Errors: {stats['error_count']}  Warnings: {stats['warning_count']}  Info: {stats['info_count']}")
    print(f"Wrote: {args.out_dir / 'validation_report.md'}")
    print(f"Wrote: {args.out_dir / 'validation_issues.csv'}")
    print(f"Wrote: {args.out_dir / 'manual_review_sample.csv'}")
    return 1 if stats["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
