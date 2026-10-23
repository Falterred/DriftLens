# DriftLens

**An offline "git diff" for RAG knowledge bases and embedding datasets.**

DriftLens compares two versions of a retrieval dataset and tells you *what changed and where* — added, removed, modified or replaced documents; metadata and schema drift; embedding-space shifts; duplicate growth — without calling any model, LLM, or network service.

> **What DriftLens is not:** a quality judge. A low drift score does not mean your retrieval answers are good, and high drift does not prove they are bad. DriftLens measures *structural change* between two datasets. Retrieval metrics (recall@k, MRR) are computed only when you supply an eval set, and they measure ranking behavior, not answer correctness.

## Why

When a RAG knowledge base is re-crawled, re-chunked, or re-embedded, everything downstream silently shifts: chunks vanish, duplicates creep in, embeddings change under a new model version. DriftLens turns that silent shift into a reviewable, gateable diff so you can decide whether to re-index, alert, or roll back.

## Product design

Three interactive diagrams (dark/light themes, zoom, pan; fully offline single-file HTML) explain how DriftLens is put together. Every node links to the source file and lines it describes.

| Diagram | What it answers | File |
|---|---|---|
| System architecture | What are the pieces (CLI, adapters, identity, analysis, reports) and how do they connect? | [docs/diagrams/driftlens-architecture.html](docs/diagrams/driftlens-architecture.html) |
| compare pipeline | What are the seven stages a comparison passes through, and where can it fail or suspend? | [docs/diagrams/driftlens-compare-pipeline.html](docs/diagrams/driftlens-compare-pipeline.html) |
| Data flow | How do two dataset versions become snapshots, reports, and a CI decision? | [docs/diagrams/driftlens-dataflow.html](docs/diagrams/driftlens-dataflow.html) |

### Design decisions that shape usage

- **Snapshots are the contract.** `driftlens snapshot` freezes a dataset into parquet + a manifest with a content-addressed `snapshot_id`. Compare accepts raw sources or snapshot dirs; snapshots make runs reproducible and diffable over time.
- **Identity before analysis.** A five-level identity model (explicit ids → document-id metadata → content hashes) decides what "the same document" means before any drift is measured. A renamed id is ADDDED + REMOVED, never MODIFIED.
- **Compatibility gating is explicit.** If the embedding model or space changed between versions, affected analyses are SKIPPED with a reason code instead of silently comparing incompatible vectors.
- **One process, seven dimensions.** Content, chunking, metadata, duplication, embeddings (MMD), clusters, and retrieval run as guarded stages; a failure names its stage.
- **Determinism is a feature.** Given the same inputs, seed, and flags, outputs are byte-identical (use `--no-timestamp` for CI diffing). Sampling is seeded and reported in the metrics it affects.
- **Honest scope.** DriftLens measures structural change. Retrieval metrics only run with your eval set; a low drift score never certifies answer quality.

## Install

Requires Python 3.10–3.13.

```bash
pip install -e .

# Optional backends and analysis extras
pip install -e ".[chroma]"    # ChromaDB directories (chromadb==1.5.9)
pip install -e ".[faiss]"    # FAISS indexes
pip install -e ".[cluster]"  # clustering analysis (scikit-learn, hdbscan)
```

## Quickstart

Build a snapshot of your dataset (JSONL example):

```bash
driftlens snapshot data/v1.jsonl --out snaps/v1
# Snapshot written to snaps/v1 (120 documents, 120 chunks)
```

Compare two snapshots (or two raw datasets — the CLI accepts both):

```bash
driftlens compare snaps/v1 snaps/v2
```

Output (excerpt):

```
STATUS: UNBASELINED
CONTENT [UNBASELINED]
  content_change_rate            0.3333 [UNBASELINED]
  documents_added                   3
  documents_removed                 2
  documents_modified                1
  documents_replaced                1   <- unrelated text under a known id
  documents_unchanged               4
METADATA [UNBASELINED]
  metadata_js                   0.0290 [UNBASELINED]
  type_changed_fields               1
```

## How change is classified

| Status | Meaning |
|---|---|
| `UNCHANGED` | identical raw content |
| `FORMAT_ONLY_CHANGE` | only whitespace/case/formatting changed |
| `MODIFIED` | small edit: 5-shingle Jaccard similarity ≥ `documents.replace_threshold` (default 0.20) |
| `REPLACED` | unrelated text under the same identity (similarity < 0.20) |
| `ADDED` / `REMOVED` | identity present on only one side |

Identity comes from explicit id columns when present, metadata document-id candidates (`source_id`, `file`, `url`, …), or content hashes as a last resort. Renaming an id is an `ADDED` + `REMOVED` pair, never a `MODIFIED`. See `docs/architecture.md` for the full five-level identity model.

### Tuning and large corpora

```bash
# Bound duplicate-detection cost on large corpora (perm cap reported in metrics)
driftlens compare snaps/v1 snaps/v2 --duplicates-max-units 2000

# Fewer MinHash permutations (A/B-verified metric-identical at 64)
driftlens compare snaps/v1 snaps/v2 --minhash-permutations 64

# Flag chunks outside your target size band (e.g. 512 tokens ±25%)
driftlens compare snaps/v1 snaps/v2 --chunking-target-tokens 512
```

ChromaDB on a locked-down host? Set `DRIFTLENS_CHROMA_WORKDIR` to move the
private DB copy out of `%TEMP%`.

## Reports and CI gating

```bash
driftlens compare snaps/v1 snaps/v2 \
  --json-out report.json \
  --report-out report.html \
  --diff-out diffdir/   # per-doc rows incl. metadata_changed + timestamp_inconsistent flags

# CI mode: exit code reflects your gates
driftlens compare snaps/v1 snaps/v2 --ci --config gates.yml

# Byte-identical JSON reports for diffing in CI (fixed created_at sentinel)
driftlens compare snaps/v1 snaps/v2 --json-out report.json --no-timestamp
```

`--ci` exit codes: `0` all gated dimensions pass · `1` a gated dimension is in WARN · `2` a gated dimension is in FAIL · `3`–`8` fatal errors (invalid input, format, internal, dependency, limit).

Dimensions are gated by `ci.fail_on`; thresholds alone do not gate the exit code. Since 1.1, `--ci` prints a stderr warning whenever a dimension is FAIL but ungated, so a red-but-zero run is never silent. Example `gates.yml`:

```yaml
drift:
  content_change_rate:
    warning: 0.2
    critical: 0.5
ci:
  fail_on:
    content: warning
```

> Note: with default gates, EMBEDDING/SEMANTIC dimensions are SKIPPED for text-only corpora and `ci.on_skipped: warn` makes even an identical-dataset run exit 1. Set `ci.on_skipped: ignore` or gate explicitly to control this.

## Supported inputs

| Format | Notes |
|---|---|
| JSONL / JSON | `id`, `text`, `embedding`, arbitrary metadata fields |
| CSV / Parquet | embedding as JSON string or prefixed columns |
| FAISS + sidecar | `index.faiss` + JSONL sidecar; plain `id` is chunk-level — a document id column (`document_id`, `source_id`, …) is needed for stable document identity |
| ChromaDB | persisted database directory (`chromadb==1.5.9`); telemetry disabled; private copy, never writes to your DB |
| Snapshot dirs | fast, exact re-comparison of previously built snapshots |

## Baselines

```bash
driftlens baseline snap_a snap_b snap_c --list snapshots.txt --out baseline.json
driftlens compare v1 v2 --baseline baseline.json
```

## Privacy

`--redact-ids` replaces every identifier in terminal, JSON, HTML, and diff outputs with a deterministic per-snapshot pseudonym. Raw ids never reach any report surface.

## Documentation

Full documentation lives in `docs/`: [architecture](docs/architecture.md), [CLI reference](docs/cli-reference.md), [configuration](docs/configuration.md), [schema versioning](docs/schema-versioning.md).

## Development

```bash
pip install -e ".[dev]"
python -m pytest                      # full suite
DRIFTLENS_SKIP_NESTED_SUITE=1 \
  python -m pytest --ignore tests/test_release_0_1_gate.py \
  --ignore tests/test_release_1_0_gate.py --ignore tests/test_default_validation.py
# fast core suite (skips the recursive release gates)
```

The test suite is fully offline and deterministic.

## Status

DriftLens 1.1.0 (report schema 1.1). Released under the [MIT License](LICENSE).

---

*DriftLens measures change; it does not judge quality.*
