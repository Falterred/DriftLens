# DriftLens v0.1 benchmark fixtures

- `generator.py` builds seeded corpus pairs (`generate_corpus_pair`) and writes
  them as JSONL/JSON/CSV/Parquet (`write_corpus_as`).
- `embedding_fixtures.py` builds FAISS/Chroma fixtures for the adapter tests.
- `adapter_fixtures/` holds one subfolder per input format for on-disk fixtures
  that need to be checked in; most fixtures are generated at runtime instead.

The default pytest run uses the fast 1K tier. Larger 10K/100K/1M tiers are gated
behind the `benchmark_scale` marker:

    pytest -m benchmark_scale
