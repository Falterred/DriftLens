# DriftLens v0.1 benchmark fixtures

- `generator.py` builds seeded corpus pairs (`generate_corpus_pair`) and writes
  them as JSONL/JSON/CSV/Parquet (`write_corpus_as`).
- `generator.py` also plants known duplicate groups (`generate_duplicate_corpus`)
  and retrieval eval fixtures (`generate_retrieval_eval_fixture`) for the
  Release-1.0 benchmark suites.
- `benchmark_results.json` records the wall-clock time of the 1M-scale tier when
  it is run with `pytest -m benchmark_scale`; no numeric time bound is asserted.
- `embedding_fixtures.py` builds FAISS/Chroma fixtures for the adapter tests.
- `adapter_fixtures/` holds one subfolder per input format for on-disk fixtures
  that need to be checked in; most fixtures are generated at runtime instead.

The default pytest run uses the fast 1K tier. Larger 10K/100K/1M tiers are gated
behind the `benchmark_scale` marker:

    pytest -m benchmark_scale

The release gate in `tests/test_release_0_1_gate.py` cross-checks this fixture
set against the full `Code` enum and imports every release-0.1 module, and runs
the full suite once more with `pytest.main` (disable the nested run with
`DRIFTLENS_SKIP_NESTED_SUITE=1`).
