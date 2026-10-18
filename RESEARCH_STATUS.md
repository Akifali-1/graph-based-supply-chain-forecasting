# Research status and unresolved details

## Current conference-paper campaign

The current paper evaluation is restricted to **ISOMORPH** and
**SupplyGraph**; Favorita is excluded from the final comparison. Existing Drive
checkpoints were evaluated with forward passes only. No training epochs,
hyperparameter searches, or Job C scripts were run for this extraction.

Verified published artifacts are in `paper_outputs/`:

- unified validation/test metrics and mean/sample-standard-deviation summaries;
- Job B real-vs-permuted paired tests;
- horizon, node-role/sparsity, learned-gate, and compute-cost tables;
- window-level significance and rigorous time-aware statistics;
- 300-DPI PNG/PDF paper figures.

The six Job B permuted STGT runs are complete on Drive. The separate
`job_b_pipeline_real_supplygraph_v2` seed-42 reproduction gate remains
interrupted with no final result row, so it is explicitly recorded as **not
passed** rather than silently inferred from another run.

Drive tokens, OAuth client files, checkpoints, raw data, and downloaded private
artifacts are not repository contents. The repository includes only portable
code, reviewed aggregate outputs, documentation, and tests.

## Historical initial-pipeline evidence

## Evidence available

- Supplied Favorita archives and small CSV metadata schemas.
- Supplied Colab STGT implementation and a few diagnostic predictions.
- Supplied optimized BiLSTM/GAT/adaptive-fusion reference code.
- Local source code and synthetic/static verification.
- CPU-only smoke verification with PyTorch 2.6.0 and PyG 2.6.1: Adaptive STGT
  forward shape/batch isolation and backward gradient-flow checks passed. This
  did not load Favorita data or submit a training run.
- Read-only `rclone` audit of Drive artifacts: the processed Favorita version is
  `db31efd8c97ba7d5`; the registry contains exactly two completed seed-42 runs,
  `stgt-6addfa8ebc70-attempt1` and `gat_lstm-b815f36bced9-attempt1`. Their
  manifests, checkpoint metadata, validation histories, and frozen final-test
  predictions/report were independently checked. STGT selected epoch is 85;
  GAT-LSTM selected epoch is 34 after 49 completed epochs.
- User-supplied Colab histories: STGT through 100 epochs and GAT-LSTM through
  at least 37 epochs. These are exploratory normalized validation-loss traces,
  not independently retrieved/verified full comparison artifacts.
- Locally audited Git state: no registry, checkpoint, processed-data manifest,
  training log, or final-test output is present in this checkout. The local
  `results/` and `data/processed/` directories contain only Git keep-markers.

## Evidence not yet available

- A specifically identified original STGT paper and verified protocol.
- Authorized Drive access and verified uploaded dataset in this session.
- Adaptive STGT, STGT seed-123/2026, and GAT-LSTM seed-123/2026 artifacts.
- Multi-seed results and ablations.

The proposed model and baseline are labeled transparently. The current GAT-LSTM
attention placement follows the supplied code (temporal pooling before GAT and
fusion), not an invented post-fusion sequence. The original paper's dataset
statistics and claimed results are not reused as results from this experiment.

## Historical initial experiment plan

1. Push the dedicated branch after user GitHub authentication.
2. Authorize Colab Drive mount and upload all source archives.
3. Pass full synthetic tensor/checkpoint tests in Colab.
4. Prepare/hash-validate versioned Favorita aggregates on Drive.
5. Audit any existing registry/checkpoints/logs and reuse only exact, verified
   model/seed artifacts.
6. Run the missing members of the three-model (`stgt`, `adaptive_stgt`,
   `gat_lstm`) × three-seed (`42`, `123`, `2026`) campaign sequentially;
   select checkpoints using validation only.
7. Inspect validation evidence; no test-driven tuning.
8. Freeze one completed run ID per model and seed and authorize the final-test
   cell.
9. Report individual-seed and mean/std metrics, runtime, parameters, selected
   epochs, and failures. Further ablations require explicit configs inside the
   remaining budget.

Search is deliberately disabled pending complete validation evidence. No new
GPU experiment has been submitted by this agent. The verified Drive campaign
has two submitted runs and its persisted deadline has expired; it must not be
reset. A new campaign registry is required before the seven missing runs can
be submitted. The approved campaign paths are
`experiment_registry_adaptive_stgt_campaign.json`,
`experiments/adaptive_stgt_campaign/`, and
`experiments/adaptive_stgt_campaign/final_test/`. No dataset, credential, or
generated model artifact belongs in GitHub.

## Limitations to disclose

- Global family nodes overlap geographic aggregates and are not store-specific
  leaves. Store→Family is not a strict conservation relation.
- Missing ledger cells are zero recorded sales under an explicit assumption;
  they are not estimates of unobserved demand or out-of-stock sales.
- The baseline's horizon extension and proposed multi-step output are adaptations,
  not verified reproduction of a published experiment.
- Epoch parity does not guarantee equal compute; time/parameter counts are reported.
- GPU determinism and Drive locking across separate sessions are not guaranteed.
- Partial epoch recovery replays that epoch in a new budget-counted retry.
- Final evaluation failure preserves logs/selection. Explicit recovery verifies
  completed stages, resumes missing work and never replaces completed results;
  see `RELIABILITY_REVIEW.md`.
