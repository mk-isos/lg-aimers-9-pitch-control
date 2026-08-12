# EXP-022 Outcome-Taxonomy Multi-Task Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with a fresh verification checkpoint after every task.

**Goal:** Build and evaluate a temporally safe auxiliary-outcome model whose deployable rolling-origin candidate must score at least 1100 Skill in each of 2022, 2023, and 2024 before any final-fit package is allowed.

**Architecture:** A row-order-independent utility reconstructs reverse, middle, ball, and strike labels from same-pitcher same-season cumulative-count transitions. Four fixed HistGradientBoosting classifiers produce prior-season-only OOF probabilities, and a strongly regularized Ridge model predicts residuals on top of frozen EXP-021 strict OOF probabilities. Machine-generated metrics record temporal selection, calibration, segments, and the uniform-1100 gate.

**Tech Stack:** Python 3.12, pandas 3.0.5, NumPy 2.5.1, scikit-learn 1.9.0, standard-library `unittest`.

## Global Constraints

- Use only current-row official inputs and `train.csv`; never aggregate other test rows.
- Derive auxiliary labels only from same-season train rows and never cross a season boundary.
- Do not use the current validation season labels to choose model capacity, correction scale, or calibration.
- Keep EXP-021 strict OOF predictions immutable.
- Predeclare two residual scales only: `0.25` and `0.50`.
- Require Skill `>=1100` in each of 2022, 2023, and 2024 and no regression versus EXP-021 strict.
- Do not create a final model or ZIP unless every gate passes.
- Commit only source, tests, documentation, and `validation_metrics.json`; never commit CSV, model, NPY/NPZ, output, ZIP, or virtual environments.

---

### Task 1: Row-order-independent auxiliary label reconstruction

**Files:**
- Create: `tests/test_outcome_taxonomy_features.py`
- Create: `experiments/outcome_taxonomy_features.py`

**Interfaces:**
- Produces: `reconstruct_outcome_labels(frame: pandas.DataFrame) -> tuple[pandas.DataFrame, dict[str, object]]`
- Produces: `assert_label_reconstruction_invariants(frame: pandas.DataFrame, labels: pandas.DataFrame, diagnostics: dict[str, object]) -> None`
- Label output columns: `aux_success`, `aux_reverse`, `aux_middle`, `aux_ball`, `aux_strike`, `pair_valid` with the original frame index preserved.

- [ ] Write a `unittest.TestCase` with two pitchers, two seasons, known binary count increments, a duplicated successor key, and shuffled input rows.
- [ ] Assert exact recovered labels, exclusion of duplicate successor keys, no cross-season pairing, and identical `row_id`-keyed output after shuffling.
- [ ] Run `python -m unittest tests.test_outcome_taxonomy_features -v` and verify failure because the module does not exist.
- [ ] Implement cumulative count reconstruction with `numpy.rint(n * rate)`, a unique-key lookup on `(pitcher_id, season, asof_pitcher_n)`, and binary-delta validation.
- [ ] Add invariants for binary labels, `aux_success == control_success` on valid pairs, and row-order independence diagnostics.
- [ ] Run the unit test and `python -m py_compile experiments/outcome_taxonomy_features.py` and verify both pass.

### Task 2: Full-train label audit

**Files:**
- Modify: `tests/test_outcome_taxonomy_features.py`
- Reuse: `experiments/outcome_taxonomy_features.py`

**Interfaces:**
- Consumes: `reconstruct_outcome_labels` from Task 1.
- Produces: a deterministic JSON-serializable diagnostic object with per-season valid rows, positive rates, duplicate-key exclusions, invalid deltas, and success mismatch count.

- [ ] Add a smoke test that loads only the required columns from `data/train.csv` when the file exists and checks success mismatch `0`, binary labels, and nonzero valid coverage in every season.
- [ ] Add an explicit permutation test on a fixed 20,000-row stratified subset and compare outputs by `row_id`.
- [ ] Run the full utility test suite and record counts from actual train data.
- [ ] Stop and fix the utility before model work if success mismatch is nonzero or row-order parity fails.

### Task 3: Temporal auxiliary classifiers and frozen base alignment

**Files:**
- Create: `experiments/train_exp022_outcome_taxonomy_multitask.py`

**Interfaces:**
- Consumes: `prepare_data()` from `train_exp017_rolling_residual.py`, `select_stable_features()` and `season_equal_weights()` from `train_exp019_histgb_residual.py`, and Task 1 labels.
- Consumes frozen base arrays: `artifacts/EXP-020/low_rank_pitcher_context_eb/predictions_lowrank_s300_r6_{season}.npy` and `targets_{season}.npy` for 2021–2024. This is the fixed rank-6 candidate packaged as EXP-021 strict, not the diagnostic fold-varying rank path.
- Produces: `build_auxiliary_oof(...) -> tuple[dict[int, numpy.ndarray], dict[str, object]]`, where each season matrix has six columns in the exact order `reverse`, `middle`, `ball`, `strike`, `strike_minus_ball`, `reverse_plus_middle`.

- [ ] Add startup assertions that raw train row order, prepared feature rows, frozen targets, and season masks align exactly.
- [ ] Select the fixed stable feature allow-list and assert no raw player/team IDs, `season`, `row_id`, or target columns are present.
- [ ] For each validation season 2021–2024 and each of four targets, fit the predeclared HistGradientBoostingClassifier on valid rows from seasons strictly earlier than the validation season using season-equal sample weights.
- [ ] Predict every row in the validation season and store auxiliary matrices as ignored NPY files.
- [ ] Record training seasons, training rows, class rates, feature schema, fit time, and prediction range for each target and fold.
- [ ] Run a bounded smoke mode on seasons through 2022 and assert shapes, finite values, `[0,1]` range, and prior-season-only training.

### Task 4: Same-fold diagnostic ceiling and temporal Ridge residual

**Files:**
- Modify: `experiments/train_exp022_outcome_taxonomy_multitask.py`

**Interfaces:**
- Consumes six-column auxiliary OOF matrices and frozen EXP-021 strict base predictions.
- Produces candidate arrays `predictions_temporal_w025_{season}.npy`, `predictions_temporal_w050_{season}.npy`, and `predictions_strict_selected_{season}.npy`.

- [ ] Implement `fit_residual_ridge(source_seasons, validation_season, alpha=5000)` with source-season-centered residuals and source-season-equal weights.
- [ ] Fit scaling statistics only on source seasons and apply them row-wise to validation features.
- [ ] Generate the two fixed correction scales and select the validation-season candidate using prior OOF worst Skill, then mean Skill, then smaller scale.
- [ ] Implement same-fold Ridge and fixed-seed five-fold cross-fitted Ridge diagnostics; mark both nondeployable and prohibit their use in temporal selection.
- [ ] Assert every selected fold uses only selection seasons earlier than the validation fold and that no affine or fixed offset is applied.

### Task 5: Metrics, segments, and machine artifact

**Files:**
- Modify: `experiments/train_exp022_outcome_taxonomy_multitask.py`
- Create by execution: `artifacts/EXP-022/outcome_taxonomy_multitask/validation_metrics.json`

**Interfaces:**
- Produces one machine-generated JSON containing protocol, label audit, model parameters, fold metrics, segment metrics, diagnostic ceilings, temporal selection, QA, and final gate decision.

- [ ] Record Brier, Skill, actual rate, prediction mean, mean gap, calibration slope/intercept for base and every candidate.
- [ ] Record R/F, month, pitcher season-n bins, and pitcher/batter new-status segments using current-row diagnostics only.
- [ ] Record aggregate mean/min/latest Skill and exact Skill-1100 Brier thresholds.
- [ ] Set `uniform_1100_passed` only when all three reported folds have Skill at least 1100, do not regress versus strict, and all temporal/test-independence QA flags pass.
- [ ] Set `final_fit_authorized` equal to `uniform_1100_passed`; the training script must not contain final model or ZIP creation code.

### Task 6: Full rolling execution and completion audit

**Files:**
- Verify: `experiments/train_exp022_outcome_taxonomy_multitask.py`
- Verify: `artifacts/EXP-022/outcome_taxonomy_multitask/validation_metrics.json`

- [ ] Run `python experiments/train_exp022_outcome_taxonomy_multitask.py` to completion.
- [ ] Parse JSON with rejection of NaN/Infinity and independently recalculate each stored Brier from ignored prediction/target arrays.
- [ ] Verify feature and target alignment, prior-season-only training/selection, finite probabilities, and exact candidate formulas.
- [ ] Run `python -m unittest tests.test_outcome_taxonomy_features -v`, `python -m py_compile`, and `git diff --check`.
- [ ] If any 2022–2024 Skill is below 1100, preserve the failed experiment and do not create a ZIP.
- [ ] If all gates pass, write a separate final-fit/package plan before creating deployable artifacts.

### Task 7: Documentation and repository hygiene

**Files:**
- Modify after execution: `README.md`
- Modify after execution: `docs/EXPERIMENT_LOG.md`
- Modify after execution: `docs/LEARNING_LOG.md`

- [ ] Copy only exact machine-recorded metrics from EXP-022 JSON into the experiment table and detailed log.
- [ ] Explain whether the new auxiliary supervision added signal, whether it transferred across time, and whether the 1100 gate passed.
- [ ] Verify no package version changed; leave root `requirements.txt` unchanged unless the actual environment changed.
- [ ] Audit staged paths for CSV, model, NPY/NPZ, output, ZIP, and virtual-environment files.
- [ ] Commit related source, tests, JSON, and documents with an EXP-022-specific message; push only after all checks pass.
