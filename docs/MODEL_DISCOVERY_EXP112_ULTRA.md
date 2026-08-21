# EXP-112+ Ultra model discovery and preregistration

Date: 2026-08-21  
Research lock: written before any EXP-112+ model implementation or score was observed  
Public reference: EXP-071 `playerphys_resid_w025`, `1053.8615519684`  
Objective: discover a genuinely new prediction basis with evidence on the order of `1e-4` Brier, not another coefficient or blend tweak.

## 1. Evidence boundary

This reset treats a negative result as evidence only for the exact recipe that was run. It does not turn a small feasibility experiment into a universal stop rule.

- EXP-071 rolling reference is fixed at `0.244900654279 / 1710.91` (2022), `0.247660513870 / 935.79` (2023), and `0.247604672468 / 881.58` (2024).
- EXP-051 remains the conservative local control at `0.244862459180`, `0.247674466216`, and `0.247608042559`. EXP-071 is nevertheless the primary reference because its residualized physical lookup is the verified Public winner.
- EXP-110 is an unsubmitted, mixed-local composition candidate. Its recent pooled gain over EXP-071 is only `6.8061e-7` Brier and is not a new model basis.
- The Public leaderboard is not used to fit, select, rank, stop, or weight any candidate in this document.
- A `+40` Skill-scale jump near the current event rate is approximately a `-1e-4` Brier change. A `1e-6` change is not called a breakthrough.

Frozen EXP-071 assets:

| Asset | SHA256 |
| --- | --- |
| 2022 prediction | `7794481d1f45cb987e104cc3593e8124747bc7aabc8d6bf239ea93cb3675e18c` |
| 2023 prediction | `e302d5cc4d2dd8a16ca4205df24b8763f39f9c381c3344edbe333ba9b0dbb1b3` |
| 2024 prediction | `c98a550d83e4b311d7da13bd59074e977c5f87f408712c40069426a09040ee1d` |
| 2022 target | `40ec827616f5192fb49034ba9299528c3a7c70e31fdb64f1291794839d05a4e8` |
| 2023 target | `7f2117cb4614e23e43bbfef3f52fb8aff65340e4a2b3f90253fd3f23874e1417` |
| 2024 target | `32a5a12d22d5e171e4227ed46fe02a9cd4fac20b0ca4b7618977333df97b238f` |
| validation manifest | `e275db2ac39af70bce5f2dcf4c40ca13f7b75162d0c50619049ffc85666860b9` |

## 2. Repository taxonomy: what is and is not exhausted

| Model space | Repository evidence | Scope-correct conclusion |
| --- | --- | --- |
| Logistic, RF/HGB, XGBoost/LightGBM/CatBoost, ordinary HPO/calibration | EXP-001~019 and later residual variants | Extensively covered; do not repeat ordinary tree/HPO work. |
| Player/team/context EB, recency, AR(1), career/workload, park/regime lookups | EXP-018~021, 027, 037, 063/064, 072~076 | Many plug-in shrinkage recipes failed or mixed; this does **not** test a joint correlated random-slope posterior. |
| Sparse FM | EXP-021 | Only pitcher/batter biases plus rank-2/4 pitcher×fixed-context and batter×fixed-context terms on regular-season rows. It is not all-field FFM, order-3 HOFM, AHOFM, DCNv2, xDeepFM, AutoInt, or AMFormer. |
| Outcome/pitch taxonomy and shallow multi-task | EXP-022~031, 077, 084, 092 preregistration | The information family is heavily tested; a shared neural head alone is not enough novelty to reopen it. |
| TrackMan alignment, moments, PCA, trends, geometry, covariance/GMM, absolute/residual targets | EXP-033~071, 083/084, 096~100 | Marginal and bounded distributional recipes are covered. EXP-071 proves residualized historical physics can transfer, but current-pitch physics remains unavailable. |
| Retrieval | EXP-089/090/093 | Fixed retrieval and one learned-distance recipe failed. True TabR remains unexecuted but has a low prior after these negatives. |
| SSL | EXP-091 | Two short MFM/SCARF recipes failed. T-JEPA remains technically unrun, but it shares the same static-X information ceiling. |
| RealMLP/TabM-mini | EXP-087 | Two small recipes failed; not evidence against explicit cross networks. |
| Foundation models | EXP-094/095/102 | TabICLv2 and original TabPFN-v2 were poor/slow. EXP-102 TabDPT was killed technically before a valid performance test. Newer checkpoints require license and runtime clearance. |
| Temporal | EXP-072/073, 101, 111 | One scalar AR and one cheap frozen GRU-like state were tested. A genuine source-fitted transition model is still distinct; exact-timestamp models are limited by the query contract. |
| Graph | EXP-103 | One cheap typed GraphSAGE route was harmful. HGT/RGCN is unrun but now low-prior. |
| Symbolic | EXP-104 | One bounded grammar found unstable signal; not all nonlinear interaction models are exhausted. |
| Correction composition | EXP-105~110 | Low correction correlation did not create stable target alignment. No more free blend/weight search is allowed. |

## 3. LG Aimers course audit

The full 37-cell hands-on notebook, six Tabular ML PDFs, Mathematics for ML, Supervised Learning, and Time-Series Analysis materials were extracted and inspected. Architecture diagrams were visually checked where layout mattered.

| Course family | Actual repository status | Decision here |
| --- | --- | --- |
| RealMLP + PLE, simplified TabM | Run in EXP-087 and poor | Closed for the same small recipes. |
| ModernNCA and fixed retrieval | Run in EXP-089/090 | True TabR is different, but not a first-line candidate. |
| MFM/SCARF | Run in EXP-091 | T-JEPA is unrun but low-prior. |
| TabICLv2 / original TabPFN-v2 | Run in EXP-094/095 | New checkpoints conditional on license/runtime only. |
| TabDPT | Only invalid 32-row smoke | Performance unknown, deployment currently inadmissible. |
| NPT/SAINT inter-sample attention | Default inference uses other query rows | Disallowed unless reformulated to one query against frozen source context. |
| FT-/TabTransformer | Unrun | Legal within-row control, but less domain-specific than explicit crosses. |
| T2G-Former / ExcelFormer / AMFormer | Unrun | Legitimate within-row interaction alternatives; AMFormer is the strongest deferred neural alternative. |
| True TabR | Unrun | Legal only with a frozen source index and one independent query. |
| Neural ODE/TFT/TimesNet | Unrun | Source-only frozen states are legal, but exact query time/order is absent and EXP-101 lowers the prior. |
| Kernel SVM/RFF | Unrun | Scalable residual kernel is legal but less aligned with the field structure. |

The course audit changes the plan in two ways. First, it confirms that within-row attention is legal while inter-sample attention is not. Second, it prevents a neural reconstructed-outcome head from being mislabeled as a new information source after EXP-022~031, 077, 096/098, and 101.

## 4. 2023–2026 literature and baseball synthesis

The literature search used primary papers and official proceedings. Older sources are included only when they define a family that recent work extends.

- [DCNv2](https://arxiv.org/abs/2008.13535) supplies an explicit bounded-degree all-field cross function with low-rank mixture experts.
- [Higher-order factorization machines](https://papers.nips.cc/paper_files/paper/2016/hash/158fc2ddd52ec2cf54d3c161f2dd6517-Abstract.html) provide auditable order-3 ANOVA interactions; [AHOFM](https://proceedings.mlr.press/v238/ruegamer24a.html) makes nonlinear tensor-product spline interactions scalable.
- [AMFormer](https://ojs.aaai.org/index.php/AAAI/article/view/29033) explicitly separates additive and multiplicative within-row attention. [T2G-Former](https://ojs.aaai.org/index.php/AAAI/article/view/26272) learns a feature-relation graph within each row.
- [Identifiable switching dynamical systems](https://proceedings.mlr.press/v235/balsells-rodas24a.html) support a discrete hidden-regime hypothesis materially different from a scalar AR(1).
- [GPBoost](https://jmlr.org/beta/papers/v23/20-322.html) and recent [scalable crossed/random-slope GLMM work](https://arxiv.org/abs/2403.03007) motivate joint partial pooling instead of independent EB tables.
- [Fishr](https://proceedings.mlr.press/v162/rame22a.html), [REx](https://proceedings.mlr.press/v139/krueger21a.html), and recent [closed-form moment alignment](https://proceedings.mlr.press/v286/chen25f.html) directly target environment-specific sign reversals, but only two strict EXP-071 source environments exist for the 2024 fold.
- Recent baseball work on [release-point variability](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2024.1447665/full), [pitch-type-dependent release mechanics](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2023.1113069/full), and [xCTRL](https://arxiv.org/abs/2508.19184) supports latent mechanics and intent. It also clarifies the ceiling: realized pitch location, catcher target, current pitch type, and current release physics are unavailable at inference, so only a source-derived prior—not a realized-pitch posterior—is legal.

## 5. Model discovery table

Scores are fixed integers from 0 to 5: novelty (`N`), theoretical relevance (`T`), expected upside (`U`), temporal robustness (`R`), expected error diversity (`D`), deployability (`P`), and compute efficiency (`C`). Total is out of 35. “1100 mechanism” answers why a model could plausibly yield about `-1e-4` Brier rather than a cosmetic `+2` Skill.

| Rank | Candidate | Core idea | Closest previous EXP | True novelty and 1100 mechanism | Expected error diversity | Compute | Leakage/deployment risk | N/T/U/R/D/P/C | Total | Decision |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| 1 | All-field order-3 HOFM/AHOFM residual | Explicit pair/triple ANOVA factors; spline factors for continuous histories | EXP-021 | EXP-021 never modeled all fields or an active third-order term. Pitcher×count×hand×reliability interactions can share sparse-cell signal at `1e-4` scale. | Medium-high | Medium | Source-only vocab/knots; deterministic algebra | 5/5/5/4/4/5/4 | **32** | Execute EXP-112 primary/secondary. |
| 2 | DCNv2 all-field residual | Two bounded low-rank cross layers plus small deep tower | EXP-021, 087 | General learned feature crosses are absent. It can share high-order player×state×history effects without enumerating cells. | Medium-high | Medium | ID cold start; exact per-row forward | 5/5/4/4/4/5/4 | **31** | Execute EXP-113. |
| 3 | Switching hierarchical dynamic logit | Sticky 3/4-state source HMM; transition prior plus row-local legal emission | EXP-021 archetype, 072, 101 | A transition-conditioned multimodal posterior is new. Public-positive EXP-072 makes a discontinuous form regime a credible diverse residual source. | High | Medium | Never filter/update on query peers; near-duplicate kill audit | 5/5/4/4/5/4/3 | **30** | Execute EXP-114 after interactions. |
| 4 | Correlated Bayesian random slopes | Joint posterior for pitcher slopes over count/hand/outs/runners/LI, crossed intercepts | EXP-019~021, 027 | Correlated slope posterior and uncertainty sharing are absent. Rare pitcher-context responses can pool jointly instead of through isolated tables. | Medium-high | Medium-high | VI identifiability; zero unseen means | 5/5/4/5/4/4/3 | **30** | Execute EXP-115; intercept-only forbidden. |
| 5 | Fishr season-invariant cross residual | Penalize between-season gradient variance on a cross encoder | EXP-020 GroupDRO | Aligns mechanisms rather than risks/weights and directly attacks sign reversals. A large jump is possible only if the cross signal is stable. | Medium | Medium-high | Too few source environments, penalty instability | 5/5/3/5/4/5/3 | **30** | Defer: 2023 has only one strict EXP-071 source environment. |
| 6 | Anchor/RFF invariant residual | Make residual orthogonal to season shifts in a nonlinear random-feature space | EXP-020, 104 | Explicit perturbation robustness is new, but low capacity makes `1e-4` less credible. | Medium | Low | Training-only anchors; fixed source map | 4/5/3/5/4/5/4 | **30** | Defer. |
| 7 | Structured official-label MMoE | Shared/task experts for control plus reconstructed outcomes | EXP-022~026, 077, 092, 096/098, 101 | Optimization is new but information is not. A `1e-4` jump is possible only through reduced negative transfer, which prior evidence weakens. | Medium | Medium | Reconstruction training-only; exact neural inference | 4/4/3/3/3/4/3 | **24** | Reject as top candidate; closure-only. |
| 8 | Conditional multimodal TrackMan marginalizer | Mixture-density p(Z|X) and deterministic teacher integration | EXP-097, 100 | Nonlinear conditional multimodality is new and could exploit EXP-071’s Public-positive physics mechanism. | High | High | No query Z; alignment/coverage/quadrature cost | 5/5/4/3/5/4/2 | **28** | Reserve after selected four. |
| 9 | Causal TCN frozen pitcher state | Multiscale causal convolutions over source events | EXP-101 | Dilated motifs differ from a GRU and might capture mechanics/fatigue, but query timing is coarse. | High | Medium-high | Frozen cutoff state only | 5/4/4/3/5/4/3 | **28** | Defer after temporal negative. |
| 10 | AutoInt field-token residual | Within-row field attention | EXP-087 | Sparse learned interactions are new and legal, but generic neural evidence is weak. | High | Medium-high | No inter-row attention/batch stats | 5/4/4/3/5/4/3 | **28** | Defer behind auditable crosses. |
| 11 | AMFormer arithmetic residual | Parallel additive/multiplicative attention | EXP-087, 104 | Recent explicit arithmetic bias can model rate×support×context effects. | High | Medium-high | Row-local tokens only; CPU timing | 5/5/4/3/5/4/3 | **29** | Best deferred neural alternative. |
| 12 | T2G-Former | Learned within-row feature-relation graph | EXP-103 | Feature graph is not the failed entity GraphSAGE; can discover structured relations. | High | High | Stabilization and CPU cost | 5/4/4/3/5/3/2 | **26** | Defer. |
| 13 | Sparse representation-level MoE | Row-local experts for game state, history, entity interactions, physics support | EXP-031, 084 | Representation routing differs from pitch-group/source mixtures; `1e-4` requires noncollapsed stable experts. | High | Medium-high | Gate collapse and 2025 extrapolation | 5/4/4/3/5/4/3 | **28** | Defer. |
| 14 | Shape-constrained NAM + interactions | Smooth marginal shapes plus heredity-constrained pairs | EXP-104, trees | Stable nonlinear rate/reliability surfaces are new, but large upside is weak. | Medium | Medium | Pair selection must be nested | 4/4/3/4/4/5/4 | **28** | Diagnostic only. |
| 15 | Stability-selected symbolic residual | Require sign/support stability across source seasons | EXP-104 | Directly repairs EXP-104’s reversal, but multiple-testing risk remains. | Medium | Low | Nested selection only | 3/4/3/4/4/5/5 | **28** | Defer. |
| 16 | S4/LS4 frozen event state | Long-memory structured state-space encoder | EXP-101 | A different sequence function; large jump possible through long regimes, but inference gets only a frozen state. | High | High | No query updates; package/runtime | 5/4/4/3/5/4/2 | **27** | Defer. |
| 17 | Joint pitch-choice/execution latent model | Integrate discrete pitch family and continuous physics jointly | EXP-084, 097 | Earlier tests separated the latents. Joint dependence is new and potentially diverse. | High | High | Current pitch/type/Z forbidden; deterministic integration | 4/5/4/3/5/4/2 | **27** | Reserve. |
| 18 | Bayesian changepoint mixture | Source-only posterior league/player changepoints | EXP-025/026, 037 | Posterior regime uncertainty is new, but 2025 change cannot be observed directly. | Medium | Low-medium | No query-order inference | 4/4/3/3/4/5/4 | **27** | Defer. |
| 19 | HGT/RGCN temporal heterograph | Relation-specific source graph, frozen entity embeddings | EXP-103 | Rich metapaths are new, but the closest graph result was strongly harmful. | High | High | Frozen source graph only | 5/4/3/3/5/4/2 | **26** | Defer. |
| 20 | Conditional privileged projection | Project a current-Z teacher through source-only p(Z|X) | EXP-083, 096/098 | Different from copying teacher probabilities; could be diverse, but missing intent remains dominant. | High | High | Strictly no validation/query Z | 4/5/3/3/5/4/2 | **26** | Reserve. |
| 21 | Hierarchical BART residual | Bayesian trees plus random entity effects | Tree/EB families | Posterior interaction uncertainty is new; scale and overlap reduce expected upside. | Medium | High | Package/posterior determinism | 5/4/3/4/4/3/2 | **25** | Defer. |
| 22 | Sparse variational GP product kernel | Kernels over state, reliability, and entity embeddings | None; ordinary kernels absent | Very different smooth uncertainty, but inducing approximation at this scale weakens the jump case. | High | Very high | Inducing selection/package cost | 5/4/3/4/5/3/1 | **25** | Defer. |
| 23 | T-JEPA full-source pretraining | Predict latent held-out feature subsets | EXP-091 | Exact objective is unrun, but no new test information and prior SSL was poor. | High | High | Source-only corruption/normalization | 5/3/3/3/5/4/2 | **25** | Reject for this tranche. |
| 24 | True TabR residual | Learned source retrieval plus labels and query-neighbor offsets | EXP-089/090 | Full TabR pathway is new; a large jump is possible only if retrieved labels become stable. | High | High | Frozen source index; singleton timing | 4/4/3/3/5/3/2 | **24** | Defer after retrieval negatives. |
| 25 | TabDPT singleton repair | Deterministic one-query adapter | EXP-102 | Performance is unknown, but the current implementation failed the frozen independence gate. | High | Very high | Technical inadmissibility/runtime | 4/4/3/3/5/1/1 | **21** | Do not execute now. |
| 26 | Newer TabPFN 2.5/2.6/3 | Legal cleared checkpoint and fixed source context | EXP-095 | New prior, but v2 was far behind and runtime/license are blocking. | High | Very high | Written license and <600 s proof required | 4/4/3/3/5/1/1 | **21** | Conditional only. |

## 6. Selected portfolio and why it is bounded

Four experiment families are frozen. They represent three independent hypotheses: explicit all-field interaction, discontinuous temporal state, and correlated posterior partial pooling. Two explicit-cross implementations are retained because their algebraic biases differ and because the repository’s prior “FM failed” statement was materially overbroad.

1. **EXP-112 — order-3 HOFM/AHOFM residual**: primary interaction screen; most auditable proof that a third-order term is active.
2. **EXP-113 — DCNv2 all-field residual**: learned bounded cross network; a separate implementation of the interaction hypothesis, not counted twice when deciding how many independent discoveries exist.
3. **EXP-114 — sticky switching pitcher regime**: near-duplicate-risk minimum falsification; must prove that the learned transition changes the query posterior.
4. **EXP-115 — correlated variational random slopes**: intercept-only models are prohibited; novelty depends on active non-intercept pitcher slopes and population covariance.

The shared-output multi-task proposal is not selected. Fishr is not selected because outer 2023 has only one strict EXP-071 residual source environment. The nonlinear TrackMan mixture is the first reserve if the four selected families expose a positive mechanism but fail only on representation capacity.

## 7. Common preregistration

### Validation and target

- Primary baseline: immutable EXP-071 arrays above. EXP-051 is a secondary control.
- Cheap falsification folds: full 2023 and 2024 validation rows.
- Outer 2023 residual training uses only 2022 EXP-071 OOF rows. Outer 2024 uses 2022 and 2023, with equal total source-season weight.
- No EXP-071 residual is invented for 2019–2021. If a family survives to full rolling, 2022 is exactly EXP-071-neutral because no strictly earlier EXP-071 OOF residual exists.
- Neural/factor correction: `c(x)=0.03*tanh(raw(x))`; candidate `p=clip(p071+0.25*c(x),0,1)`. This fixed conservative integration is not tuned.
- All official-row candidates exclude `row_id`, current/future outcome fields, and `season` as a predictive feature. Season may identify a source environment only.
- Categorical vocabularies, numerical imputation/scaling, knots, factors, posteriors, encoders, and state transitions are fit only inside the outer cutoff.
- No outer validation label selects epochs, architecture, rank, state count, prior, or weight.

### Required diagnostics

For every configuration and fold, save:

- Brier, delta Brier, Skill, delta Skill;
- paired loss mean and standard deviation;
- prediction correlation with EXP-071;
- candidate-error correlation with EXP-071 error;
- correction-to-target-residual correlation;
- same-fold one-dimensional correction oracle coefficient and Brier ceiling, marked nondeployable;
- 2,000-draw reconstructed-game block bootstrap with seed `20260821 + season`;
- train and inference runtime, peak resident memory, row count, parameter/state size;
- singleton, canonical batch, reverse, random permutation, split, and duplicate prediction parity.

### Kill, survival, and promotion

Immediate kill occurs for leakage, target/order mismatch, current/query TrackMan use, query-peer aggregation or attention, query-order state updates, batch statistics, nonfinite/out-of-range probabilities, or failed row independence.

The objective’s minimum fast-kill rule is preserved: if both 2023 and 2024 have `delta Brier > +0.0001`, stop. A family is also rejected when it is materially worse and adds no error diversity.

A configuration becomes a cheap survivor only by one of these frozen routes:

- **Tier A route:** both recent folds improve, pooled delta is at most `-1e-5`, and each game-block bootstrap has `P(delta<0) >= 0.60`.
- **Tier B route:** pooled delta is at most `-5e-5`, neither fold is worse than `+2e-5`, and at least one fold reaches `-7.5e-5`.
- **Diversity route:** neither fold is worse than `+1e-4`, candidate error correlation is at most `0.995`, and the nondeployable one-dimensional oracle improves pooled Brier by at least `2e-5`.

At most two configurations advance. If two configurations from one family qualify, the preregistered primary wins; the secondary is reported but does not create a second survivor slot. Full rolling is 2022/2023/2024 with exactly the same model definition and eight fixed epochs where applicable. No ensemble is attempted unless at least two independent new bases survive.

## 8. Family-specific frozen definitions

### EXP-112 — HOFM/AHOFM

Primary `F1-hofm3`:

- one-hot/source-vocabulary categorical fields; source-standardized semantic numeric fields plus missing indicators;
- separate order-2 and order-3 ANOVA factors, ranks `16` and `16`; factors are not shared across orders;
- AdamW `lr=1e-3`, linear L2 `1e-4`, factor L2 `1e-3`, effective batch `8192`, eight fixed cheap epochs, seed `20260821`.

Secondary `F2-ahofm3`:

- same categorical fields;
- six cubic B-spline bases per continuous field with source-only quantile knots and homogeneous marginal degrees of freedom `4`;
- order-2 and order-3 factor ranks `8` and `8`; all other settings identical.

Novelty gate: zeroing the order-3 contribution must change predictions by RMS at least `1e-4`; otherwise the result is classified as a renamed pairwise FM/GAM and cannot survive.

### EXP-113 — DCNv2

Single `D1-mix` configuration:

- legal official fields only; source-only PWL numerical embeddings with 16 knots and width 8;
- player-ID embedding width 12, team/base/calendar/inning width 4, remaining categorical width 2, unknown index zero, ID dropout `0.15`;
- two DCN-Mix layers, four experts per layer, rank `32`, parallel `[256,128]` SiLU tower, LayerNorm and dropout `0.10`;
- AdamW `lr=1e-3`, weight decay `3e-4`, effective batch `8192`, six fixed cheap epochs, seed `20260821`.

Novelty gate: zeroing the cross tower must change predictions by RMS at least `1e-4`; otherwise it is an EXP-087-style MLP under a new name and cannot survive.

### EXP-114 — sticky switching state

Configurations `H1-sticky3` (primary) and `H2-sticky4` (secondary):

- one source observation per target-free reconstructed pitcher-game;
- diagonal Gaussian emissions over source-standardized logits of prev1/3/5 success and middle rates plus their six missing indicators; variance floor `0.05^2`;
- sticky transition pseudocounts diagonal `20`, off-diagonal `1`; five deterministic initializations selected only by source likelihood; states ordered by mean success emission;
- query prior is the source-frozen pitcher terminal posterior propagated once through the learned transition per season gap; unseen pitchers use the source stationary distribution;
- query posterior uses that prior and only the current row’s legal six prev-game fields;
- residual map is posterior-weighted `state × count_index × batter_hand` with smoothing `300` and the common bounded integration.

Novelty gate: each state must have at least 5% occupancy, median maximum posterior at least `0.60`, and the learned transition must change query corrections by RMS at least `1e-4` versus an emission-only soft archetype. The posterior is never updated from validation/test peers.

### EXP-115 — correlated variational random slopes

Configurations `V1-diagonal` (primary) and `V2-rank2` (secondary):

- Gaussian EXP-071 residual likelihood;
- pitcher random-slope vector over `[1, centered balls, centered strikes, batter_hand_sign, centered outs, centered runner_count, standardized log1p(li)]`;
- batter, pitcher-team, batter-team, and `count_index × pitcher_hand × batter_hand` crossed random intercepts;
- source-season-equal ELBO; source-only empirical-Bayes prior scales bounded to `[1e-4,0.05]`; unseen entity posterior mean exactly zero;
- V1 uses a diagonal seven-dimensional population covariance; V2 uses rank-2-plus-diagonal covariance;
- Adam `lr=3e-3`, batch `16384`, 12 fixed epochs, seed `20260821`.

Novelty gate: removing all non-intercept pitcher slopes must change predictions by RMS at least `1e-4`; otherwise the family is another EB/random-intercept parameterization and cannot survive.

## 9. Row-independence contract

Every prediction must be a function of one query row and frozen source artifacts. The audit compares the same rows under singleton, ordinary batch, reverse order, a seeded permutation, two split batches, and duplication. No test-row normalization, attention, retrieval index insertion, frequency, target encoding, state update, or batch statistic is permitted. A submission candidate requires exact zero maximum difference; a numerically row-local model that is not bit-identical must export a deterministic scalar inference path and pass again before packaging.

## 10. Decision lock

No architecture, rank, state count, correction weight, threshold, seed, epoch count, or promotion gate above may be altered after 2023/2024 metrics are observed. A failed candidate is documented and stopped. A survivor is scaled only through the prespecified full rolling run. Public submission is outside this task.
