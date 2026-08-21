# Ultra model discovery — EXP-112+

This is the result-aware index for the EXP-112+ research reset. The immutable pre-result taxonomy, course/literature audit, exact configurations, seven-axis scores, and promotion gates remain in [`MODEL_DISCOVERY_EXP112_ULTRA.md`](MODEL_DISCOVERY_EXP112_ULTRA.md), locked before training at SHA256 `1bc9dd6384a721d93205521f6058ff4c0d368d1a2efbb5fa44a32942723184d0`. Final metrics and interpretation are in [`ULTRA_MODEL_RESEARCH_REPORT.md`](ULTRA_MODEL_RESEARCH_REPORT.md).

## Research conclusion

The repository has not tested the most direct mathematical formulation of the task: explicit all-field third-order interaction over player, count, handedness, base/game state, and reliability/history fields. The old EXP-021 “FM” result covers only a narrow rank-2/4 player×context model. It does not close HOFM/AHOFM or DCNv2.

Three independent hypotheses were defensible before execution:

1. **Explicit all-field interactions** — EXP-112 HOFM/AHOFM and EXP-113 DCNv2.
2. **Discontinuous latent pitcher form** — EXP-114 sticky switching states, with a mandatory transition-vs-archetype novelty check.
3. **Joint posterior partial pooling** — EXP-115 correlated pitcher random slopes, with intercept-only models prohibited.

The selected portfolio was deliberately narrower than the 26 investigated families. Multi-task labels, generic SSL, retrieval, ordinary sequence encoders, entity GraphSAGE, correction stacking, and marginal TrackMan geometry already had relevant negative evidence. Fishr could not be identified cleanly for outer 2023 because there was only one strict prior EXP-071 residual environment.

All four preregistered families were executed without post-result architecture, rank, state-count, sign, epoch, correction-weight, or gate changes. None passed the frozen survivor gate:

- EXP-112 proved that active third-order terms were present, but both variants harmed both recent folds.
- EXP-113 harmed both folds and its cross tower failed the mandatory novelty-ablation threshold.
- EXP-114 made a small 2023 improvement, then reversed in 2024 while its rare state collapsed below 5% occupancy.
- EXP-115 improved both folds, but best pooled ΔBrier was only `-4.922879e-6`, below the Tier-A `-1e-5` floor and far below a credible 1100-level basis.

There were zero survivors. Per the locked protocol, no new 2022 full rolling fit, ensemble, 2025 final fit, ZIP, or submission was produced; EXP-071 remains selected.

## Frozen 26-family novelty matrix and execution status

Scores are fixed integers from 0 to 5 in this order: novelty (`N`), theoretical relevance (`T`), expected upside (`U`), temporal robustness (`R`), expected error diversity (`D`), deployability (`P`), and compute efficiency (`C`). Detailed distinctions from prior EXPs and the proposed +40-Skill mechanism are in the immutable discovery document.

| Rank | Candidate family | N/T/U/R/D/P/C | Total | Execution status and final decision |
| ---: | --- | --- | ---: | --- |
| 1 | All-field order-3 HOFM/AHOFM residual | 5/5/5/4/4/5/4 | **32** | EXP-112 F1/F2 executed; both rejected |
| 2 | DCNv2 all-field residual | 5/5/4/4/4/5/4 | **31** | EXP-113 D1 executed; rejected |
| 3 | Switching hierarchical dynamic logit | 5/5/4/4/5/4/3 | **30** | EXP-114 H1/H2 executed; rejected |
| 4 | Correlated Bayesian random slopes | 5/5/4/5/4/4/3 | **30** | EXP-115 V1/V2 executed; best new, below gate |
| 5 | Fishr season-invariant cross residual | 5/5/3/5/4/5/3 | **30** | Deferred: outer-2023 has one strict residual environment |
| 6 | Anchor/RFF invariant residual | 4/5/3/5/4/5/4 | **30** | Deferred |
| 7 | Structured official-label MMoE | 4/4/3/3/3/4/3 | **24** | Closed for this tranche; too close to prior auxiliary branches |
| 8 | Conditional multimodal TrackMan marginalizer | 5/5/4/3/5/4/2 | **28** | Mechanistic reserve only |
| 9 | Causal TCN frozen pitcher state | 5/4/4/3/5/4/3 | **28** | Deferred after EXP-101 negative evidence |
| 10 | AutoInt field-token residual | 5/4/4/3/5/4/3 | **28** | Deferred behind auditable crosses |
| 11 | AMFormer arithmetic residual | 5/5/4/3/5/4/3 | **29** | Best deferred neural reserve |
| 12 | T2G-Former | 5/4/4/3/5/3/2 | **26** | Deferred |
| 13 | Sparse representation-level MoE | 5/4/4/3/5/4/3 | **28** | Deferred |
| 14 | Shape-constrained NAM + interactions | 4/4/3/4/4/5/4 | **28** | Diagnostic only |
| 15 | Stability-selected symbolic residual | 3/4/3/4/4/5/5 | **28** | Deferred |
| 16 | S4/LS4 frozen event state | 5/4/4/3/5/4/2 | **27** | Deferred |
| 17 | Joint pitch-choice/execution latent model | 4/5/4/3/5/4/2 | **27** | Reserve |
| 18 | Bayesian changepoint mixture | 4/4/3/3/4/5/4 | **27** | Deferred |
| 19 | HGT/RGCN temporal heterograph | 5/4/3/3/5/4/2 | **26** | Deferred after EXP-103 negative evidence |
| 20 | Conditional privileged projection | 4/5/3/3/5/4/2 | **26** | Reserve |
| 21 | Hierarchical BART residual | 5/4/3/4/4/3/2 | **25** | Deferred |
| 22 | Sparse variational GP product kernel | 5/4/3/4/5/3/1 | **25** | Deferred |
| 23 | T-JEPA full-source pretraining | 5/3/3/3/5/4/2 | **25** | Rejected for this tranche |
| 24 | True TabR residual | 4/4/3/3/5/3/2 | **24** | Deferred after retrieval negatives |
| 25 | TabDPT singleton repair | 4/4/3/3/5/1/1 | **21** | Not admissible until exact-row/runtime repair |
| 26 | Newer TabPFN 2.5/2.6/3 | 4/4/3/3/5/1/1 | **21** | Conditional on written license/runtime clearance |

## Frozen execution order

1. EXP-112 F1/F2 cheap 2023/2024.
2. EXP-113 D1 cheap 2023/2024.
3. EXP-114 H1/H2 cheap 2023/2024.
4. EXP-115 V1/V2 cheap 2023/2024.
5. Promote at most two configurations through the preregistered Tier A/B/diversity gates.
6. Run 2022/2023/2024 full rolling only for those survivors.
7. Build an ensemble only if at least two independent bases survive.
8. Build at most two frozen ZIPs, never submit them from Codex.

## Executed configurations

`Δ` is candidate Brier minus EXP-071; negative is better.

| EXP / config | 2023 ΔBrier | 2024 ΔBrier | Recent pooled Δ | Critical diagnostic | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| EXP-112 F1 HOFM3 | `+1.013326e-4` | `+3.507433e-5` | `+6.767355e-5` | order-3 ablation RMS `0.007106/0.006778` | Reject |
| EXP-112 F2 AHOFM3 | `+1.157655e-4` | `+4.230126e-5` | `+7.844584e-5` | order-3 ablation RMS `0.004524/0.005584` | Reject |
| EXP-113 D1 DCNv2 | `+1.334995e-4` | `+6.952671e-5` | `+1.010015e-4` | cross ablation RMS `4.433e-5/9.412e-5` | Reject |
| EXP-114 H1 sticky3 | `-4.347161e-6` | `+1.929472e-5` | `+7.662856e-6` | minimum state occupancy `2.155%/1.879%` | Reject |
| EXP-114 H2 sticky4 | `-1.610561e-6` | `+1.936831e-5` | `+9.046651e-6` | minimum state occupancy below 5% | Reject |
| EXP-115 V1 diagonal | `-8.511948e-6` | `-1.063348e-6` | `-4.728078e-6` | both folds improve; CI includes zero | Below gate |
| EXP-115 V2 rank2 | `-8.951273e-6` | `-1.021324e-6` | **`-4.922879e-6`** | pooled error corr `0.999988` | Best new; below gate |

## Terminal execution status

1. Full 2022/2023/2024 rolling: not run because no cheap survivor.
2. Survivor-only ensemble: not run because there were no independent survivors.
3. Public Candidate A/B: none.
4. New submission ZIP/SHA: none.
5. Selected production/leaderboard basis: EXP-071, unchanged.
