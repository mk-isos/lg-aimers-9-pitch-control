# Ultra model research report — EXP-112~115

Date: 2026-08-21<br>
Primary reference: EXP-071 `playerphys_resid_w025`, Public `1053.8615519684`<br>
Research protocol lock: `1bc9dd6384a721d93205521f6058ff4c0d368d1a2efbb5fa44a32942723184d0`

## 결론

**새로운 1100-level model basis를 발견하지 못했다.**

26개 미탐색·부분탐색 family를 조사하고, 실제로 다른 세 가지 가설 공간인 고차 all-field interaction, hidden switching state, correlated hierarchical posterior를 네 EXP·일곱 configuration으로 사전 등록해 2023/2024에서 검증했다. 결과를 본 뒤 architecture, rank, state 수, correction weight, epoch, 부호 또는 gate를 바꾸지 않았다.

- EXP-112 HOFM/AHOFM의 3차 항은 강하게 활성화됐지만 다음 시즌에서 잔차 방향이 틀렸다.
- EXP-113 DCNv2는 두 fold 모두 악화했고 cross ablation novelty gate도 실패했다.
- EXP-114 HMM은 transition이 posterior를 바꾸기는 했지만 작은 state가 1.9~2.2%로 붕괴했고 2024 방향이 역전됐다.
- EXP-115 correlated random slopes만 2023/2024를 모두 개선했으나 best V2의 recent pooled gain은 Brier `4.922879e-6`, 약 `+1.97` pooled-Skill 규모에 불과하다. 1100을 뒷받침하는 `5e-5~1e-4` signal보다 한 자릿수 이상 작다.

사전 survivor는 0개다. 따라서 full rolling 재학습, survivor ensemble, final 2025 fit, submission ZIP, Public 후보 생성은 실행하지 않았다. EXP-071을 현재 선택으로 유지한다.

## 1. 연구 절차와 재현 잠금

코드 전에 다음을 완료했다.

1. EXP-001~111, artifact, submission log, Public evidence를 family 단위로 재감사했다.
2. LG Aimers Tabular ML notebook·PDF 6개, Mathematics for ML, Supervised Learning, Time-Series Analysis 자료를 전체 추출하고 architecture-heavy page를 렌더링해 시각 확인했다.
3. 2023~2026 primary literature와 baseball intent/release/mechanics 연구를 조사했다.
4. 26개 candidate를 novelty, theory, upside, temporal robustness, diversity, deployability, compute의 7축 0~5점으로 고정 평가했다.
5. 결과 전 [`MODEL_DISCOVERY_EXP112_ULTRA.md`](MODEL_DISCOVERY_EXP112_ULTRA.md)를 SHA256 `1bc9dd...184d0`으로 잠갔다.

EXP-071 reference와 target array SHA도 lock에 포함했다. Public score는 candidate fit, selection, threshold, coefficient에 사용하지 않았다.

## 2. 조사한 새로운 model space — Top 10

| Rank | Family | Score / 35 | 핵심 새 inductive bias | 실행 |
| ---: | --- | ---: | --- | --- |
| 1 | Order-3 all-field HOFM/AHOFM | 32 | 명시적 2·3차 ANOVA 및 nonlinear tensor-product spline | EXP-112 |
| 2 | DCNv2 all-field residual | 31 | low-rank mixture cross layer | EXP-113 |
| 3 | Sticky switching dynamic logit | 30 | discrete source-fitted transition regime | EXP-114 |
| 4 | Correlated Bayesian random slopes | 30 | joint posterior random slopes와 crossed pooling | EXP-115 |
| 5 | Fishr invariant cross learner | 30 | source-season gradient-variance alignment | 보류: 2023 strict residual environment가 1개뿐 |
| 6 | Anchor/RFF invariant residual | 30 | shift-orthogonal nonlinear residual | 보류 |
| 7 | AMFormer arithmetic attention | 29 | within-row additive/multiplicative attention | neural reserve |
| 8 | Conditional multimodal TrackMan marginalizer | 28 | nonlinear p(Z|X)와 deterministic marginalization | mechanistic reserve |
| 9 | Causal TCN frozen state | 28 | multiscale source-history motifs | EXP-101 negative 뒤 보류 |
| 10 | AutoInt field-token residual | 28 | within-row sparse field attention | auditable cross 뒤 보류 |

전체 26개 matrix와 prior EXP distinction은 [`ULTRA_MODEL_DISCOVERY.md`](ULTRA_MODEL_DISCOVERY.md) 및 locked discovery 문서에 있다.

## 3. 공통 validation contract

- Cheap outer 2023은 2022 strict EXP-071 OOF residual만 학습한다.
- Cheap outer 2024는 2022·2023 strict OOF residual을 source-season equal total weight로 학습한다.
- 2019~2021의 가짜 EXP-071 residual은 만들지 않았다.
- 공통 correction은 `p = clip(p071 + 0.25 × 0.03 × tanh(raw), 0, 1)`로 고정했다. Gaussian residual-effect 모델은 동일 식을 보존하도록 `raw=atanh(clip(effect)/0.03)`로 export했다.
- validation label은 Brier·Skill·paired loss·bootstrap·oracle 진단에만 사용했다.
- same-fold oracle coefficient는 potential 진단일 뿐 deployable selection이 아니다.
- block bootstrap은 target-free reconstructed game 809개(2023), 822개(2024)를 2,000회 재표본했다.
- 모든 scored query는 다른 validation/test row를 feature, attention, normalization, retrieval, state update에 사용하지 않는다.

2022는 survivor만 full rolling한다는 사전 규칙 때문에 신규 모델을 적합하지 않았다. 아래 표의 `2022 neutral`은 “실행 결과”가 아니라 strict prior residual이 없어 candidate를 정확히 EXP-071로 유지하는 protocol-defined 값이다.

## 4. 실제 결과

### 4.1 Brier / Skill / delta

`Δ`는 candidate Brier minus EXP-071 Brier다. 음수가 개선이다.

| EXP / configuration | 2022 neutral Brier / Skill | 2023 Brier / Skill / Δ | 2024 Brier / Skill / Δ | Recent pooled Δ | Pooled error corr | Pooled oracle gain | 결과 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| EXP-071 reference | `0.244900654279 / 1710.91` | `0.247660513870 / 935.79 / 0` | `0.247604672468 / 881.58 / 0` | `0` | `1.0` | `0` | Public best |
| EXP-112 F1 HOFM3 | same as EXP-071 | `0.247761846427 / 895.26 / +1.013326e-4` | `0.247639746801 / 867.54 / +3.507433e-5` | `+6.767355e-5` | `0.999934` | `7.685848e-6` | reject |
| EXP-112 F2 AHOFM3 | same as EXP-071 | `0.247776279336 / 889.49 / +1.157655e-4` | `0.247646973729 / 864.65 / +4.230126e-5` | `+7.844584e-5` | `0.999936` | `1.594881e-5` | reject |
| EXP-113 D1 DCNv2 | same as EXP-071 | `0.247794013381 / 882.39 / +1.334995e-4` | `0.247674199181 / 853.75 / +6.952671e-5` | `+1.010015e-4` | `0.999905` | `1.073183e-5` | reject |
| EXP-114 H1 sticky3 | same as EXP-071 | `0.247656166709 / 937.53 / -4.347161e-6` | `0.247623967189 / 873.86 / +1.929472e-5` | `+7.662856e-6` | `0.999995` | `1.024180e-7` | reject |
| EXP-114 H2 sticky4 | same as EXP-071 | `0.247658903309 / 936.44 / -1.610561e-6` | `0.247624040775 / 873.83 / +1.936831e-5` | `+9.046651e-6` | `0.999995` | `2.906922e-7` | reject |
| EXP-115 V1 diagonal | same as EXP-071 | `0.247652001921 / 939.20 / -8.511948e-6` | `0.247603609120 / 882.01 / -1.063348e-6` | `-4.728078e-6` | `0.999988` | `5.067956e-6` | below gate |
| EXP-115 V2 rank2 | same as EXP-071 | `0.247651562596 / 939.37 / -8.951273e-6` | `0.247603651144 / 881.99 / -1.021324e-6` | **`-4.922879e-6`** | `0.999988` | `5.231043e-6` | best new, below gate |

### 4.2 Bootstrap와 signal strength

| Candidate | 2023 bootstrap `P(Δ<0)` / CI95 | 2024 bootstrap `P(Δ<0)` / CI95 | 해석 |
| --- | --- | --- | --- |
| EXP-112 F1 | `0.000 / [+6.826e-5,+1.338e-4]` | `0.010 / [+6.452e-6,+6.396e-5]` | 명확한 harm |
| EXP-112 F2 | `0.000 / [+8.295e-5,+1.494e-4]` | `0.0015 / [+1.488e-5,+7.108e-5]` | 명확한 harm |
| EXP-113 D1 | `0.000 / [+9.564e-5,+1.703e-4]` | `0.000 / [+2.913e-5,+1.095e-4]` | 명확한 harm |
| EXP-114 H1 | `0.6815 / [-2.322e-5,+1.584e-5]` | `0.001 / [+7.020e-6,+3.053e-5]` | 2024 역전 |
| EXP-114 H2 | `0.5680 / [-2.017e-5,+1.823e-5]` | `0.000 / [+8.037e-6,+3.020e-5]` | 2024 역전 |
| EXP-115 V1 | `0.8525 / [-2.521e-5,+7.216e-6]` | `0.5340 / [-1.467e-5,+1.346e-5]` | 방향은 일관되나 0 포함 |
| EXP-115 V2 | `0.8600 / [-2.572e-5,+6.756e-6]` | `0.5275 / [-1.448e-5,+1.341e-5]` | 방향은 일관되나 0 포함 |

어느 configuration도 사전 Tier A/B/diversity route를 통과하지 못했다. EXP-115의 correction-to-target-residual correlation도 recent pooled `0.00705`이고, 완성 error correlation은 `0.999988`이다. 이는 독립적인 large-error basis가 아니라 EXP-071 주변의 매우 작은 conditional shift다.

## 5. EXP별 발견

### EXP-112 — HOFM/AHOFM

F1은 모든 legal field의 separate rank-16 order-2/order-3 ANOVA factor를, F2는 6개 cubic B-spline basis와 rank-8 order-2/order-3 AHOFM을 사용했다. 3차 항 제거 시 prediction RMS는 F1 `0.007106/0.006778`, F2 `0.004524/0.005584`로 novelty threshold `1e-4`를 크게 넘었다.

즉 구현은 과거 EXP-021의 pairwise FM을 반복하지 않았다. 그러나 다음-season 잔차 alignment가 음수 또는 거의 0이었다. F2 oracle coefficient는 2023 `-1.378`, 2024 `-0.111`로 둘 다 반대 방향을 요구했고, pooled oracle gain도 `1.595e-5`뿐이다. 결과 뒤 부호 반전은 금지했으며, 이 구조는 “새 함수 공간이 없어서”가 아니라 “새 함수 공간이 temporal residual 방향을 보존하지 못해서” 기각됐다.

### EXP-113 — DCNv2

두 DCN-Mix layer, layer당 expert 4개·rank 32, within-row PWL numeric embedding과 `[256,128]` tower를 사용했다. 두 fold 모두 통계적으로 명확하게 악화했다. 또한 cross tower 제거 RMS가 2023 `4.433e-5`, 2024 `9.412e-5`로 frozen novelty threshold를 못 넘었다.

이 결과는 DCNv2 전체의 보편적 실패라기보다 이 fixed D1 recipe가 사실상 deep tower 중심으로 수렴했고, 그 tower가 residual을 과적합했다는 증거다. 같은 family rank/layer/dropout sweep은 하지 않는다.

### EXP-114 — sticky switching regime

2019~cutoff target-free pitcher-game sequence에 3/4-state diagonal Gaussian sticky HMM을 5개 deterministic initialization으로 적합했다. query는 source-frozen pitcher terminal posterior를 season gap만큼 transition한 prior와, 현재 행 자체의 legal prev1/3/5 success/middle rate만 결합했다. validation row 순서 filtering은 없다.

Transition은 emission-only archetype과 실제로 다른 posterior를 만들었다. 그러나 작은 state 점유가 2023 `2.155%`, 2024 `1.879%`로 5% gate를 실패했다. median maximum posterior가 `0.99` 수준인 것은 form confidence라기보다 missing/rate pattern 분리에 가까웠다. 2023의 작은 gain은 2024에서 약 `+1.93e-5` harm으로 역전돼 EXP-072보다 복잡한 state가 안정된 새 signal을 만들지 못했다.

### EXP-115 — correlated variational random slopes

Pitcher별 `[intercept, balls, strikes, batter hand, outs, runners, log1p(LI)]` slope와 batter/team/context crossed intercept를 joint Gaussian posterior로 학습했다. V1은 diagonal population covariance, V2는 rank-2-plus-diagonal covariance다. prior scale은 source-only ELBO로 `[1e-4,0.05]` 안에서 학습했고 unseen entity mean은 정확히 0이다.

Non-intercept slope 제거 RMS가 V2에서 2023 `0.000902`, 2024 `0.001431`로 novelty gate를 통과했다. 두 fold 모두 EXP-071과 EXP-051보다 낮은 Brier를 기록한 유일한 family다. 그러나 pooled gain이 `4.923e-6`, oracle ceiling도 `5.231e-6`이므로 +40 Skill 가설과는 거리가 크다. 이 모델은 “joint random slope signal은 존재한다”는 긍정적 과학 결과지만 submission model로 승격할 정도의 evidence는 아니다.

## 6. Runtime, memory, and row independence

| EXP | 전체 cheap runtime | 최대 RSS | scored inference | exact row audit |
| --- | ---: | ---: | --- | --- |
| EXP-112 | `155.7 s` | `1,851.6 MB` | deterministic NumPy | singleton/batch/reverse/permutation/split/duplicate max `0` |
| EXP-113 | `106.8 s` | `1,483.8 MB` | vectorized Torch; deterministic scalar audit export | scalar export max `0`; vector-vs-scalar max `7.41e-9` |
| EXP-114 | `988.0 s` | `1,088.0 MB` | NumPy frozen state | all max `0` |
| EXP-115 | `11.7 s` | `795.5 MB` | NumPy scalar posterior mean | all max `0` |

모든 모델은 28GB memory와 600초 inference envelope 안에 들어가는 row-local inference path를 갖는다. 단 EXP-113 vectorized path는 singleton numerical equality가 bitwise가 아니므로, hypothetical packaging에는 느린 scalar export를 사용해야 한다. 이 모델은 성능·novelty gate에서 이미 탈락했으므로 package를 만들지 않았다.

## 7. Full rolling, ensemble, and Public policy

### Full rolling

Cheap survivor가 0이므로 신규 model fit을 2022까지 확장하지 않았다. 이것은 누락이 아니라 “survivor만 2022/2023/2024 full rolling”이라는 frozen protocol의 결과다. 모든 killed candidate의 2022 row prediction은 strict residual source 부재 때문에 EXP-071 neutral로 정의된다.

### Ensemble

독립적인 strong basis가 최소 2개 생기지 않아 ensemble을 실행하지 않았다. EXP-115를 EXP-071과 다시 섞는 것은 이미 fixed `0.25` correction weight를 사후 조정하는 것과 같고, low-correlation correction composition EXP-105~110을 반복한다.

### Public Candidate A / B

- Candidate A: 없음
- Candidate B: 없음
- 신규 ZIP: 없음
- 신규 SHA256: 없음
- DACON 제출: 실행하지 않음

기존 control은 `ready_to_submit/2026-08-20-post58/EXP-071-PLAYERPHYS-RESID.zip`, SHA256 `27ab6adbc4e2f460d705438dc5b6cbd0d9dffd9a0d519354a47d26aa6d240ad3`이며 재제출하지 않는다.

## 8. 가장 중요한 발견

1. **고차 interaction 함수 공간은 실제로 미탐색이었다.** EXP-112의 order-3 ablation은 이를 확인했다. 그러나 해당 interaction은 source residual을 강하게 맞추면서 next-season 부호를 보존하지 못했다.
2. **구조보다 information boundary가 계속 지배한다.** Baseball 연구가 강조하는 catcher target, realized location, current pitch type, release angle/physics가 query에서 없다. 더 복잡한 network는 이 posterior information을 생성하지 못한다.
3. **joint Bayesian pooling에는 작지만 일관된 신호가 있다.** EXP-115는 유일하게 두 recent fold를 모두 개선했다. 다만 signal은 1100 기준의 약 5~10%가 아니라 대략 1/20 수준이고 error diversity도 거의 없다.
4. **switching state는 missingness/archetype을 재발견했다.** 작은 state occupancy와 2024 reversal 때문에 EXP-072 AR(1)보다 유효한 hidden regime 근거가 되지 못했다.
5. **Public-positive residual physics는 여전히 특수하다.** EXP-070 absolute physics가 실패하고 EXP-071 residual physics가 Public winner였던 기존 증거를 새 official-X model이 대체하지 못했다.

## 9. Reproducibility and authoritative artifacts

| Artifact | SHA256 |
| --- | --- |
| EXP-112 report | `202a6cf7627b5c329abd1396d135cccdb3182e513c262c8b444c2998aadc765d` |
| EXP-113 report | `8a752db9b14222d70befbf060ef2153a1be3ee3a833376fe7d98dfbdbcafd442` |
| EXP-114 report | `76f8c3b0dfc2ac0996d4712c9833f8f3887f8b31ebbace42f67637bb73cba9ed` |
| EXP-115 report | `97523498642bd76c982b0d30d55847dfcf1c6c14619c0acb58ba73f863e953e6` |

Run commands:

```bash
PYTHONPATH=experiments .venv/bin/python experiments/train_exp112_hofm_ahofm_residual.py --device cpu
PYTHONPATH=experiments .venv/bin/python experiments/train_exp113_dcnv2_all_field_residual.py
PYTHONPATH=experiments .venv/bin/python experiments/train_exp114_sticky_switching_regime.py
PYTHONPATH=experiments .venv/bin/python experiments/train_exp115_variational_random_slopes.py --config all
```

Independent post-run checks reloaded every prediction/target/reference array and reproduced every saved delta Brier exactly.

## 10. 1100 가능성과 다음 행동

**1100 가능성: Low under the current legal feature boundary.**

이 결론은 “모든 ML model이 소진됐다”는 뜻이 아니다. 이번에 실제 실행한 fixed recipes와 동일 information boundary에서 architecture만 바꿔 `~1e-4` signal을 얻을 사전 확률이 낮다는 뜻이다.

권장 행동:

1. EXP-071을 leaderboard 선택으로 유지하고 이번 EXP-112~115는 제출하지 않는다.
2. 추가 architecture sweep, sign flip, rank/state/weight tuning, EXP-115 blend search를 중단한다.
3. 다음 tranche를 열려면 catcher target, intended location, 합법적인 current-pitch selection/physics proxy처럼 **새로운 independent row-level information**이 먼저 필요하다.
4. 새 정보 없이 한 번 더 연구한다면 사전에 별도 잠근 conditional multimodal TrackMan marginalizer 또는 작은 AMFormer만 허용하되, 기대치는 1100이 아니라 정보-ceiling 검증으로 둔다.
