# LG Aimers 9기 — 투구 제구 성공 확률 예측

투구 직전의 경기 상황과 과거 이력을 이용해 `control_success = 1`일 확률을 예측하는 과정을 기록한 학습·실험 저장소입니다.

운영진 RandomForest 베이스라인을 재현한 뒤, 피처 엔지니어링과 여러 트리 기반 모델을 같은 시간 기준 검증 방식으로 비교합니다. 최고 점수만 남기기보다 가설, 코드 변경, 실패한 실험과 배운 내용을 함께 기록하는 것을 목표로 합니다.

> 이 저장소에는 대회 원본 데이터, 학습된 모델, 예측 결과와 제출 ZIP을 포함하지 않습니다. 실험 표의 Brier/Skill은 로컬 검증 결과이며, `Public`으로 명시한 값만 리더보드 점수입니다.

## 문제와 검증 방식

- Target: `control_success`
- 평가 지표: Brier Score 및 대회식 Brier Skill Score
- 주 검증: 2022·2023·2024 rolling-origin 검증
- 선택 규칙: 현재 검증 시즌의 정답을 모델·보정·rank 선택에 다시 사용하지 않고, 평균뿐 아니라 최신 시즌과 최저 Skill을 함께 확인
- 원칙: 현재 투구 직전에 알 수 있는 정보만 사용하고 테스트 데이터 내부 집계는 사용하지 않음

## 실험 결과

2024년을 검증 시즌으로 사용한 결과입니다. Brier Score는 낮을수록, Skill Score는 높을수록 좋습니다.

| 실험 | 주요 변경 | Brier Score | Skill Score | 결과 |
| --- | --- | ---: | ---: | --- |
| EXP-001 | 운영진 RandomForest 베이스라인 재현 | 0.248767 | 416.18 | 비교 기준 |
| EXP-002 | 상황 조합 피처 6개 추가 | 0.248637 | 468.44 | 개선 |
| EXP-003 | HistGradientBoosting | 0.248075 | 693.20 | 모델 비교 기준 |
| EXP-004 | HistGradientBoosting 용량·규제 조정 | 0.248129 | 671.69 | EXP-003보다 하락 |
| EXP-005 | 선수 ID Target Encoding | 0.248493 | 525.87 | EXP-003보다 하락 |
| EXP-006 | CatBoost 범주형 처리 | 0.248303 | 602.05 | EXP-003보다 하락 |
| EXP-007 | LightGBM 범주형 처리 | 0.248866 | 376.80 | EXP-003보다 하락 |
| EXP-008 | 로지스틱 회귀 | 0.249790 | 6.63 | EXP-003보다 하락 |
| EXP-009 | 선수 ID 제거 + HistGradientBoosting | 0.248094 | 685.73 | EXP-003에 근접 |
| EXP-010 | 작은 트리 + 강한 규제 HistGradientBoosting | 0.248151 | 663.02 | EXP-003보다 하락 |
| EXP-011 | EXP-003 입력 표현 + LightGBM | 0.248043 | 706.03 | 이전 단일 모델 기준 |
| EXP-012 | EXP-003 입력 표현 + XGBoost | 0.248079 | 691.82 | EXP-011보다 하락 |
| EXP-013 | CatBoost 28.7% + LightGBM 71.3% 앙상블 + 선형 확률 보정 | 0.247862497 | 778.37 | 앙상블 기준 실험 |
| EXP-014 | 엔지니어드 피처 + LightGBM 규제 조정 + 선형 확률 보정 | **0.247857248** | **780.47** | 같은 시즌 보정으로 낙관 편향 |
| EXP-018 | 계층적 as-of 기준값 + 안정적 그룹 효과 + 최근 residual 15% | **0.247820261** | **795.28** | Public 895.84로 EXP-013보다 하락, 비채택 |
| EXP-019 | LightGBM·HistGradientBoosting residual 50:50 + source-season Team EB | **0.247682562** | **850.40** | 3개 시즌 모두 기준 개선, 다음 실험의 기준값으로 채택 |
| EXP-020 | 투수×24개 count/타자 손 문맥 Team residual의 low-rank SVD | **0.247633803** | **869.92** | 이전 fold만으로 rank를 선택한 strict 후보 채택 |
| EXP-021 | strict rank-6 전체 학습·직렬화·제출 패키지 | **0.247633803** | **869.92** | **Public 1043.607420, 최종 리더보드 선택** |
| EXP-021-AGGR | R/F gate + 투수×count×타자 손 EB | **0.247622900** | **874.29** | Public 1043.187131, strict보다 0.420289 낮아 비채택 |
| EXP-022 | 누적 rate 증분 보조 outcome 4종 + prior-only Ridge residual | 0.247627980 | 872.25 | 2023 하락·same-fold ceiling도 1100 미달, 비채택 |
| EXP-023 | success·failure joint taxonomy 5-class HGB + prior blend | 0.247629644 | 871.59 | same-fold 1100은 통과했지만 시간 전이 실패 |
| EXP-024 | source-season별 joint expert bagging | **0.247609821** | **879.52** | 2024 개선, 2023 832.85로 붕괴해 비채택 |
| EXP-025 | row-local source-season similarity gate | 0.247644599 | 865.60 | unseen 시즌 gate extrapolation 실패 |
| EXP-026 | source expert 예측의 행별 시간 외삽 | 0.247625302 | 873.32 | 2023 기준 미달, taxonomy branch 중단 |
| EXP-027 | 투수×count 문맥 EB를 logit odds-ratio로 재추정 | 0.247636243 | 868.94 | 2023 소폭 개선·2024 하락, 비채택 |
| EXP-028 | 시즌·class prevalence 균형 joint taxonomy | 0.247623456 | 874.06 | 2024 개선·2023 하락, 비채택 |
| EXP-029 | 복원 pitch group×outcome 15-class HGB | 0.247632500 | 870.44 | 신규 보조 label은 유효하나 시간 전이 실패 |
| EXP-030 | 예측 구종 성향−공식 장기 mix Ridge residual | 0.247635566 | 869.22 | 2023·2024 모두 기준 하락, branch 중단 |
| EXP-031 | 구종군별 success mixture-of-experts | 0.247636685 | 868.77 | 2022 개선·2023 급락, pitchmix branch 종료 |
| EXP-032 | strict·aggressive·recency bounded consensus | 0.247618091 | 876.21 | Public **1046.988993**, 현재 리더보드 선택 |
| EXP-033 | TrackMan sequence·fine-pitch·시간 추세 LightGBM residual | 0.247639600 | 867.60 | 2023·2024 하락, 비채택 |
| EXP-034 | EXP-033 TrackMan 매핑 비용 범위 확장 | 0.247641946 | 866.66 | 매핑 확대도 최신 성능 하락, 비채택 |
| EXP-035 | TrackMan 타자·투수×타자 matchup profile residual | 0.247638519 | 868.03 | 기준보다 하락, 비채택 |
| EXP-036 | TrackMan count-transition control proxy | 0.247630668 | 871.18 | 2024 소폭 개선, 2023 하락으로 비채택 |
| EXP-037 | low-rank source-season recency 정책 | 0.247628091 | 872.21 | 2024 소폭 개선, hard gate 미달로 비채택 |
| EXP-038 | 저장 예측 전체의 same-fold convex oracle 상한 감사 | 0.247577414 | 892.49 | 2023·2024 Skill 1000 불가, 재가중 탐색 중단 |
| EXP-039 | 과거 OOF 기반 4-expert LightGBM stack | 0.247631638 | 870.79 | 2023 하락, 비채택 |
| EXP-040 | recency:aggressive 70:30 bounded consensus | 0.247620314 | 875.32 | 2024 개선이나 hard gate 미달, 비채택 |
| EXP-041 | exact game-sequence TrackMan 정렬 residual | 0.247636022 | 869.03 | 2023 하락, 비채택 |
| EXP-042 | EXP-041 + source recency2 정책 | 0.247647232 | 864.55 | 최신 성능 하락, 비채택 |
| EXP-043 | exact-aligned fine-pitch control empirical Bayes | 0.247616774 | 876.74 | 2023·2024 개선, hard gate 미달로 단독 비채택 |
| EXP-044 | exact TrackMan control + recentaggr 50:50 | **0.247611615** | **878.80** | Public 1046.949994, EXP-032 recentaggr보다 0.038999 낮아 비채택 |

EXP-014의 2024년 최고 구성은 엔지니어드 피처, 원-핫 인코딩, `num_leaves=63`, `min_child_samples=1000`인 LightGBM이다. 보정 후 Brier Score `0.24785724834181783`, Skill Score `780.4741206669407`로 EXP-013의 JSON 기록(`0.247862497`, `778.37`)보다 소폭 개선됐다. 그러나 같은 설정의 2023년 보정 점수는 Brier Score `0.24989886191192345`, Skill Score `40.4545039712767`로 급락했다. 따라서 EXP-014는 2024년 로컬 최고로 기록하되, 시간 일반화가 확인되지 않아 최종 모델로는 채택하지 않는다.

EXP-018은 공식 누적 `asof_*`와 학습 데이터에서 확정한 직전 시즌 종료 상태의 차이로 현재 시즌 투수·타자 표본과 성공률을 행별로 복원한다. 여기에 과거 3시즌의 `count × 투수 손 × 타자 손` 안정 효과와 직전 1시즌 residual LightGBM의 15%만 더한다. 2022·2023·2024 rolling Skill은 각각 `1856.48`, `769.85`, `795.28`이고 평균 `1140.54`, 최저 `769.85`다. 같은 rolling에서 EXP-013 raw 앙상블은 평균 `582.06`, 최저 `-1303.57`이었다. 그러나 2025 Public은 `895.8368767677`로 EXP-013 Public `935.8108097065`보다 `39.9739329388` 낮았다. 다중 시간축 평균 개선이 비공개 시즌 개선으로 이어지지 않았으므로 EXP-018은 비채택했고, 당시 최고 제출은 EXP-013으로 유지했다.

EXP-019는 R행 residual LightGBM과 HistGradientBoosting을 branch별로 확률 범위에 자른 뒤 50:50으로 결합하고, 각 과거 OOF 시즌의 팀 잔차 효과를 독립적으로 추정해 동일 가중 평균했다. `all_prior_s1000`은 2022·2023·2024 Skill `1784.40`, `898.61`, `850.40`을 기록했고 고정 50:50 기준보다 세 시즌 모두 개선했다. EXP-020은 이 기준 위에 투수별 24개 `count_index × batter_hand` 문맥 효과를 smoothing `300`으로 축소한 뒤 rank `6` SVD로 압축했다. 고정 rank-6의 rolling Brier는 `0.244704584`, `0.247731144`, `0.247633803`, Skill은 `1789.60`, `907.54`, `869.92`이며 기준 대비 Skill 변화는 `+5.20`, `+8.93`, `+19.52`다. 2021~2024 OOF만 사용한 2025 prospective 선택도 rank `6`을 선택했으므로 이를 EXP-021 strict 최종 후보로 전체 학습·직렬화했다. 다만 13개 저장 후보의 같은-fold convex oracle조차 2023·2024 Skill이 `936.33`, `880.59`여서 1100에 미달했다. 현재 신호 집합의 단순 재가중만으로 목표를 달성할 수 있다는 근거는 없다.

EXP-021의 2025 Public 결과는 strict `1043.6074197937`, aggressive `1043.1871309639`였다. 두 후보 모두 이전 최고 EXP-013 `935.8108097065`를 크게 넘었지만, 과거 OOF만으로 선택한 strict가 최근 시즌 로컬 점수를 보고 사후 선택한 aggressive보다 `0.4202888298` 높았다. 따라서 최종 리더보드 선택은 strict로 변경한다. 이 결과는 최신 fold의 소폭 우위만 좇기보다 선택 절차의 시간적 독립성과 여러 시즌의 하방 안정성을 함께 보는 편이 더 안전하다는 근거로 기록한다.

EXP-022는 `(pitcher_id, season, asof_pitcher_n+1)` 상태를 행 순서와 무관하게 찾아 누적 rate count 증분으로 reverse·middle·ball·strike 보조 label을 복원했다. 1,475,092행 중 1,472,040행이 유효했고 복원 success와 `control_success` mismatch는 `0`이었다. 그러나 prior-only residual 후보의 2022·2023·2024 Skill은 `1838.95`, `871.51`, `872.25`로 2023이 EXP-021 strict보다 하락했다. 현재 fold 정답까지 허용한 고정 Ridge 진단도 2023 `919.25`, 2024 `873.42`에 그쳐 linear multi-task family의 1100 gate를 통과하지 못했다. 전체 학습과 ZIP은 생성하지 않는다.

EXP-023은 중첩 label을 success·reverse-only·middle-only·reverse+middle·other-failure의 5개 상호 배타 class로 바꿔 하나의 HistGradientBoostingClassifier가 공동 학습하게 했다. 같은 시즌 정답을 허용한 비배포 ceiling은 2023 `1417.94`, 2024 `1331.79`로 1100을 넘었지만 prior-only 선택은 `2022.15`, `849.32`, `871.59`로 전이되지 않았다. EXP-024 source-season expert bagging, EXP-025 row-local regime similarity gate, EXP-026 행별 expert trend 외삽까지 제한해 검증했지만 어느 후보도 2023과 2024를 함께 개선하지 못했다. EXP-026 최종 Skill은 `1879.61`, `905.94`, `873.32`였고 hard gate는 `false`다. outcome taxonomy branch는 종료하고 EXP-021 strict를 유지한다.

EXP-027은 기존 additive 확률 residual 대신 source-season별 투수×24개 문맥 odds-ratio를 rank 6으로 압축해 logit 공간에 결합했지만 Skill `1795.16`, `908.50`, `868.94`에 그쳤다. EXP-028은 각 source season과 5개 outcome class에 같은 학습 질량을 주어 prevalence drift를 제거했으나 `1852.64`, `894.66`, `874.06`으로 2023이 악화됐다. EXP-029는 누적 pitchmix 상태의 다음 keyed state로 1,472,832개 fastball·breaking·offspeed label을 one-hot 오차 없이 복원하고 outcome taxonomy와 교차한 15-class 모델을 학습했다. 선택 Skill은 `1855.69`, `887.32`, `870.44`였다. EXP-030에서 구종 성향만 분리한 8차원 Ridge residual도 `1793.81`, `904.84`, `869.22`로 기준을 넘지 못했다. pitchmix supervision은 규정 내 새 label이지만 미래 시즌 성공 확률 개선으로 전이되지 않아 final fit과 ZIP을 생성하지 않는다.

EXP-031은 앞선 두 pitchmix 모델과 구조를 달리해 fastball·breaking·offspeed별 success HGB를 독립 학습하고, EXP-030의 prior-only 행별 구종 성향으로 세 expert를 혼합했다. prior-only blend 선택 결과 Skill은 `1961.10`, `858.35`, `868.77`이었다. 2022의 큰 개선이 2023에서 역전됐고 2024도 기준보다 낮아 pitchmix conditional expert branch까지 종료했다.

EXP-032의 recentaggr는 recency2 low-rank 효과와 aggressive branch를 50:50으로 고정 결합한 제출 후보이며 Public `1046.9889925352`를 기록했다. EXP-033~043은 TrackMan 과거 이력, source-season 정책, 저장 예측의 convex 상한, exact game-sequence 정렬을 순서대로 검증했다. 단독 후보 중 EXP-043의 exact-aligned fine-pitch control은 2023·2024에서 기준보다 높았지만, 세 시즌 Skill 1000 hard gate를 넘지 못해 전체 학습 후보로는 채택하지 않았다. EXP-044는 그 신호를 recentaggr와 50:50으로 결합해 2024 Brier `0.24761161511381488`, Skill `878.803350841606`, Public `1046.9499938833`을 기록했다. Public은 recentaggr보다 `0.03899865189987395` 낮아 최종 리더보드 선택은 EXP-032 recentaggr로 유지한다.

## 베이스라인과 노트북

- `[Baseline_Train]_RandomForest를 활용한 모델 학습 및 피쳐엔지니어링 (학습).ipynb`: 베이스라인 학습과 시간 기준 검증
- `[Baseline_Inference]_RandomForest를 활용한 모델 학습 및 피쳐엔지니어링 (추론).ipynb`: 샘플 데이터 추론과 제출 형식 확인
- `[EXP-002_Train]_RandomForest_상황피처_6개.ipynb`: 상황 조합 피처 6개를 추가한 첫 개선 실험

베이스라인은 이후 실험의 출발점과 비교 기준을 보존하기 위해 수정본과 분리해 유지합니다. 반복 실행과 비교가 필요한 실험은 `experiments/`의 Python 스크립트로도 관리합니다.

## 저장소 구성

```text
.
├── [Baseline_Train]_....ipynb
├── [Baseline_Inference]_....ipynb
├── [EXP-002_Train]_....ipynb
├── experiments/              # EXP-002~044 학습·검증·패키징 코드
├── submissions/              # 모델을 제외한 추론 코드와 환경 명세
├── artifacts/                # 작은 validation_metrics.json만 추적
├── docs/                     # 실험·학습·환경·제출 기록
├── data/README.md            # 필요한 데이터 파일 안내
├── requirements.txt
└── README.md
```

## 실행 방법

Python 3.12 환경을 기준으로 기록했습니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

대회에서 제공받은 파일을 다음 위치에 직접 배치합니다.

```text
data/train.csv
data/test.csv
data/sample_submission.csv
data/trackman_history.csv
```

예를 들어 최신 LightGBM 탐색 실험인 EXP-014의 2024년 검증은 다음과 같이 실행합니다.

```bash
python experiments/train_exp014_temporal_categorical_lgbm.py \
  --validation-season 2024 \
  --num-leaves 63 \
  --feature-set engineered \
  --category-mode onehot \
  --min-child-samples 1000
```

검증 시즌을 인자로 받는 EXP-003의 2023년 추가 검증은 다음과 같이 실행합니다.

```bash
python experiments/train_exp003_histgb.py --validation-season 2023
```

실행 중 생성되는 모델, 예측 배열과 출력 파일은 Git에서 제외되고 요약 지표 JSON만 기록됩니다.

## 기록 문서

- [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md): 가설, 변경 내용, 검증 결과와 다음 실험
- [`docs/LEARNING_LOG.md`](docs/LEARNING_LOG.md): 프로젝트를 진행하며 공부한 개념과 블로그 기록
- [`docs/DATA_NOTES.md`](docs/DATA_NOTES.md): 데이터 의미, 피처 후보와 누수 점검
- [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md): 로컬·평가 서버 환경과 호환성
- [`docs/SUBMISSION_LOG.md`](docs/SUBMISSION_LOG.md): 제출 구조, 점검 항목과 결과

## 다음 단계

- EXP-032 recentaggr Public `1046.9889925352`를 현재 최종 리더보드 선택으로 유지하기
- EXP-044 Public `1046.9499938833`은 recentaggr보다 `0.03899865189987395` 낮았으므로 exact TrackMan 신호의 단독 결합은 채택하지 않기
- EXP-038의 낙관적 same-fold convex oracle도 2023·2024 Skill 1000에 도달하지 못했으므로 저장 예측의 추가 재가중 탐색을 중단하기
- EXP-033~042의 TrackMan sequence·matchup·proxy·source policy·expert stack은 세 시즌 하방을 개선하지 못했으므로 해당 단독 branch를 중단하기
- EXP-043의 exact-aligned fine-pitch control은 2023·2024 개선 신호로 보존하되, 독립적 과거 OOF 선택과 세 시즌 hard gate를 만족하는 새 행별 신호가 있을 때만 재검증하기
- 학습·추론 피처 parity와 테스트 행 독립성 검사를 계속 유지하기

## 관련 글

- [데이터로 야구를 이해하고 예측하는 방법](https://mkisos.tistory.com/entry/lgaimers9-project-1)
- [모델 개발부터 코드 제출까지 전체 과정 정리](https://mkisos.tistory.com/entry/lgaimers9-project-2)
