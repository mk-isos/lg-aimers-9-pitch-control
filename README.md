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

EXP-014의 2024년 최고 구성은 엔지니어드 피처, 원-핫 인코딩, `num_leaves=63`, `min_child_samples=1000`인 LightGBM이다. 보정 후 Brier Score `0.24785724834181783`, Skill Score `780.4741206669407`로 EXP-013의 JSON 기록(`0.247862497`, `778.37`)보다 소폭 개선됐다. 그러나 같은 설정의 2023년 보정 점수는 Brier Score `0.24989886191192345`, Skill Score `40.4545039712767`로 급락했다. 따라서 EXP-014는 2024년 로컬 최고로 기록하되, 시간 일반화가 확인되지 않아 최종 모델로는 채택하지 않는다.

EXP-018은 공식 누적 `asof_*`와 학습 데이터에서 확정한 직전 시즌 종료 상태의 차이로 현재 시즌 투수·타자 표본과 성공률을 행별로 복원한다. 여기에 과거 3시즌의 `count × 투수 손 × 타자 손` 안정 효과와 직전 1시즌 residual LightGBM의 15%만 더한다. 2022·2023·2024 rolling Skill은 각각 `1856.48`, `769.85`, `795.28`이고 평균 `1140.54`, 최저 `769.85`다. 같은 rolling에서 EXP-013 raw 앙상블은 평균 `582.06`, 최저 `-1303.57`이었다. 그러나 2025 Public은 `895.8368767677`로 EXP-013 Public `935.8108097065`보다 `39.9739329388` 낮았다. 다중 시간축 평균 개선이 비공개 시즌 개선으로 이어지지 않았으므로 EXP-018은 비채택했고, 당시 최고 제출은 EXP-013으로 유지했다.

EXP-019는 R행 residual LightGBM과 HistGradientBoosting을 branch별로 확률 범위에 자른 뒤 50:50으로 결합하고, 각 과거 OOF 시즌의 팀 잔차 효과를 독립적으로 추정해 동일 가중 평균했다. `all_prior_s1000`은 2022·2023·2024 Skill `1784.40`, `898.61`, `850.40`을 기록했고 고정 50:50 기준보다 세 시즌 모두 개선했다. EXP-020은 이 기준 위에 투수별 24개 `count_index × batter_hand` 문맥 효과를 smoothing `300`으로 축소한 뒤 rank `6` SVD로 압축했다. 고정 rank-6의 rolling Brier는 `0.244704584`, `0.247731144`, `0.247633803`, Skill은 `1789.60`, `907.54`, `869.92`이며 기준 대비 Skill 변화는 `+5.20`, `+8.93`, `+19.52`다. 2021~2024 OOF만 사용한 2025 prospective 선택도 rank `6`을 선택했으므로 이를 EXP-021 strict 최종 후보로 전체 학습·직렬화했다. 다만 13개 저장 후보의 같은-fold convex oracle조차 2023·2024 Skill이 `936.33`, `880.59`여서 1100에 미달했다. 현재 신호 집합의 단순 재가중만으로 목표를 달성할 수 있다는 근거는 없다.

EXP-021의 2025 Public 결과는 strict `1043.6074197937`, aggressive `1043.1871309639`였다. 두 후보 모두 이전 최고 EXP-013 `935.8108097065`를 크게 넘었지만, 과거 OOF만으로 선택한 strict가 최근 시즌 로컬 점수를 보고 사후 선택한 aggressive보다 `0.4202888298` 높았다. 따라서 최종 리더보드 선택은 strict로 변경한다. 이 결과는 최신 fold의 소폭 우위만 좇기보다 선택 절차의 시간적 독립성과 여러 시즌의 하방 안정성을 함께 보는 편이 더 안전하다는 근거로 기록한다.

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
├── experiments/              # EXP-002~021 학습·검증·패키징 코드
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

- EXP-021 strict Public `1043.6074197937`을 현재 최종 리더보드 선택으로 유지하기
- aggressive Public `1043.1871309639`은 strict보다 `0.4202888298` 낮았으므로 사후 선택 위험을 보여 준 진단 결과로 보존하기
- EXP-013 대비 strict 개선 `+107.7966100872`와 로컬 rolling 선택 기준의 성공·한계를 함께 기록하기
- 추가 탐색을 재개한다면 기존 예측의 재가중이 아니라 2023·2024에 시간 전이되는 새로운 행별 신호를 요구하기
- 학습·추론 피처 parity와 테스트 행 독립성 검사를 계속 유지하기

## 관련 글

- [데이터로 야구를 이해하고 예측하는 방법](https://mkisos.tistory.com/entry/lgaimers9-project-1)
- [모델 개발부터 코드 제출까지 전체 과정 정리](https://mkisos.tistory.com/entry/lgaimers9-project-2)
