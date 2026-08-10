# LG Aimers 9기 — 투구 제구 성공 확률 예측

투구 직전의 경기 상황과 과거 이력을 이용해 `control_success = 1`일 확률을 예측하는 과정을 기록한 학습·실험 저장소입니다.

운영진 RandomForest 베이스라인을 재현한 뒤, 피처 엔지니어링과 여러 트리 기반 모델을 같은 시간 기준 검증 방식으로 비교합니다. 최고 점수만 남기기보다 가설, 코드 변경, 실패한 실험과 배운 내용을 함께 기록하는 것을 목표로 합니다.

> 이 저장소에는 대회 원본 데이터, 학습된 모델, 예측 결과와 제출 ZIP을 포함하지 않습니다. 아래 점수는 리더보드 점수가 아닌 로컬 검증 결과입니다.

## 문제와 검증 방식

- Target: `control_success`
- 평가 지표: Brier Score 및 대회식 Brier Skill Score
- 주 검증: 2019~2023년 학습 → 2024년 검증
- 추가 검증: 2019~2022년 학습 → 2023년 검증
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
| EXP-011 | EXP-003 입력 표현 + LightGBM | **0.248043** | **706.03** | 현재 최고 |
| EXP-012 | EXP-003 입력 표현 + XGBoost | 0.248079 | 691.82 | EXP-011보다 하락 |

EXP-011은 2024년 검증에서 가장 좋았습니다. 다만 모델 비교 기준인 EXP-003이 2023년 추가 검증에서는 기준 모델을 넘지 못했으므로, 한 시즌의 결과만으로 일반화 성능을 확정하지 않고 여러 시간 구간에서 안정성을 확인할 예정입니다.

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
├── experiments/              # EXP-002~012 학습·검증 코드
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

예를 들어 현재 최고 실험인 EXP-011의 2024년 검증은 다음과 같이 실행합니다.

```bash
python experiments/train_exp011_lightgbm_hgb_features.py
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

- 여러 검증 시즌에서 안정적인 모델과 피처 찾기
- EXP-002의 6개 피처를 개별 제거해 기여도 확인하기
- 확률 보정 전후 Brier Score 비교하기
- 학습과 추론의 공통 피처 생성 로직 분리하기
- 평가 서버와 동일한 Python 3.11·scikit-learn 1.8.0 환경에서 최종 재현하기

## 관련 글

- [데이터로 야구를 이해하고 예측하는 방법](https://mkisos.tistory.com/entry/lgaimers9-project-1)
- [모델 개발부터 코드 제출까지 전체 과정 정리](https://mkisos.tistory.com/entry/lgaimers9-project-2)
