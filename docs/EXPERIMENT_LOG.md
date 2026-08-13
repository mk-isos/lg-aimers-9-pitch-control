# 모델 실험 기록

모델, 피처, 전처리 또는 확률 보정을 변경할 때마다 기록한다. 결과가 나빠진 실험도 삭제하지 않는다.

## 실험 운영 규칙

1. 실험 ID는 `EXP-001`부터 순서대로 사용한다.
2. 한 실험에서는 핵심 변경을 가능하면 하나만 적용한다.
3. 비교 실험은 같은 학습·검증 분할과 같은 평가식으로 실행한다.
4. 결과를 기록하기 전까지 기존 최고 모델을 덮어쓰지 않는다.
5. 모델 파일, 코드 파일, Git commit을 실험 ID와 연결한다.
6. 측정하지 않은 값은 추측해서 적지 않고 `측정 전`으로 남긴다.

## 실험 요약

| 실험 ID | 날짜 | 변경 사항 | 검증 기간 | Brier Score | Skill Score | 결론 |
| --- | --- | --- | --- | ---: | ---: | --- |
| EXP-000 | 2026-08-10 | 운영진 RandomForest 구조 확인 | 2024 예정 | 측정 전 | 측정 전 | 기준 구조 확인 |
| EXP-001 | 2026-08-10 | 베이스라인 검증 점수 재현 | 2024 | 0.248767 | 416.18 | 기준점 채택 |
| EXP-002 | 2026-08-10 | 상황 조합 피처 6개 추가 | 2024 | 0.248637 | 468.44 | 개선 확인, 추가 검증 필요 |
| EXP-003 | 2026-08-10 | HistGradientBoosting | 2024 | 0.248075 | 693.20 | 모델 비교 기준, 연도별 안정성 확인 필요 |
| EXP-004 | 2026-08-10 | HistGradientBoosting 용량·규제 조정 | 2024 | 0.248129 | 671.69 | EXP-003보다 하락 |
| EXP-005 | 2026-08-10 | 선수 ID Target Encoding | 2024 | 0.248493 | 525.87 | EXP-003보다 하락 |
| EXP-006 | 2026-08-10 | CatBoost 범주형 처리 | 2024 | 0.248303 | 602.05 | EXP-003보다 하락 |
| EXP-007 | 2026-08-10 | LightGBM 범주형 처리 | 2024 | 0.248866 | 376.80 | EXP-003보다 하락 |
| EXP-008 | 2026-08-10 | 로지스틱 회귀 | 2024 | 0.249790 | 6.63 | 기준 모델에 가까움 |
| EXP-009 | 2026-08-10 | 선수 ID 제거 + HistGradientBoosting | 2024 | 0.248094 | 685.73 | EXP-003에 근접 |
| EXP-010 | 2026-08-10 | 작은 트리 + 강한 규제 HistGradientBoosting | 2024 | 0.248151 | 663.02 | EXP-003보다 하락 |
| EXP-011 | 2026-08-10 | EXP-003 입력 표현 + LightGBM | 2024 | 0.248043 | 706.03 | 이전 단일 모델 기준 |
| EXP-012 | 2026-08-10 | EXP-003 입력 표현 + XGBoost | 2024 | 0.248079 | 691.82 | EXP-011보다 하락 |
| EXP-013 | 2026-08-10 | CatBoost + LightGBM 앙상블 + 선형 확률 보정 | 2024 | 0.247862497 | 778.37 | 앙상블 기준 실험 |
| EXP-014 | 2026-08-10 | 시간 가중치·피처·범주형·규제 LightGBM 탐색 | 2024 | 0.247857248 | 780.47 | 2024 로컬 최고, 시간 일반화 미확인 |
| EXP-015 | 2026-08-10 | 개선 LightGBM + 고정 2025 `-0.005` 동시 변경 | Public 2025 | - | 927.712979 | EXP-013 Public보다 하락, 원인 분리 불완전 |
| EXP-017 | 2026-08-10 | 현재 시즌 as-of 복원 + residual 후보 탐색 | 2021~2024 | 구성별 | 구성별 | residual 단독의 구조 변화 취약성 확인 |
| EXP-018 | 2026-08-10 | 계층적 기준값 + 그룹 효과 + 최근 residual 15% | 2024 | 0.247820261 | 795.28 | Public 895.84, EXP-013보다 하락해 비채택 |
| EXP-032 | 2026-08-12 | strict·aggressive·recency bounded consensus | 2022~2024 | 0.247618091 | 876.21 | recentaggr Public 1046.99, 현재 리더보드 선택 |
| EXP-033 | 2026-08-12 | TrackMan sequence·fine-pitch·시간 추세 residual | 2022~2024 | 0.247639600 | 867.60 | 2023·2024 하락, 비채택 |
| EXP-034 | 2026-08-12 | EXP-033 매핑 비용 범위 확장 | 2022~2024 | 0.247641946 | 866.66 | 매핑 확대도 하락, 비채택 |
| EXP-035 | 2026-08-12 | TrackMan 타자·matchup profile residual | 2022~2024 | 0.247638519 | 868.03 | 기준보다 하락, 비채택 |
| EXP-036 | 2026-08-12 | TrackMan count-transition control proxy | 2022~2024 | 0.247630668 | 871.18 | 2024 개선·2023 하락, 비채택 |
| EXP-037 | 2026-08-12 | low-rank source-season recency 정책 | 2022~2024 | 0.247628091 | 872.21 | hard gate 미달, 비채택 |
| EXP-038 | 2026-08-12 | 저장 예측 전체 same-fold convex oracle 감사 | 2022~2024 | 0.247577414 | 892.49 | 비배포 상한도 2023·2024 Skill 1000 미달 |
| EXP-039 | 2026-08-12 | prior-OOF 4-expert LightGBM stack | 2022~2024 | 0.247631638 | 870.79 | 2023 하락, 비채택 |
| EXP-040 | 2026-08-12 | recency:aggressive 70:30 bounded consensus | 2022~2024 | 0.247620314 | 875.32 | hard gate 미달, 비채택 |
| EXP-041 | 2026-08-12 | exact game-sequence TrackMan 정렬 residual | 2022~2024 | 0.247636022 | 869.03 | 2023 하락, 비채택 |
| EXP-042 | 2026-08-12 | EXP-041 + source recency2 정책 | 2022~2024 | 0.247647232 | 864.55 | 최신 성능 하락, 비채택 |
| EXP-043 | 2026-08-12 | exact fine-pitch control empirical Bayes | 2022~2024 | 0.247616774 | 876.74 | 2023·2024 개선, hard gate 미달 |
| EXP-044 | 2026-08-12 | exact TrackMan control + recentaggr 50:50 | 2022~2024 | 0.247611615 | 878.80 | Public 1046.95, recentaggr보다 낮아 비채택 |

## EXP-000 — 운영진 베이스라인 구조 확인

### 기본 정보

- 날짜: 2026-08-10
- 작성자: 김문기
- 상태: 구조 확인 완료, 검증 점수 미측정
- 관련 코드: 베이스라인 학습·추론 노트북, `baseline_submit/script.py`
- 모델: `baseline_submit/model/rf.pkl`
- 모델 생성 환경: scikit-learn 1.8.0

### 목적

첫 개선 실험 전에 운영진 베이스라인의 데이터 분할, 전처리, 모델 파라미터와 제출 구조를 이해한다.

### 확인한 학습 방식

- 학습 데이터: 2019~2023년
- 검증 데이터: 2024년
- 최종 모델: 2019~2024년 전체 데이터로 재학습
- Target: `control_success`
- 입력: `row_id`를 제외한 47개 피처

### 전처리

- 범주형: `top_bottom`, `game_type`, `base_state`
- 범주형 인코딩: `OrdinalEncoder`
- 알 수 없는 범주: -1
- 수치형 결측값: 중앙값

### 모델

```text
RandomForestClassifier
n_estimators=100
max_depth=10
min_samples_leaf=200
n_jobs=-1
random_state=42
```

### 결과

- Brier Score: 측정 전
- 기준 Brier Score: 측정 전
- Brier Skill Score: 측정 전
- 학습 시간: 측정 전
- 2024년 추론 시간: 측정 전

### 확인한 개선 후보

- 선수와 팀 ID의 처리 방식
- 카운트 및 경기 상황 조합 피처
- 최근 기록과 장기 기록의 차이
- Brier Score에 맞춘 확률 보정
- 다른 트리 모델과의 비교

### 결론

베이스라인 코드는 덮어쓰지 않고 보존한다. 다음 실험에서 평가 서버와 호환되는 환경으로 검증 점수를 재현한다.

## EXP-001 — 베이스라인 검증 점수 재현

### 기본 정보

- 날짜: 2026-08-10
- 작성자: 김문기
- 상태: 완료
- 기준 실험: EXP-000
- 관련 코드: `[Baseline_Train]_RandomForest를 활용한 모델 학습 및 피쳐엔지니어링 (학습).ipynb`
- 저장 모델: `model/rf.pkl`
- Git commit: 생성 예정

### 목적

2019~2023년 학습, 2024년 검증 조건에서 운영진 RandomForest 베이스라인 점수를 정확히 측정한다.

### 가설

운영진과 동일한 데이터, 전처리와 모델 파라미터를 사용하면 이후 개선 실험과 비교할 수 있는 기준 점수를 얻을 수 있다.

실제 실행은 현재 로컬 scikit-learn 1.9.0 환경에서 진행했다. 평가 서버는 1.8.0이므로 검증 기준점으로는 사용하되, 이 실행에서 저장한 최종 모델을 그대로 제출하지 않는다.

### 결과

- 전체 데이터: 1,475,092행 × 48열(47개 피처와 Target)
- 학습 행 수: 1,221,585
- 검증 행 수: 253,507
- 검증 실제 성공률: 0.486105
- 평균 예측 확률: 측정 전
- Brier Score: 0.248767
- 기준 Brier Score: 0.249807
- Brier Skill Score: 416.18
- 학습 시간: 40.4초
- 검증 추론 시간: 측정 전
- 전체 데이터 재학습 시간: 49.0초
- 모델 크기: 3,957,793바이트(약 3.8MiB)

### 로컬 샘플 추론 확인

- 테스트 행 수: 5
- 입력 피처 수: 47
- 예측 행 수: 5
- 결과 파일: `output/submission.csv`
- `row_id` 순서 일치: 통과
- 중복 `row_id` 없음: 통과
- 결측 예측값 없음: 통과
- 예측 확률 0~1 범위: 통과
- 샘플 평균 예측 확률: 0.478354
- 샘플 최소·최대 예측 확률: 0.447574~0.506271

### 결론

- [x] 채택
- [ ] 보류
- [ ] 폐기

결정 이유: 기존 베이스라인의 시간 기준 검증 점수와 실행 시간을 재현해 첫 개선 실험의 비교 기준을 확보했다.

### 다음 작업

EXP-002에서 동일한 검증 방식으로 상황 조합 피처를 추가하고 Brier 0.248767보다 낮아지는지 비교한다.

## EXP-002 — RandomForest + 상황 조합 피처 6개

### 기본 정보

- 날짜: 2026-08-10
- 작성자: 김문기
- 상태: 2024년 검증 완료, 최종 모델 저장 전
- 기준 실험: EXP-001
- 학습 코드: `experiments/train_exp002.py`
- 학습 노트북: `[EXP-002_Train]_RandomForest_상황피처_6개.ipynb`
- 제출 추론 코드: `submissions/EXP-002/script.py`
- 검증 결과: `artifacts/EXP-002/validation_metrics.json`
- 저장 모델: 아직 없음

### 실험 목적

RandomForest 설정을 유지한 상태에서 상황 조합 피처를 추가하면 제한된 트리 깊이에서도 상호작용을 더 쉽게 학습할 수 있는지 확인한다.

### 기준 실험과의 차이

- 기존 피처: 47개 유지
- 추가 피처: 6개
- 전체 피처: 53개
- 모델 파라미터: EXP-001과 동일
- 학습·검증 분할: EXP-001과 동일

### 추가 피처

1. `count_code`
2. `is_full_count`
3. `runner_in_scoring_position`
4. `same_hand`
5. `pitcher_batter_success_gap`
6. `pitcher_recent_success_delta`

모든 피처는 현재 투구 직전에 제공되는 원본 컬럼으로 만들었다. 테스트 데이터의 다른 행은 이용하지 않는다.

### 데이터 분할

- 학습: 2019~2023년, 1,221,585행
- 검증: 2024년, 253,507행
- 검증 실제 성공률: 0.486105

### 모델

```text
RandomForestClassifier
n_estimators=100
max_depth=10
min_samples_leaf=200
n_jobs=-1
random_state=42
```

### 실행 환경

- Python: 3.12.10
- pandas: 3.0.5
- numpy: 2.5.1
- scikit-learn: 1.9.0
- joblib: 1.5.3

### 결과

- Brier Score: 0.248636734
- 기준 Brier Score: 0.249806927
- Brier Skill Score: 468.44
- 평균 예측 확률: 0.501058
- 최소 예측 확률: 0.399060
- 최대 예측 확률: 0.635321
- 학습 시간: 49.8초
- 검증 추론 시간: 0.3초

### EXP-001과 비교

- Brier Score 변화: -0.000130266, 개선
- Validation Score 변화: +52.26
- 피처 수 변화: 47개 → 53개

### 결과 해석

2024년 검증에서는 상황 조합 피처 묶음이 베이스라인보다 좋은 결과를 냈다. 다만 실제 성공률 0.486105보다 평균 예측 확률 0.501058이 높아 전체적으로 성공 확률을 다소 높게 예측하는 경향이 남아 있다.

6개 피처를 동시에 추가했기 때문에 어떤 피처가 성능 향상에 기여했는지는 아직 알 수 없다. 2023년 추가 검증과 피처 제거 실험으로 안정성을 확인해야 한다.

### 결론

- [x] 1차 개선 후보로 채택
- [ ] 최종 제출 모델 확정

결정 이유: 같은 2024년 검증 조건에서 EXP-001보다 Brier가 낮아지고 Validation Score가 52.26 상승했다.

### 다음 작업

1. 2019~2022년 학습, 2023년 검증으로 방향성을 한 번 더 확인한다.
2. 평가 서버와 호환되는 Python 3.11 및 scikit-learn 1.8.0 환경을 만든다.
3. 호환 환경에서 2019~2024년 전체 데이터로 최종 재학습한다.
4. `submissions/EXP-002/model/rf_exp002.pkl`에 모델을 배치한다.
5. 로컬 추론과 ZIP 구조 검사를 통과한 뒤 제출한다.

## EXP-003~012 — 모델과 범주형 처리 비교

EXP-002의 상황 조합 피처 6개를 유지하면서 모델 종류, 모델 용량과 범주형 처리 방식을 비교했다. 모든 아래 결과는 2019~2023년 학습, 2024년 검증 조건이다.

| 실험 | 비교 내용 | Brier Score | Skill Score | EXP-003 대비 |
| --- | --- | ---: | ---: | --- |
| EXP-003 | HistGradientBoosting | 0.248075 | 693.20 | 기준 |
| EXP-004 | 반복 수·리프 수 증가 및 규제 조정 | 0.248129 | 671.69 | -21.51점 |
| EXP-005 | 투수·타자 ID Target Encoding | 0.248493 | 525.87 | -167.33점 |
| EXP-006 | CatBoost 범주형 직접 처리 | 0.248303 | 602.05 | -91.15점 |
| EXP-007 | LightGBM 범주형 직접 처리 | 0.248866 | 376.80 | -316.41점 |
| EXP-008 | 선형 로지스틱 회귀 | 0.249790 | 6.63 | -686.57점 |
| EXP-009 | 선수 ID 제거 + HistGradientBoosting | 0.248094 | 685.73 | -7.47점 |
| EXP-010 | 작은 트리 + 강한 규제 HistGradientBoosting | 0.248151 | 663.02 | -30.18점 |
| EXP-011 | EXP-003 입력 표현 + LightGBM | 0.248043 | 706.03 | +12.82점 |
| EXP-012 | EXP-003 입력 표현 + XGBoost | 0.248079 | 691.82 | EXP-011 대비 -14.20점 |

### 해석

- 2024년 검증에서는 EXP-011이 EXP-003보다 12.82점 높아 현재 가장 좋았다.
- EXP-009가 EXP-003에 근접해 고유 선수 ID를 그대로 수치형으로 사용하는 효과를 추가로 점검할 필요가 있다.
- 모델 복잡도를 높이거나 범주형 전용 모델을 사용하는 것만으로는 개선되지 않았다.
- EXP-003을 2023년 검증에 적용했을 때 Brier Score가 0.253493으로 기준 예측보다 나빴다. 시즌별 성공률 변화와 시간 일반화 문제가 남아 있다.
- 상세 수치와 실행 시간은 각 `artifacts/EXP-*/.../validation_metrics.json`에 보존한다.

## EXP-013 — CatBoost + LightGBM 보정 앙상블

### 기본 정보

- 날짜: 2026-08-10
- 상태: 검증 및 전체 데이터 최종 학습 완료
- 기준 실험: EXP-011
- 최종 학습 코드: `experiments/train_exp013_final_ensemble.py`
- 제출 추론 코드: `submissions/EXP-013/script.py`
- 검증 결과: `artifacts/EXP-013/2024/validation_metrics.json`

### 실험 목적과 가설

서로 다른 오차를 내는 CatBoost와 LightGBM의 2024년 검증 확률을 혼합하고, 평균 확률 편향을 줄여 첫 고득점 제출 후보를 만든다.

CatBoost의 범주형 직접 처리와 LightGBM의 원-핫 입력 표현이 서로 다른 패턴을 학습하므로 두 확률을 앙상블하면 단일 모델 EXP-011보다 Brier Score가 낮아질 것이라고 가정했다. 여기에 선형 확률 보정을 적용하면 확률의 전체적인 치우침도 줄일 수 있다고 보았다.

### 기준 실험과 달라진 점

- EXP-011의 LightGBM 단일 예측에서 CatBoost와 LightGBM의 가중 평균으로 변경했다.
- 가중 평균 확률에 선형 변환 후 0~1 범위로 자르는 확률 보정을 추가했다.
- EXP-002에서 만든 상황 피처 6개와 원본 피처를 유지했다.
- 최종 제출 모델은 scikit-learn pickle 대신 CatBoost와 LightGBM의 네이티브 형식으로 저장했다.

### 모델과 주요 파라미터

- CatBoost 가중치: 0.28719567
- LightGBM 가중치: 0.71280433
- 보정식: `clip(1.12708208 × ensemble - 0.07336118, 0, 1)`
- CatBoost: 범주형 9개 직접 처리, `iterations=400`, `depth=8`, `learning_rate=0.05`, `l2_leaf_reg=5.0`, `subsample=0.8`, `rsm=0.8`, `has_time=True`, `random_seed=42`
- LightGBM: 기본 범주형 3개 원-핫, `n_estimators=335`, `learning_rate=0.015`, `num_leaves=31`, `min_child_samples=500`, `subsample=0.85`, `colsample_bytree=0.9`, `reg_alpha=0.2`, `reg_lambda=3.0`, `random_state=42`

### 검증 기간

- 학습: 2019~2023년
- 검증: 2024년
- Target: `control_success`

### Brier Score와 Skill Score

- Brier Score: 0.247862497
- Skill Score: 778.37

위 값은 `artifacts/EXP-013/2024/validation_metrics.json`에 기록된 원값이다.

### 기준 실험 대비 변화

- EXP-011 Brier Score: 0.24804322476344168 → EXP-013: 0.247862497
- EXP-011 Skill Score: 706.0260563004462 → EXP-013: 778.37
- Brier Score는 낮아졌고 Skill Score는 높아져 같은 2024년 검증에서 개선됐다.

### 결과 해석

2024년 검증에서는 당시 단일 LightGBM보다 CatBoost를 함께 사용하고 확률을 보정한 구성이 더 좋은 확률 예측을 만들었다. EXP-013은 이후 EXP-014가 기록되기 전까지 로컬 최고였다.

다만 앙상블 가중치와 확률 보정 계수를 2024년 검증 결과에 맞춰 선택했기 때문에 같은 시즌 점수는 실제 일반화 성능보다 낙관적일 수 있다. 특히 2025년 평균 제구 성공률이 달라지면 선형 보정 효과도 달라질 수 있다.

### 최종 전체 학습

- 학습 데이터: 2019~2024년 전체 1,475,092행
- CatBoost 학습 시간: 179.1초
- LightGBM 학습 시간: 38.6초
- 저장 형식: CatBoost `.cbm`, LightGBM `.txt`
- CatBoost 모델 크기: 45,970,516바이트
- LightGBM 모델 크기: 1,230,044바이트
- 제출 ZIP 크기: 약 17MB
- 253,507행 피처 생성 및 모델 추론: 1.4초

### 재현 파일

- 최종 학습: `experiments/train_exp013_final_ensemble.py`
- 제출 추론: `submissions/EXP-013/script.py`
- 검증 지표: `artifacts/EXP-013/2024/validation_metrics.json`
- 패키지: `catboost==1.2.8`, `lightgbm==4.6.0`

### 채택 여부와 다음 실험

- [x] EXP-014 이전 당시 로컬 최고 실험으로 채택
- [x] 첫 앙상블 제출 후보로 채택
- [ ] 다른 검증 시즌에서 일반화 성능 확인 완료

다음 실험에서는 같은 모델 예측을 사용해 보정 전·후 제출 결과를 비교하거나, 다른 검증 시즌에서 앙상블 가중치와 보정 계수의 안정성을 확인한다. 실제 리더보드 점수가 나오면 로컬 점수와 함께 `docs/SUBMISSION_LOG.md`에 기록한다.

## EXP-014 — 시간축·피처·범주형·규제 LightGBM 탐색

### 기본 정보

- 날짜: 2026-08-10
- 상태: 2023·2024년 시간축 검증 완료
- 기준 실험: EXP-013
- 학습·검증 코드: `experiments/train_exp014_temporal_categorical_lgbm.py`
- Trackman 피처 코드: `experiments/trackman_features.py`
- 과거 시즌 Target 피처 코드: `experiments/temporal_target_features.py`
- 검증 결과: `artifacts/EXP-014/*/validation_metrics.json`

### 실험 목적과 가설

EXP-013보다 안정적인 단일 LightGBM 후보를 찾기 위해 시간 감쇠, 범주형 표현, 엔지니어드 피처, 트리 용량과 규제, Trackman 과거 요약 및 과거 시즌 Target 피처를 같은 시간축 분할에서 비교한다.

최근 시즌에 더 큰 가중치를 주고, 현재 투구 직전에 제공되는 누적·최근 기록을 표본 수에 따라 수축하며, 카운트와 경기 상황의 상호작용을 명시하면 2024년 확률 예측이 개선될 것으로 가정했다. 동시에 2023년을 별도 검증해 한 시즌에만 맞는 설정인지 확인하고자 했다.

### 기준 실험과 달라진 점

- EXP-013의 CatBoost·LightGBM 앙상블 대신 LightGBM 단일 모델의 입력과 규제를 탐색했다.
- 기존 6개 상황 피처를 넘어 카운트 상태, 경기 압박, 최근 기록 차이, 누적 표본 로그, 성공률 수축, 구종 구성 엔트로피를 추가했다.
- 기본 범주형 3개의 원-핫 처리와 네이티브 범주형 처리를 비교했다.
- 시즌 감쇠 가중치 `1.0`, `0.8`, `0.6`을 비교했다.
- `num_leaves`, `min_child_samples`, objective, seed를 바꿔 과적합과 모델 분산을 확인했다.
- Trackman 과거 요약과 과거 시즌 Target 피처는 별도 옵션으로 격리해 비교했다.

모든 추가 피처는 현재 행의 공식 입력 또는 검증 시즌 이전 데이터만 사용한다. 평가 데이터 내부의 행 간 집계는 사용하지 않는다.

### 모델과 주요 파라미터

2024년 최고 구성은 다음과 같다.

- 모델: `LGBMClassifier`
- objective: `binary`
- `n_estimators=1800`, early stopping 100회
- `learning_rate=0.015`
- `num_leaves=63`
- `min_child_samples=1000`
- `subsample=0.85`, `subsample_freq=1`
- `colsample_bytree=0.9`
- `reg_alpha=0.2`, `reg_lambda=4.0`
- `random_state=42`
- 피처 세트: `engineered`, 77개
- 범주형 처리: 기본 범주형 3개 원-핫 인코딩
- 시간 감쇠: `1.0`
- Trackman 피처: 사용하지 않음
- 과거 시즌 Target 피처: 사용하지 않음
- 최적 반복: 278
- 확률 보정: 검증 예측에 최소제곱 선형 변환 후 0~1 clip

### 검증 기간

주 검증:

- 학습: 2019~2023년, 1,221,585행
- 검증: 2024년, 253,507행

추가 검증:

- 학습: 2019~2022년, 976,060행
- 검증: 2023년, 245,525행

### Brier Score와 Skill Score

2024년 최고 구성의 `validation_metrics.json` 기록:

- 보정 전 Brier Score: 0.24799272977828155
- 보정 전 Skill Score: 726.2396611895094
- 보정 후 Brier Score: 0.24785724834181783
- 보정 후 Skill Score: 780.4741206669407
- 보정 scale: 1.0823101687544654
- 보정 intercept: -0.05207343908248591

같은 설정의 2023년 기록:

- 보정 전 Brier Score: 0.25118392215145385
- 보정 전 Skill Score: 0.0
- 보정 후 Brier Score: 0.24989886191192345
- 보정 후 Skill Score: 40.4545039712767
- 최적 반복: 25

### 주요 비교 결과

아래 값은 각 구성의 `validation_metrics.json`에 저장된 보정 후 지표다.

| 구성 | Brier Score | Skill Score | 해석 |
| --- | ---: | ---: | --- |
| base + 원-핫 | 0.24808599527335284 | 688.9046295784351 | 입력 기준 |
| legacy + 원-핫 | 0.2479093835178941 | 759.6039323366388 | 기존 상황 피처 개선 |
| engineered, leaves 31 | 0.2478975050802203 | 764.3589796958228 | 추가 피처 개선 |
| engineered, leaves 63, min child 500 | 0.24786463490194588 | 777.5172130167207 | 모델 용량 증가 개선 |
| engineered, leaves 63, min child 1000 | 0.24785724834181783 | 780.4741206669407 | 2024 최고 |
| engineered, leaves 127 | 0.2478824820073057 | 770.3728533340116 | 최고 구성보다 하락 |
| decay 0.8 + legacy | 0.24791198551579166 | 758.5623287544863 | 감쇠 없는 legacy보다 하락 |
| decay 0.6 + legacy | 0.24792876879427206 | 751.8438287119623 | 감쇠 강화 시 추가 하락 |
| Trackman 과거 요약 | 0.2479474552275529 | 744.3634783776298 | 매핑 노이즈로 채택하지 않음 |
| 과거 시즌 Target 피처 | 0.2478927286156798 | 766.2710421844521 | 최고 구성보다 하락 |
| regression L2 objective | 0.2478778451368275 | 772.229035041383 | binary보다 하락 |
| seed 7 | 0.2478772053916647 | 772.4851308875346 | seed 42보다 하락 |

### 기준 실험 대비 변화

- EXP-013: Brier Score `0.247862497`, Skill Score `778.37`
- EXP-014 2024 최고: Brier Score `0.24785724834181783`, Skill Score `780.4741206669407`
- 같은 2024년 검증에서는 Brier Score가 낮아지고 Skill Score가 높아졌다.
- 반면 2023년 보정 후 Skill Score는 `40.4545039712767`이므로 여러 시즌에서 안정적인 개선으로 볼 수 없다.

### 결과 해석

2024년에는 엔지니어드 피처와 `min_child_samples=1000`의 강한 리프 규제가 작은 개선을 만들었다. 시간 감쇠, 더 큰 트리, Trackman 요약, 과거 시즌 Target 피처 및 regression objective는 최고 구성보다 낮았다.

가장 중요한 결과는 2023년과 2024년의 차이다. 동일한 파라미터가 2024년에는 최고였지만 2023년에는 기준 예측에 가까운 수준까지 하락했다. 단일 시즌에서 선택한 피처와 선형 보정값만으로 다음 시즌 성능을 판단하면 낙관 편향이 커질 수 있음을 확인했다.

### 채택 여부와 다음 실험

- [x] 2024년 로컬 최고 구성으로 기록
- [ ] 최종 제출 모델로 채택
- [ ] 다중 시간축에서 일반화 확인

다음 실험은 2022·2023·2024의 rolling 검증 예측을 함께 사용해 설정을 선택한다. 단일 시즌 정답으로 affine 보정을 맞추는 대신, 이전 시즌에서 정한 보정식을 다음 시즌에 적용하거나 공식 `asof_*` 기준 확률의 잔차를 학습해 보정 과적합을 줄인다. Trackman 피처와 과거 시즌 Target 피처는 현재 구성으로는 채택하지 않는다.

## EXP-015~016 — Public 하락 원인 분리

### 실제 제출 결과

- EXP-013 Public: `935.8108097065`
- EXP-015 Public: `927.7129792368`
- EXP-015는 엔지니어드 LightGBM과 `season >= 2025`일 때의 고정 `-0.005`를 동시에 바꿨다.
- EXP-016은 고정 보정만 제거한 진단 후보지만 기대 개선이 작아 제출하지 않았다.

### rolling 재진단

`experiments/evaluate_exp013_rolling.py`로 EXP-013 구조를 각 검증 시즌 이전 데이터에 다시 학습했다. 고정 가중치는 2024에서 정해졌으므로 과거 fold 결과에는 look-ahead가 남지만, 구조 변화 민감도를 확인하는 진단으로 사용했다.

| 검증 시즌 | EXP-013 raw Skill | EXP-013 고정 제출식 Skill |
| ---: | ---: | ---: |
| 2022 | 2323.59 | 2326.97 |
| 2023 | -1303.57 | -1595.82 |
| 2024 | 726.18 | 778.37 |

raw 평균은 `582.06`, 최저는 `-1303.57`이다. 2023년 성공률이 2022년 `0.528920`에서 `0.499957`로 급락했을 때 과거 수준을 직접 예측하는 모델이 크게 과대 예측했고, 2024에서 맞춘 고정 affine을 적용하면 하락 폭이 더 커졌다. 따라서 EXP-015의 Public 하락은 `-0.005` 하나로 설명하기보다, 한 시즌에서 모델·앙상블·보정을 함께 선택한 구조적 취약성과 변경 원인 혼합으로 해석한다.

## EXP-017 — 현재 시즌 복원 residual 탐색

### 핵심 피처와 검증 규칙

`asof_pitcher_n × asof_pitcher_success_rate`와 같은 공식 누적값은 현재 행 직전까지의 성공 수를 복원한다. 학습 데이터에서 확정한 직전 시즌 종료 누적값을 빼면 테스트 다른 행을 집계하지 않고도 현재 행 시점의 이번 시즌 표본 수와 성공 수를 얻을 수 있다. 2020~2024 학습 데이터에서 음수 표본, 음수 성공, 표본보다 큰 성공 수가 한 건도 없음을 확인했다.

- 검증 시즌 반복 수는 바로 전 시즌에서만 정하고 검증 시즌 이전 데이터로 다시 학습했다.
- 확률 보정은 현재 fold보다 과거인 OOF만 사용했다.
- 같은 검증 시즌 정답으로 early stopping과 점수 평가를 함께 하지 않았다.
- 선수 ID 원값을 제거하고 표본 수·수축률·cold-start 상태를 사용했다.

### 주요 실패와 학습

- residual 전체 비중은 2021·2022에서 강했지만 2023 구조 단절에서 양의 잔차를 이월해 실패했다.
- 시즌별 residual 평균을 제거해도 고용량 모델은 2023에서 불안정했다.
- 직전 1시즌 모델은 2024에서 Skill `823.73`까지 올랐지만 2023 충격에는 취약했다.
- 따라서 최근 모델을 단독 채택하지 않고 안정적 기준값에 제한된 비중만 더하기로 했다.

모든 후보의 실행 결과는 `artifacts/EXP-017/*/validation_metrics.json`에 자동 저장했다.

## EXP-018 — constrained multiscale 제출 및 비채택

### 실험 목적과 가설

EXP-013과 EXP-015가 2023 구조 변화와 실제 2025 Public에서 흔들린 원인을 줄이는 것이 목적이다. 공식 `asof_*`에서 복원한 현재 시즌 수준을 안정적인 기준값으로 두고, 과거 상황 효과와 최근 residual의 영향력을 제한하면 단일 시즌 affine 보정보다 다음 시즌 Brier Score가 안정적일 것이라고 가정했다.

### 기준 실험과 달라진 점

- 기준 실험: EXP-013 CatBoost + LightGBM 앙상블과 2024 정답으로 맞춘 affine 보정
- EXP-018은 원시 선수 ID 대신 현재 시즌 투수·타자 표본과 계층적 shrinkage 확률을 사용한다.
- CatBoost를 제거하고 LightGBM을 절대 확률 모델이 아닌 직전 1시즌의 평균 제거 residual 모델로 제한한다.
- 과거 3시즌 그룹 효과와 recent residual 15%를 결합한다.
- 같은 검증 fold에서 맞춘 calibration과 임의의 2025 offset을 제거하고 identity를 사용한다.
- 2022·2023·2024 rolling-origin으로 평균·최저·수준 오차를 함께 확인한다.

### 구성

1. 투수 70% + 타자 30%의 현재 시즌 계층적 기준 확률
2. 과거 3시즌의 `count_index × pitcher_hand × batter_hand` 잔차 효과
3. 투수 `asof_pitcher_reverse_rate` 0.05 구간 조건부 효과를 그룹 효과에 30% 혼합
4. 직전 1시즌의 시즌 평균 제거 residual LightGBM을 15%만 반영
5. 확률 보정은 identity

LightGBM은 `iterations=200`, `num_leaves=15`, `min_child_samples=2000`, `learning_rate=0.015`를 사용한다. final residual 비중은 실험 스크립트에 고정한 15%이며, 아래 결과는 이 설정으로 생성한 `validation_metrics.json`의 기록이다.

### rolling-origin 결과

| 검증 시즌 | Brier | Skill | 예측 평균 | 실제 성공률 | 평균 차이 | 진단 slope | 진단 intercept |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 0.244537937 | 1856.48 | 0.522090 | 0.528920 | -0.006831 | 1.233869 | -0.115270 |
| 2023 | 0.248075365 | 769.85 | 0.500292 | 0.499957 | +0.000335 | 1.027132 | -0.013909 |
| 2024 | 0.247820261 | 795.28 | 0.491661 | 0.486105 | +0.005556 | 1.038154 | -0.024315 |

- 3시즌 평균 Skill: `1140.5374522029906`
- 3시즌 최저 Skill: `769.8531122158125`
- 이전 fold 평균 편향을 다음 fold에서 빼는 보정의 평균은 `1140.26`, 최저는 `749.36`으로 raw보다 최저 성능이 나빠 identity를 채택했다.
- 별도 EXP-013 rolling JSON에 기록된 2024 제출식은 Brier `0.24786249883749928`, Skill `778.3722991723918`이고 EXP-018은 Brier `0.24782026100344834`, Skill `795.2803300245504`다. 변화량은 각각 `-0.00004223783405099546`, `+16.908030852158618`이다.
- EXP-013 raw rolling 대비 평균 Skill 변화는 `+558.4727404299939`, 최저 Skill 변화는 `+2073.4214378369866`이다.

### 표본 수와 신규 선수 진단

2024 최종 후보의 주요 구간은 다음과 같다.

| 구간 | 행 수 | Skill | 평균 차이 |
| --- | ---: | ---: | ---: |
| 투수 현재 시즌 n=0 | 391 | -96.24 | +0.02534 |
| n=1~19 | 7,300 | 293.56 | +0.01962 |
| n=20~99 | 26,583 | 647.64 | +0.00617 |
| n=100~499 | 91,421 | 677.79 | +0.00544 |
| n>=500 | 127,812 | 900.61 | +0.00464 |
| 신규 투수 | 50,348 | 1018.17 | +0.00275 |
| 기존 투수 | 203,159 | 738.22 | +0.00625 |

신규 투수 전체는 양호하지만 현재 시즌 표본 `0~19`인 극초기 구간이 남은 핵심 위험이다. 이전 fold 구간별 평균 편향을 다음 시즌에 적용하는 추가 보정은 모든 시즌을 개선하지 못해 채택하지 않았다.

### Trackman과 calibration 결정

- 기존 Trackman 매핑 후보는 2024에서 최고 구성보다 낮았고 익명 ID 일대일 매핑 비용에 의존했다. EXP-018에서는 사용하지 않는다.
- 같은 fold 정답으로 affine을 맞추지 않는다.
- 임의의 2025 고정 offset을 사용하지 않는다.
- 최종 2025 추론은 현재 행 공식 입력과 2019~2024 학습 이력 상태만 사용한다.

### 결론

- [x] 다중 시즌 평균·최저와 최신 2024 성능을 근거로 제출 후보 선정
- [x] 전체 2019~2024 이력 상태와 2024 recent residual로 최종 학습
- [x] Public 결과 확인 후 최종 채택 여부 재평가

### Public 결과와 기준 대비 변화

- 제출 파일: `exp018_multiscale.zip`
- Public Score: `895.8368767677`
- EXP-013 Public Score: `935.8108097065`
- 기준 대비 변화: `-39.9739329388`
- EXP-015 Public Score `927.7129792368`보다도 낮았다.

평균 rolling Skill `1140.5374522029906`은 2022의 높은 Skill `1856.478914368609`에 크게 영향을 받았고, 최신 2023·2024는 각각 `769.8531122158125`, `795.2803300245504`였다. Public 하락은 현재 시즌 복원과 constrained residual이 과거 구조 단절의 최저 성능은 방어했지만 2025의 절대 확률 수준 또는 시즌 초 표본 구조에는 충분히 일반화되지 않았음을 보여준다. 특히 2024에서 투수 현재 시즌 표본 `0~19` 구간의 과대 예측이 이미 관찰됐으므로 이 구간 위험을 더 크게 반영했어야 했다.

- [ ] 최종 채택: EXP-018
- [x] 리더보드 선택 유지: EXP-013

다음 실험은 2022 고점에 좌우되는 단순 평균 대신 최신 시즌, 최저 시즌, 시즌별 예측 평균 오차에 제약을 두고 선택한다. EXP-013 대비 구성 요소를 한 번에 하나씩 바꾸고, 현재 시즌 `0~19` 표본 구간용 기준값은 이전 fold에서만 정한 규칙으로 검증한다.

## EXP-019 — 안정 residual backbone과 source-season Team EB

### 실험 목적과 가설

EXP-018의 2025 일반화 실패가 단순 확률 보정보다 행별 순위 신호 부족과 시즌 구조 변화에 있는지 진단했다. 공식 입력과 과거 학습 이력으로 만든 계층적 기준값 위에서 서로 다른 트리 residual을 보수적으로 결합하고, 시즌별 팀 효과를 강하게 축소하면 2023·2024 최저 성능을 함께 높일 수 있다고 가정했다.

### 기준 실험과 달라진 점

- 기준 실험: EXP-018의 계층적 as-of 기준값, 그룹 효과, recent residual 15%
- R행의 시즌 평균 제거 residual을 LightGBM과 HistGradientBoosting으로 따로 학습한다.
- 각 branch를 먼저 `[0, 1]`로 제한한 뒤 50:50으로 결합한다.
- 과거 OOF 시즌마다 투수 팀·타자 팀 효과를 별도로 추정하고, source season에 없는 key는 0으로 포함해 동일 가중 평균한다.
- 현재 검증 fold 정답과 테스트 다른 행의 집계를 효과 학습에 사용하지 않는다.

### 모델과 주요 파라미터

- LightGBM: `num_leaves=63`, `min_child_samples=1000`, `iterations=300`, residual weight `0.75`
- HistGradientBoosting: `max_leaf_nodes=15`, `max_depth=4`, `min_samples_leaf=3000`, `max_iter=160`, residual weight `1.0`
- backbone: 두 branch 50:50
- Team EB: 투수 팀·타자 팀 family 각각 50%, `all_prior_s1000`, source-season residual 평균 제거
- 확률 보정: identity

### 검증 기간과 결과

2021을 warm-up OOF로 만들고 2022·2023·2024를 보고 시즌으로 사용했다. 각 효과에는 검증 시즌보다 앞선 OOF 시즌만 사용했다.

| 검증 시즌 | 고정 50:50 Brier | 고정 50:50 Skill | Team EB Brier | Team EB Skill |
| ---: | ---: | ---: | ---: | ---: |
| 2022 | 0.244890133493 | 1715.13 | 0.244717539011 | 1784.40 |
| 2023 | 0.247853474558 | 858.61 | 0.247753471026 | 898.61 |
| 2024 | 0.247766621813 | 816.75 | 0.247682561558 | 850.40 |

`all_prior_s1000`은 평균 Skill `1177.8037005229169`, 최저 Skill `850.4028396019714`를 기록했다. JSON에 기록된 고정 50:50 기준 대비 평균 변화는 `+47.640347107965226`, 최저 변화는 `+33.65008951217169`이며 세 보고 시즌 모두 개선했다.

### 결과 해석과 채택 여부

팀 ID를 트리의 순서형 숫자로 직접 넣는 방식은 시간 전이가 불안정했지만, source season마다 평균을 제거한 강한 shrinkage 효과는 세 시즌에 같은 방향으로 작동했다. 다만 후보 비교 자체는 완전한 nested 선택이 아니고 최저 Skill도 1100에 미달한다.

- [x] 다음 실험의 고정 기준값으로 채택
- [ ] 이 실험만 최종 제출 모델로 채택

다음 실험에서는 팀 기준 위의 잔차를 선수×상황 문맥으로 분해하되, 현재 fold 성능으로 복잡도를 선택하지 않는 low-rank 구조를 검증한다.

## EXP-020 — 투수 문맥 low-rank EB와 재가중 상한 감사

### 실험 목적과 가설

EXP-019 Team EB가 놓치는 투수별 count·타자 손 문맥을 계층적으로 공유하는 것이 목적이다. 투수×24개 공식 문맥의 잔차 행렬을 SVD로 저차원화하면 포화된 세부 group보다 표본 부족을 줄이고 2023·2024에 함께 전이될 것으로 가정했다.

### 기준 실험과 달라진 점

- 기준 실험: EXP-019 `all_prior_s1000` Team EB
- 문맥: `count_index=4×balls_before+strikes_before`와 `batter_hand`의 24개 고정 조합
- source OOF 시즌별 잔차를 평균 제거한 뒤 투수×문맥 효과를 smoothing `300` 또는 `600`으로 축소한다.
- SVD rank 후보 `2, 4, 6, 8, 12`를 비교한다.
- 각 검증 시즌의 rank는 그보다 과거인 OOF fold의 최저 Skill, 평균 Skill, 낮은 rank 순서로만 선택한다.
- 별도로 13개 저장 후보의 nonnegative convex ensemble 상한을 감사한다.

### rolling-origin 결과

strict rank 경로는 2022 rank 2, 2023 rank 4, 2024 rank 6이었다.

| 검증 시즌 | 선택 rank | Brier | Skill |
| ---: | ---: | ---: | ---: |
| 2022 | 2 | 0.244692096549 | 1794.61 |
| 2023 | 4 | 0.247732737845 | 906.90 |
| 2024 | 6 | 0.247633803416 | 869.92 |

- strict 경로 평균 Skill: `1190.4779505072147`
- strict 경로 최저 Skill: `869.9211702032806`
- 2021~2024 OOF만 사용한 2025 prospective 선택: smoothing `300`, rank `6`
- 2025 정답 사용 여부: `false`

최종 고정 rank-6 후보의 2022·2023·2024 Brier는 `0.24470458400867237`, `0.2477311440988816`, `0.24763380341629648`이고 Skill은 `1789.5967932082258`, `907.5416355312283`, `869.9211702032806`이다. Team EB 기준 대비 Skill 변화는 JSON 기록상 `+5.199395859112428`, `+8.930770913562242`, `+19.518330601309117`로 모두 양수다.

### ensemble 상한과 결과 해석

13개 frozen 후보의 같은-fold 정답을 허용한 비배포 convex oracle도 다음 결과에 그쳤다.

| 검증 시즌 | oracle Brier | oracle Skill | Skill 1100 Brier 기준 | 판정 |
| ---: | ---: | ---: | ---: | --- |
| 2023 | 0.247659185595 | 936.33 | 0.247249998191 | 1100 불가 인증 |
| 2024 | 0.247607157188 | 880.59 | 0.247059050562 | 1100 불가 인증 |

Frank-Wolfe gap은 두 시즌 모두 `0.0`으로 기록됐다. 따라서 저장 후보의 단순 convex 재가중은 1100 격차를 닫지 못하며, 다음 개선에는 새로운 시간 안정적 행별 신호가 필요하다.

- [x] 과거 fold로 선택한 strict rank-6을 최종 패키지 후보로 채택
- [ ] R-specific, F-transfer, parametric, weighted ALS 등 사후 우수 후보를 주 모델로 채택

## EXP-021 — strict·aggressive 최종 패키지

### 실험 목적과 기준 실험 대비 변화

로컬 검증식을 전체 2019~2024 학습 상태와 네이티브/고정 모델 파일로 직렬화해 평가 환경에서 재학습 없이 실행하는 것이 목적이다. EXP-020 strict rank-6을 1순위로 유지하고, 최신 2023·2024가 조금 높은 `r_gated_team_pc_all`을 사후 선택 위험이 있는 2순위 진단 후보로만 패키징했다. 두 후보 모두 affine 또는 고정 offset을 사용하지 않는다.

### 최종 구성과 주요 파라미터

공통 backbone은 EXP-019의 branch별 clip 후 LightGBM·HistGradientBoosting 50:50이며, 2021~2024 source OOF의 Team EB를 missing=0 포함 동일 가중 평균한다.

- strict: Team EB 위 투수×24문맥 low-rank, smoothing `300`, rank `6`
- aggressive: `game_type=R`이면 Team EB, `F`이면 고정 backbone을 선택한 뒤 투수×count×타자 손 EB smoothing `600`을 두 regime 모두에 적용
- LightGBM 모델 형식: native text
- HistGradientBoosting 모델 형식: 각 트리를 JSON으로 내보내 NumPy로 추론하고 원본 모델과 parity 확인
- 제출 `requirements.txt`: 평가 이미지 기본 NumPy·pandas·scikit-learn·joblib을 재설치하지 않고 `lightgbm==4.6.0`만 설치
- 로컬 생성 환경: NumPy `2.5.1`, pandas `3.0.5`, LightGBM `4.6.0`, scikit-learn `1.9.0`, joblib `1.5.3`

### 실제 validation_metrics.json 결과

| 후보 | 2022 Brier / Skill | 2023 Brier / Skill | 2024 Brier / Skill | 평균 Skill | 최저 Skill |
| --- | ---: | ---: | ---: | ---: | ---: |
| strict rank-6 | 0.244704584009 / 1789.60 | 0.247731144099 / 907.54 | 0.247633803416 / 869.92 | 1189.019866 | 869.921170 |
| aggressive R/F gate | 0.245030403600 / 1658.83 | 0.247659185595 / 936.33 | 0.247622900300 / 874.29 | 1156.480766 | 874.285787 |

strict는 Team EB 기준보다 세 시즌 모두 개선했고 2025 rank 선택도 과거 OOF만 사용했다. aggressive는 2023·2024는 strict보다 높지만 2022가 크게 낮고 후보 정의가 post-hoc/non-nested이므로 기대값보다 변동성 진단용 성격이 강하다.

### 실행·구조·CRC 검증

- `submit_exp021_strict.zip`: `1,942,657` bytes, SHA256 `e4b1cd4868551df0ec9886bd5dae9c6e3f9707029c9cad31ebd9bba5bb7a8be5`, CRC 통과
- `submit_exp021_aggr.zip`: `1,942,498` bytes, SHA256 `68fb18791010794cc4670403c56e24848629ea74cfff02d803010cb01c583191`, CRC 통과
- 두 ZIP 모두 최상위 `script.py`, `requirements.txt`, `model/` 구조와 총 12개 파일을 확인했다.
- 별도 임시 디렉터리에서 train 없이 5행 smoke 추론을 실행했고 row_id 순서·중복·결측·확률 범위를 통과했다.
- strict와 aggressive의 smoke 추론 시간은 각각 `1.3881361484527588`초, `1.2347588539123535`초였다.
- 원본 HistGradientBoosting과 JSON-tree NumPy 추론은 고정 4,096행에서 최대 절대 오차 `0.0`으로 parity를 통과했다.
- batch/singleton 및 행 순열 예측은 bitwise 동일했고 테스트 다른 행 집계가 없음을 확인했다.

### 실제 Public 결과

| 후보 | 제출 일시 | Public Score | 실행 시간 | 판정 |
| --- | --- | ---: | ---: | --- |
| strict rank-6 | 2026-08-11 08:58:48 | **1043.6074197937** | 8초 | **최종 선택** |
| aggressive R/F gate | 2026-08-11 09:03:41 | 1043.1871309639 | 8초 | strict보다 0.4202888298 낮아 비채택 |

strict는 이전 리더보드 최고 EXP-013 `935.8108097065`보다 `107.7966100872` 높았고, aggressive도 `107.3763212574` 높았다. 최근 2023·2024 rolling에서는 aggressive가 소폭 우세했지만 실제 Public에서는 과거 OOF만으로 rank를 선택하고 2022 하방도 함께 보호한 strict가 더 높았다. 차이는 작지만 방향은 사후 선택 위험을 보수적으로 평가한 사전 결론과 일치한다.

### 채택 여부와 다음 작업

- [x] 최종 리더보드 선택: EXP-021 strict rank-6, Public `1043.6074197937`
- [x] EXP-021 aggressive 진단 제출: Public `1043.1871309639`, strict보다 낮아 비채택
- [x] 모델·예측 배열·ZIP은 Git 제외
- [x] 이전 최고 EXP-013 대비 strict Public `+107.7966100872` 개선 확인

두 후보의 최종 결과가 확인됐으므로 EXP-013 대신 strict를 리더보드 선택으로 확정한다. aggressive의 최신 fold 우위는 2025 Public 우위로 이어지지 않았으므로, 추가 실험을 재개한다면 post-hoc 후보 확대보다 2023·2024에 함께 전이되고 과거 fold만으로 선택 가능한 새로운 행별 신호가 전제다.

---

## EXP-022 — 누적 outcome taxonomy 보조 감독과 temporal Ridge

### 실험 목적과 가설

기존 13개 frozen 후보의 same-fold convex oracle도 2023·2024 Skill이 각각 `936.3250374376797`, `880.587899179186`에 그쳤으므로 재가중 대신 새로운 행별 감독 신호를 검증했다. 같은 투수·시즌의 `asof_pitcher_n+1` 상태에서 누적 count 증분을 구해 reverse·middle·ball·strike 결과를 복원하고, 이 결과를 예측한 확률이 성공 residual의 시간 안정적 보조 표현이 될 것으로 가정했다.

### 기준 실험과 달라진 점

- 기준: EXP-021 strict 고정 smoothing `300`, rank `6` OOF
- 파일 순서나 `row_id` 순서를 가정하지 않고 `(pitcher_id, season, asof_pitcher_n)` unique key로 후속 상태를 찾는다.
- 같은 시즌 내부의 `n→n+1` count 증분이 binary인 행만 보조 label로 사용한다.
- reverse·middle·ball·strike별 prior-season HistGradientBoostingClassifier를 학습한다.
- 6개 보조 확률 표현으로 EXP-021 strict의 residual을 Ridge로 예측한다.
- correction scale은 `0.25`, `0.50`만 사전 고정하고 prior OOF 최저 Skill, 평균 Skill, 작은 scale 순으로 선택한다.
- same-fold와 5-fold cross-fit Ridge는 진단용이며 배포 후보 선택에는 사용하지 않는다.

### 모델과 주요 파라미터

- 보조 모델: `HistGradientBoostingClassifier`
- `learning_rate=0.025`, `max_iter=160`, `max_leaf_nodes=15`, `max_depth=4`
- `min_samples_leaf=3000`, `l2_regularization=30`, `max_bins=127`, `max_features=0.70`
- stable row-local 피처 `84`개, source-season equal weighting
- residual: `Ridge(alpha=5000, fit_intercept=False)`
- 확률 보정: identity

### label 복원 검증

- train 전체 행: `1,475,092`
- unique key 행: `1,475,092`
- duplicate key 행: `0`
- candidate pair: `1,472,832`
- 유효 binary pair: `1,472,040`
- invalid delta 행: `792`
- 복원 success와 `control_success` mismatch: `0`
- 원본 행 순열 전후 `row_id` 기준 label parity: 통과

### rolling-origin 결과

| 검증 시즌 | EXP-021 strict Brier / Skill | 선택 scale | EXP-022 Brier / Skill | same-fold Ridge Skill | 5-fold cross-fit Skill |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 0.244704584009 / 1789.60 | 0.50 | 0.244581625862 / 1838.95 | 1893.08 | 1888.59 |
| 2023 | 0.247731144099 / 907.54 | 0.50 | 0.247821222482 / 871.51 | 919.25 | 915.97 |
| 2024 | 0.247633803416 / 869.92 | 0.25 | 0.247627979502 / 872.25 | 873.42 | 870.46 |

- 평균 Skill: `1194.2359895912562`
- 최저 Skill: `871.5102821166498`
- 최신 2024 Skill: `872.2525365782108`
- 시즌별 Skill 1100 통과: `false`
- 세 시즌 모두 EXP-021 strict 이상: `false`
- full fit 및 ZIP 허가: `false`

### 기준 실험 대비 변화와 해석

2022는 기준 `1789.5967932082258`에서 `1838.9451500789078`로 높아졌고 2024도 `869.9211702032806`에서 `872.2525365782108`로 소폭 높아졌다. 반면 2023은 기준 `907.5416355312283`에서 `871.5102821166498`로 하락했다. 보조 label 자체는 거의 전 행에서 정확하게 복원됐지만, 보조 결과 확률과 성공 residual의 관계가 2022에서 2023으로 전이되지 않았다.

현재 fold 정답을 허용한 고정 Ridge조차 2023·2024 Skill `1100`에 도달하지 못했으므로 correction scale을 추가 탐색하거나 affine 보정을 붙일 근거가 없다. 이 결과는 label 복원 가능성과 최종 target에 유용한 시간 안정적 신호가 서로 다른 문제임을 보여 준다.

### 채택 여부와 다음 실험

- [ ] 최종 후보 채택
- [x] linear outcome-probability residual family 중단
- [x] 전체 학습·모델·ZIP 생성 금지
- [x] EXP-021 strict 리더보드 선택 유지

추가 실험은 같은 보조 확률의 weight 탐색이 아니라 success·reverse-only·middle-only·reverse-and-middle·other-failure의 상호 배타적 joint taxonomy를 직접 학습하는 단일 bounded multiclass 구조만 검토한다. 이 구조도 2023·2024 same-fold 또는 temporal gate를 통과하지 못하면 outcome taxonomy branch 전체를 중단한다.

---

## EXP-023~026 — joint outcome taxonomy의 시간 전이 감사

### 실험 목적과 가설

EXP-022의 독립 보조 확률은 same-fold ceiling이 낮았으므로, 중첩 outcome을 하나의 shared multiclass 구조로 바꾸면 target 성공과 failure subtype을 함께 학습해 더 안정적인 split을 얻을 수 있는지 확인했다. same-fold 신호가 확인된 뒤에는 pooled 과거 학습의 전이 실패를 source-season expert, 행별 regime gate, 행별 trend 외삽으로 각각 분리 진단했다.

### joint label과 기준 실험 대비 변화

유효 pair `1,472,040`개는 다음 5개 class로 전부 분할됐고 invalid overlap은 `0`이었다.

| class | 행 수 |
| --- | ---: |
| success | 770,759 |
| reverse-only | 287,063 |
| middle-only | 170,000 |
| reverse+middle | 50,208 |
| other-failure | 194,010 |

공통 기준은 EXP-021 strict 고정 rank-6 OOF이며, 모든 deployable 후보의 model·expert·blend 선택에는 validation season보다 과거인 source만 사용했다.

### EXP-023 모델과 결과

- 모델: 5-class `HistGradientBoostingClassifier`
- 파라미터: EXP-022와 같은 `max_iter=160`, `max_leaf_nodes=15`, `max_depth=4`, `min_samples_leaf=3000`, `l2_regularization=30`
- blend 후보: `0.10`, `0.25`, `0.50`
- 선택: prior OOF 최저 Skill, 평균 Skill, 작은 weight 순

| 시즌 | direct multiclass Skill | 선택 weight | 선택 Brier | 선택 Skill | same-fold best Skill |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 2077.80 | 0.50 | 0.244125145341 | 2022.15 | 2710.55 |
| 2023 | 711.52 | 0.50 | 0.247876693515 | 849.32 | 1417.94 |
| 2024 | 764.54 | 0.10 | 0.247629643815 | 871.59 | 1331.79 |

same-fold best는 2023·2024 모두 1100을 넘었으므로 taxonomy 표현에 현재 시즌 내 신호가 있음은 확인됐다. 그러나 prior-only direct와 blend는 2023에서 기준 `907.5416355312283`보다 낮았다.

### EXP-024 source-season expert bagging

2019~2023 각 source season만으로 5개 multiclass expert를 학습하고 `last`, `equal`, `recency2`, `median`, `consensus` 정책과 blend `0.25`, `0.50`의 10개 후보를 사전 고정했다.

| 시즌 | 선택 후보 | Brier | Skill |
| ---: | --- | ---: | ---: |
| 2022 | median_w050 | 0.244334536221 | 1938.11 |
| 2023 | median_w050 | 0.247917870343 | 832.85 |
| 2024 | last_w025 | 0.247609820933 | 879.52 |

2024는 EXP-021 strict보다 높았지만 2023의 사후 최고 후보조차 Skill `892.3904772110358`로 기준에 못 미쳤다. 전역 source bagging으로 2023 구조 단절을 해결할 수 없었다.

### EXP-025 row-local regime similarity gate

stable 84개 공식 피처로 현재 행이 과거 어느 source season과 유사한지 분류하고 그 확률로 source expert를 결합했다. validation/test 행 간 집계는 없었다.

| 시즌 | 평균 최대 gate weight | gate가 주로 선택한 source | 선택 Skill |
| ---: | ---: | --- | ---: |
| 2022 | 0.9997 | 2021 | 1863.45 |
| 2023 | 0.9990 | 2021 | 900.11 |
| 2024 | 0.9768 | 2019 | 865.60 |

gate가 unseen season에서 확률을 거의 one-hot으로 외삽했고, 2023을 2021로, 2024를 2019로 보내며 유용한 expert를 안정적으로 선택하지 못했다.

### EXP-026 행별 source expert trend

저장된 source expert 예측만 사용해 `last + 0.25/0.50 × 직전 차이`, source-year 선형 외삽, `last±0.03` 제한 외삽을 만들고 base blend `0.10`, `0.25`를 prior OOF로 선택했다.

| 시즌 | 선택 후보 | Brier | Skill |
| ---: | --- | ---: | ---: |
| 2022 | delta025_w010 | 0.244480296881 | 1879.61 |
| 2023 | delta025_w010 | 0.247735158345 | 905.94 |
| 2024 | linear_w010 | 0.247625302031 | 873.32 |

2024는 소폭 개선됐지만 2023은 기준보다 낮았고 세 시즌 1100 gate는 `false`였다.

### 결과 해석과 채택 여부

joint taxonomy는 same-fold에서 1100을 넘을 만큼 target 관련 신호를 표현했지만, 그 관계가 다음 시즌으로 이전되지 않았다. source를 pooled·bagging·similarity gating·trend extrapolation으로 바꿔도 2023과 2024를 동시에 개선하지 못했다. 이는 모델 혼합 방식보다 feature-target 관계 자체의 시즌 전환이 병목이라는 증거다.

- [ ] EXP-023~026 채택
- [x] 2022·2023·2024 균일 1100 gate 실패
- [x] outcome taxonomy branch 전체 중단
- [x] final fit·모델·ZIP 생성 금지
- [x] EXP-021 strict 리더보드 선택 유지

다음 탐색은 동일 source expert의 weight·gate·trend를 추가하지 않는다. 규정 안에서 기존 공식 피처와 독립적인 새 행별 정보가 확인되지 않으면 제출 후보를 새로 만들지 않는다.

---

## EXP-027~031 — logit 문맥 효과와 pitchmix 보조 supervision

### 실험 목적과 가설

EXP-021 strict의 2023·2024 병목이 additive 확률 residual의 표현 한계인지, taxonomy class prevalence drift인지, 제공된 누적 pitchmix에서 복원할 수 있는 구종군 supervision의 부재인지 순서대로 분리했다. 모든 후보는 현재 검증 시즌보다 과거인 OOF만 학습에 사용했고 평가 행끼리 집계하지 않았다.

### 기준 실험

고정 기준은 EXP-021 strict rank-6 OOF다.

| 시즌 | Brier | Skill |
| ---: | ---: | ---: |
| 2022 | 0.244704584009 | 1789.60 |
| 2023 | 0.247731144099 | 907.54 |
| 2024 | 0.247633803416 | 869.92 |

### EXP-027 — logit-offset pitcher-context EB

- 가설: 투수×`count_index × batter_hand` 효과를 확률에 더하지 않고 odds-ratio로 추정하면 현재 행의 as-of 기준 확률과 더 자연스럽게 결합된다.
- 모델: source-season별 penalized logistic offset, ridge `75`, rank `6` SVD
- nuisance source global logit offset: 추정 후 폐기
- 고정 effect weight: `0.50`, `1.00`

| 시즌 | `logit_offset_w100` Brier | Skill |
| ---: | ---: | ---: |
| 2022 | 0.244690717086 | 1795.16 |
| 2023 | 0.247728748411 | 908.50 |
| 2024 | 0.247636243208 | 868.94 |

2023은 기준보다 소폭 높지만 2024는 낮았다. additive 표현이 최신 시즌 병목의 원인이라는 가설을 기각했다.

### EXP-028 — prevalence-invariant joint taxonomy

- 가설: 각 source season과 5개 outcome class에 같은 학습 질량을 주면 시즌별 class prevalence drift를 제거하고 공통 경계만 남길 수 있다.
- 모델: 5-class `HistGradientBoostingClassifier`
- 파라미터: `max_iter=160`, `max_leaf_nodes=15`, `max_depth=4`, `min_samples_leaf=3000`, `l2_regularization=30`
- source-only score 중심화·표준화, 고정 logit weight `0.02`, `0.05`, `0.10`

| 시즌 | `invariant_logit_w020` Brier | Skill |
| ---: | ---: | ---: |
| 2022 | 0.244547494472 | 1852.64 |
| 2023 | 0.247763345026 | 894.66 |
| 2024 | 0.247623456306 | 874.06 |

2024는 개선됐지만 2023은 악화됐다. 단순 class prevalence가 아니라 피처와 outcome의 조건부 관계가 시즌 사이에 바뀐다는 증거다.

### EXP-029 — pitch group×outcome 15-class taxonomy

누적 `asof_pitcher_pitchmix_n`과 fastball·breaking·offspeed rate의 keyed 다음 상태를 이용해 train-only 현재 구종군 label을 복원했다.

- 전체 행: `1,475,092`
- unique pitcher-state key: `1,475,092`
- 연속 pair 및 유효 one-hot label: `1,472,832`
- invalid pair: `0`
- 2019~2024 모든 시즌에서 pitchmix 표본 수 증분은 정확히 `1`, 세 count 증분은 정확히 one-hot
- 추론 입력에는 현재 투구 실제 구종을 사용하지 않음

복원 구종군 3개와 outcome taxonomy 5개를 교차한 15-class HGB의 success class 합을 EXP-021 strict와 prior-only weight로 결합했다.

| 시즌 | 선택 Brier | 선택 Skill |
| ---: | ---: | ---: |
| 2022 | 0.244539915730 | 1855.69 |
| 2023 | 0.247781700121 | 887.32 |
| 2024 | 0.247632499827 | 870.44 |

새 label은 정확히 복원됐지만 2023 전이가 실패했다.

### EXP-030 — pitch-selection propensity residual

15-class 결합 실패와 구종 성향 자체의 가치를 분리하기 위해 3-class 구종군 HGB를 별도로 학습했다. 예측 확률 3개, 공식 장기 pitchmix와의 차이 3개, entropy와 최대 확률을 합친 8개 표현만 Ridge residual에 사용했다.

- Ridge alpha: `5000`
- source residual: 시즌별 중심화
- 고정 correction scale: `0.25`, `0.50`

| 시즌 | `pitch_residual_w025` Brier | Skill |
| ---: | ---: | ---: |
| 2022 | 0.244694088083 | 1793.81 |
| 2023 | 0.247737897344 | 904.84 |
| 2024 | 0.247635566184 | 869.22 |

구종 선택 성향 residual은 2023·2024에서 모두 기준보다 낮았다.

### 결과 해석과 채택 여부

- [ ] EXP-027~030 채택
- [x] 세 시즌 균일 Skill 1100 gate 실패
- [x] 현재 fold label을 학습·선택·보정에 사용하지 않음
- [x] 테스트 행 집계 없음
- [x] 모델 전체 학습 및 ZIP 생성 없음
- [x] EXP-021 strict 유지

odds-ratio 재매개변수화, class prevalence 균형, 새로운 pitchmix 보조 label 모두 2023·2024를 함께 개선하지 못했다. 이 네 후보의 추가 weight 탐색은 중단한다.

### EXP-031 — pitch-group conditional mixture-of-experts

15-class shared split과 저차원 residual의 실패를 분리하기 위해 fastball·breaking·offspeed마다 독립된 success HGB를 학습했다. 현재 행의 실제 구종은 사용하지 않고 EXP-030의 prior-only 3-class propensity로 세 expert를 확률 혼합했다.

- success expert: 구종군별 `HistGradientBoostingClassifier` 3개
- propensity: EXP-030 prior-only 3-class HGB 예측
- 공통 파라미터: `max_iter=160`, `max_leaf_nodes=15`, `max_depth=4`, `min_samples_leaf=3000`, `l2_regularization=30`
- 고정 base blend: `0.10`, `0.25`, `0.50`; 이전 OOF 최저 Skill 우선 선택

| 시즌 | 선택 Brier | 선택 Skill |
| ---: | ---: | ---: |
| 2022 | 0.244277269498 | 1961.10 |
| 2023 | 0.247854134076 | 858.35 |
| 2024 | 0.247636685474 | 868.77 |

2022의 개선 폭은 컸지만 2023에서 기준 `907.54`보다 크게 하락했고 2024도 기준에 못 미쳤다. 구종군별 조건부 성공 관계도 시간 전이가 안정적이지 않으므로 EXP-031은 비채택하고 pitchmix branch를 종료한다.

---

## EXP-032 — strict·aggressive·recency bounded consensus

### 목적과 설계

EXP-021 strict, EXP-021 aggressive R/F gate, EXP-037 recency2 low-rank branch는 모두 과거 OOF 기반 시간 검증을 거쳤다. 이 실험에서는 branch 자체나 데이터 피처를 새로 탐색하지 않고, 아래의 고정 조합을 제출 후보로 패키징했다.

| 후보 | strict | aggressive | recency |
| --- | ---: | ---: | ---: |
| dualrank | 0.50 | 0.00 | 0.00 |
| stableaggr | 0.50 | 0.50 | 0.00 |
| recentaggr | 0.00 | 0.50 | 0.50 |

- strict: all-row pitcher-context low-rank SVD, `smoothing=300`, `rank=6`
- aggressive: R/F-gated team base + pitcher-count empirical-Bayes correction
- recency: source season별 pitcher-context low-rank SVD, `smoothing=300`, `rank=6`, 가중치 `1:2:4:8`
- 공통 backbone: LightGBM residual + HistGradientBoosting residual 50:50
- source season: `2021~2024`; 보고 fold: `2022~2024`
- 현재 fold 정답으로 component model을 적합하지 않았고, test 행 집계도 사용하지 않았다.
- calibration: identity; 고정 offset 없음

### 로컬 결과

| 후보 | 2022 Brier / Skill | 2023 Brier / Skill | 2024 Brier / Skill | 평균 Skill | 최저 Skill |
| --- | --- | --- | --- | ---: | ---: |
| dualrank | 0.244690656 / 1795.19 | 0.247716002 / 913.60 | 0.247632158 / 870.58 | **1193.12** | 870.58 |
| stableaggr | 0.244851775 / 1730.52 | 0.247686394 / 925.44 | 0.247622396 / 874.49 | 1176.82 | 874.49 |
| recentaggr | 0.244851775 / 1730.52 | 0.247685310 / 925.88 | 0.247618091 / 876.21 | 1177.54 | **876.21** |

평균만 보면 dualrank가 가장 높지만, 최신 2024와 하방 성능은 recentaggr가 가장 높았다. 단, 이 고정 가중치들은 같은 결과를 보고 고른 완전한 nested selection이 아니라 bounded post-hoc consensus 진단이라는 한계를 명시한다.

### 패키지와 재현성

- 생성 코드: `experiments/build_exp032_consensus_candidates.py`, `experiments/exp021_submission_inference.py`
- 공통 inference 확장: `experiments/exp021_submission_inference.py`
- 결과 artifact: `artifacts/EXP-032/consensus_candidates/validation_metrics.json`
- ZIP 구조: 최상위 `script.py`, `requirements.txt`, `model/`만 포함
- QA: 세 후보 모두 CRC·5행 smoke inference·행 순서·중복·결측·확률 범위 검사 통과

### Public 결과와 결론

2026-08-12에 확인한 Public Score는 dualrank `1042.9008134487`, stableaggr `1045.1827084551`, recentaggr `1046.9889925352`였다.

- [ ] dualrank 채택
- [ ] stableaggr 채택
- [x] recentaggr 채택 — 현재 확인 기준 리더보드 1위

strict·aggressive 단독 후보가 각각 `1043.6074197937`, `1043.1871309639`였고, recentaggr는 aggressive보다 `+3.8018615713` 높았다. 현재 선택은 recency low-rank와 R/F-gated aggressive 신호의 50:50 결합이다.

다음 작업은 bounded consensus 비율 탐색이 아니라 TrackMan처럼 기존 branch와 독립적인 새 행별 신호의 과거 OOF 검증이다.

---

## EXP-033~044 — TrackMan 신호·source policy·결합 후보

### 공통 목적·가설과 검증 기준

목적은 EXP-021 strict의 2023·2024 병목을 기존 예측의 추가 재가중이 아닌, 과거 TrackMan 이력과 정확 정렬된 행별 상태로 완화할 수 있는지 확인하는 것이다. EXP-033~037·039·041~043은 2021~2024 outer fold 중 2022~2024를 보고했고, 현재 validation season보다 과거의 OOF season만 residual 적합·후보 선택에 사용했다. EXP-040·044는 고정된 배포 branch만 결합했고, 테스트 행 집계는 사용하지 않았다.

고정 비교 기준 EXP-021 strict는 2022/2023/2024 Brier `0.244704584009`/`0.247731144099`/`0.247633803416`, Skill `1789.60`/`907.54`/`869.92`다. 아래 Brier·Skill은 각 JSON의 선택 후보 2024 값이며, 변화는 같은 기준 대비 Skill 변화다.

| 실험 | 목적·기준 대비 변경 | 모델·주요 파라미터 | 2024 Brier / Skill | 기준 대비 Skill | 해석·채택 여부 |
| --- | --- | --- | --- | ---: | --- |
| EXP-033 | TrackMan sequence·fine-pitch·시간 추세 추가 | LightGBM residual, 208 features, `iter=200`, `lr=.015`, leaves `7`, min child `2000`, mapping cost `.10`, clip `.03` | 0.247639600 / 867.60 | -2.32 | 2022 개선이 2023·2024로 전이되지 않아 비채택 |
| EXP-034 | EXP-033의 mapping cost를 `.50`으로 확장 | EXP-033 동일, mapping cost `.50` | 0.247641946 / 866.66 | -3.26 | 넓은 매핑도 최신 성능을 개선하지 못해 비채택 |
| EXP-035 | TrackMan 타자와 pitcher×batter matchup profile 추가 | LightGBM residual, 116 features, `iter=200`, leaves `7`, min child `2000`, mapping cost `.02` | 0.247638519 / 868.03 | -1.89 | 기준보다 낮아 비채택 |
| EXP-036 | count-transition 기반 control proxy 추가 | Ridge `alpha=5000` 및 LightGBM `iter=160`, leaves `7`, min child `3000`; smoothing `300/100/100` | 0.247630668 / 871.18 | +1.26 | 2024만 소폭 개선, 2023 하락으로 비채택 |
| EXP-037 | low-rank source season을 recency policy로 결합 | smoothing `300`, rank `6`, 정책 `equal/last/recency2/...`; 선택 `recency2` | 0.247628091 / 872.21 | +2.29 | hard gate 미달, 단독 후보 비채택 |
| EXP-038 | 저장된 2022~2024 예측 전체의 convex 재가중 상한 감사 | same-fold Frank-Wolfe convex oracle, 최대 300회; 배포 불가 진단 | 0.247577414 / 892.49 | +22.57 | 2023 937.43·2024 892.49로 Skill 1000 불가, 재가중 탐색 중단 |
| EXP-039 | strict/R-specific/aggressive/recency 4 expert의 prior-OOF stack | LightGBM residual, 59 features, `iter=200`, `lr=.015`, leaves `7`, min child `5000`, weight `.25` | 0.247631638 / 870.79 | +0.87 | 2023 Skill 869.88로 하락, 비채택 |
| EXP-040 | recency:aggressive 비율을 70:30으로 고정 | EXP-032 frozen branch convex blend, identity calibration | 0.247620314 / 875.32 | +5.40 | 2024 개선이나 세 시즌 Skill 1000 gate 미달, 비채택 |
| EXP-041 | exact full-game sequence TrackMan ID 정렬 | EXP-033 구성, exact match·mapping purity `.99`, mapping cost `.10` | 0.247636022 / 869.03 | -0.89 | 2023 하락, 비채택 |
| EXP-042 | EXP-041에 source recency2 적용 | EXP-041 + source weights `1:2:4:8` | 0.247647232 / 864.55 | -5.38 | recency가 최신 성능을 더 낮춰 비채택 |
| EXP-043 | exact-aligned pitcher×fine-pitch control EB | smoothing pitcher/type/context/propensity `500/200/100/20`, direct residual weight `.25` | 0.247616774 / 876.74 | +6.82 | 2023·2024는 개선하나 hard gate 미달, 단독 비채택 |
| EXP-044 | EXP-043 exact control과 recentaggr 50:50 결합 | TrackMan/recent `50:50`, direct correction `.25`, identity calibration | **0.247611615 / 878.80** | **+8.88** | Public 1046.949994가 recentaggr보다 0.038999 낮아 비채택 |

### EXP-038 해석

EXP-038은 388개 고유 저장 후보(423개 완전 후보에서 중복 제거)를 현재 fold 정답으로 최적으로 섞는 낙관적 상한 감사다. 2022에서는 Brier `0.243206653836`, Skill `2390.78`로 1000 달성이 가능했지만, 2023은 `0.247656426198`/`937.43`, 2024는 `0.247577413835`/`892.49`였다. 이는 valid selection procedure가 아니며, 같은 저장 예측을 더 많이 재가중해도 2023·2024 목표를 달성할 근거가 없다는 중단 기준으로만 사용한다.

### EXP-044 결과 해석·다음 실험

EXP-044의 exact game alignment는 4,868개 공식 게임 중 2,418개를 exact match하고, 728,342개 행을 정렬했다. 해당 상태에서 587개 pitcher mapping과 fine-pitch control·propensity 표를 동결해 추론 시 평가 행 집계를 하지 않는다. 로컬 2024는 이번 범위 최고지만, 2026-08-12 Public `1046.9499938833`은 EXP-032 recentaggr `1046.9889925352`보다 `0.03899865189987395` 낮았다.

- [ ] EXP-033~042 채택
- [ ] EXP-043 단독 채택
- [ ] EXP-044 채택
- [x] EXP-032 recentaggr 유지

다음 실험은 기존 TrackMan 결합의 weight를 다시 탐색하지 않고, 2023·2024·Public에서 동시에 재현될 수 있는 독립 행별 신호가 확보될 때만 과거 OOF 선택 절차로 검증한다.

---

## 새 실험 템플릿

```markdown
## EXP-000 — 실험 제목

### 기본 정보

- 날짜:
- 작성자:
- 상태: 진행 전 / 진행 중 / 완료
- 기준 실험:
- 관련 코드:
- 저장 모델:
- Git commit:

### 실험 목적

무엇을 확인하려는지 작성한다.

### 가설

왜 이 변경이 점수를 높일 수 있다고 생각했는지 작성한다.

### 기준 실험과의 차이

- 추가한 내용:
- 제거한 내용:
- 유지한 내용:

### 데이터 분할

- 학습 기간:
- 검증 기간:
- 학습 행 수:
- 검증 행 수:
- Target:

### 사용 피처와 전처리

- 추가 피처:
- 제거 피처:
- 범주형 처리:
- 결측값 처리:
- 데이터 누수 점검:

### 모델

- 모델 이름:
- 주요 파라미터:
- Random Seed:
- 확률 보정:

### 실행 환경

- Python:
- pandas:
- numpy:
- scikit-learn:
- 추가 패키지:

### 결과

- 검증 실제 성공률:
- 평균 예측 확률:
- Brier Score:
- 기준 Brier Score:
- Brier Skill Score:
- 학습 시간:
- 검증 추론 시간:
- 모델 크기:

### 기준 실험과 비교

- Brier Score 변화:
- Skill Score 변화:
- 실행 시간 변화:

### 결과 해석

왜 이런 결과가 나왔다고 생각하는지 작성한다.

### 결론

- [ ] 채택
- [ ] 보류
- [ ] 폐기

결정 이유:

### 다음 작업

다음에 확인할 내용을 작성한다.
```
