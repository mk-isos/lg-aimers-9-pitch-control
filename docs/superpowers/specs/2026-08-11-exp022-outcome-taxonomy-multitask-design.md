# EXP-022 Outcome-Taxonomy Temporal Multi-Task Design

## 1. 목적과 성공 기준

EXP-022의 목적은 EXP-021 strict가 남긴 2023·2024 성능 병목을 기존 예측의 재가중이 아닌 새로운 행별 감독 신호로 줄이는 것이다.

최종 채택 기준은 다음과 같다.

- rolling-origin 검증 시즌 2022·2023·2024의 Skill Score가 각각 `1100` 이상이어야 한다.
- 세 시즌 모두 EXP-021 strict보다 나빠지지 않아야 한다.
- 현재 검증 시즌의 정답으로 모델, 후보 weight, 확률 보정 또는 threshold를 선택하지 않는다.
- 테스트 데이터의 다른 행을 이용한 집계, 빈도, 분포, rolling, target encoding을 사용하지 않는다.
- 이 기준을 통과하지 못하면 전체 학습과 제출 ZIP 생성을 진행하지 않는다.

EXP-021 strict의 기계 기록 기준은 다음과 같다.

| 시즌 | Brier Score | Skill Score |
| ---: | ---: | ---: |
| 2022 | 0.24470458400867237 | 1789.5967932082258 |
| 2023 | 0.2477311440988816 | 907.5416355312283 |
| 2024 | 0.24763380341629648 | 869.9211702032806 |

2023·2024의 Skill `1100` Brier 기준은 각각 `0.24724999819122953`, `0.2470590505624633`이다.

## 2. 왜 새 신호가 필요한가

EXP-020 final ensemble ceiling audit의 13개 frozen 후보를 같은 fold 정답으로 최적 재가중한 비배포 convex oracle은 다음에 그쳤다.

| 시즌 | oracle Brier | oracle Skill | 1100 Brier 기준 | 판정 |
| ---: | ---: | ---: | ---: | --- |
| 2023 | 0.2476591855946419 | 936.3250374376797 | 0.24724999819122953 | 기존 convex hull로 1100 불가 |
| 2024 | 0.2476071571884563 | 880.587899179186 | 0.2470590505624633 | 기존 convex hull로 1100 불가 |

두 시즌의 Frank-Wolfe gap은 `0.0`이다. 따라서 기존 EXP-019~021 예측을 다시 섞거나 scalar calibration만 바꾸는 실험은 EXP-022 범위에서 제외한다.

## 3. 핵심 가설

`train.csv`의 현재 행에는 투수의 투구 직전 누적 `asof_pitcher_*_rate`와 `asof_pitcher_n`이 있다. 같은 투수·시즌에서 표본 수가 정확히 1 큰 상태를 찾으면 누적 count의 차이로 현재 투구의 보조 결과를 복원할 수 있다. 파일 행 순서나 `row_id` 정렬은 시간 순서로 가정하지 않는다.

이 보조 결과는 `control_success`와 다른 투구 결과의 구조를 학습시켜, 성공 여부만 직접 학습한 residual 모델이 놓친 행별 제구 상태를 표현할 수 있다는 것이 가설이다.

보조 label 후보는 다음 네 개다.

- `aux_reverse`
- `aux_middle`
- `aux_ball`
- `aux_strike`

`reverse`와 `middle`은 동시에 1일 수 있으므로 multinomial partition으로 취급하지 않고 독립 multi-label binary target으로 다룬다. `ball`과 `strike`는 동시 발생하지 않는 binary target으로 다루되, 별도의 강제 합 제약은 두지 않는다.

`success` 증분은 `control_success`와 일치하는지 확인하는 QA에만 쓰고 보조 모델의 입력이나 별도 target으로 사용하지 않는다.

## 4. 보조 label 복원 규칙

### 4.1 누적 count 복원

투수 표본 수를 `n`, 누적 rate를 `r`이라 할 때 누적 count는 다음처럼 복원한다.

```text
count = round(n × r)
```

현재 행 `i`마다 다음 key를 가진 유일한 후속 상태 `j`를 hash join으로 찾는다.

```text
pitcher_id[i] == pitcher_id[j]
season[i] == season[j]
asof_pitcher_n[j] == asof_pitcher_n[i] + 1
```

join key는 `(pitcher_id, season, asof_pitcher_n)`이며 source 쪽 key가 중복되면 해당 pair를 사용하지 않고 중복 수를 QA에 기록한다. 따라서 원본 행 순열에 관계없이 같은 label이 만들어져야 한다. 각 누적 count의 `count[j] - count[i]`가 0 또는 1일 때만 유효하다. 다른 값, 결측, 비단조 표본 수는 해당 보조 label 학습에서 제외한다.

### 4.2 시간 누수 방지

검증 시즌을 `v`라 할 때 보조 모델 학습에는 두 행 모두 `season < v`인 pair만 사용한다. 검증 시즌 첫 행을 사용해 이전 시즌 마지막 행의 label을 만들지 않으며, 시즌 경계를 넘는 pair는 전부 제외한다.

2025 전체 학습에서도 2019~2024 train 내부의 같은 시즌 pair만 사용한다. `test.csv`의 순서나 다른 행은 label 복원, 피처 생성, 빈도 계산에 사용하지 않는다.

### 4.3 QA 불변식

- 복원된 success 증분과 `control_success`의 mismatch가 0이어야 한다.
- 각 보조 label은 `{0, 1}`이어야 한다.
- `(pitcher_id, season, asof_pitcher_n)` source key 중복 수와 제외 수를 기록한다.
- source season과 validation season이 겹치지 않아야 한다.
- 원본 train 행 순열 전후에 복원된 label이 `row_id` 기준으로 동일해야 한다.
- 행 순열 또는 singleton 추론에서 같은 `row_id`의 보조 예측이 bitwise 동일해야 한다.

## 5. 보조 모델

### 5.1 입력 피처

EXP-019 HistGradientBoosting residual에서 검증한 row-independent stable allow-list를 재사용한다.

- raw `pitcher_id`, `batter_id`, team ID, `season`, `row_id` 제외
- 현재 행의 공식 상황 피처와 공식 `asof_*` 값만 사용
- 학습 이력으로 만든 temporal feature는 validation season 이전에 frozen된 상태만 사용
- `control_success`와 복원한 보조 label은 입력에서 제외
- `top_bottom_*`, `base_state_*`는 학습 schema에 고정한 one-hot만 사용

### 5.2 고정 모델 사양

보조 target마다 독립 `HistGradientBoostingClassifier` 하나를 학습한다.

```text
learning_rate=0.025
max_iter=160
max_leaf_nodes=15
max_depth=4
min_samples_leaf=3000
l2_regularization=30.0
max_bins=127
max_features=0.70
early_stopping=False
random_state=42
```

source season의 총 sample weight가 같아지도록 season-equal weighting을 적용한다. 현재 fold 결과를 보고 모델 용량을 바꾸지 않는다.

### 5.3 temporal OOF 생성

2021을 warm-up으로 두고 2022·2023·2024를 보고 fold로 사용한다.

각 season `s`의 보조 OOF 예측은 `season < s`의 유효 pair로 학습한 모델에서 생성한다. 보조 OOF 예측은 다음 six-column representation으로 저장한다.

```text
p_reverse
p_middle
p_ball
p_strike
p_strike - p_ball
p_reverse + p_middle
```

마지막 항목은 확률 partition이 아니라 중첩 가능한 failure-pressure 표현이다.

## 6. EXP-021 strict와 결합

EXP-021 strict OOF 예측을 immutable base로 사용한다. 새 모델은 base 확률 자체를 다시 학습하지 않고 보조 OOF representation으로 `y - base` residual만 예측한다.

deployable residual combiner는 다음 규칙을 따른다.

- source OOF season의 residual 평균을 source season 안에서 제거한다.
- 보조 representation의 scaling 통계는 source training season에서만 계산해 저장한다.
- `Ridge(alpha=5000, fit_intercept=False)`를 사용한다.
- correction scale 후보는 사전 고정한 `0.25`, `0.50` 두 개만 둔다.
- 2022 후보 선택은 2021 warm-up, 2023은 2021~2022, 2024는 2021~2023만 사용한다.
- 선택 목적은 prior-fold 최저 Skill, 평균 Skill, 더 작은 scale 순서의 lexicographic rule이다.
- affine, isotonic, sigmoid 또는 고정 offset 보정을 추가하지 않는다.

최종 확률은 다음과 같다.

```text
p = clip(EXP021_strict_base + selected_scale × aux_residual, 0, 1)
```

## 7. 진단 ceiling과 중단 규칙

temporal 후보와 별도로 현재 fold label을 허용한 진단용 representation ceiling을 계산한다. 이 결과는 배포 후보 선택에 사용하지 않는다.

- same-fold ridge residual fit
- 5-fold cross-fitted ridge residual fit, fixed seed 42
- deployable temporal prior-fold ridge

2023 또는 2024에서 same-fold ridge조차 Skill `1100`에 도달하지 못하면 이 linear multi-task 결합 family를 중단한다. same-fold은 통과하지만 cross-fitted 또는 temporal 후보가 실패하면 보조 신호는 존재하지만 시간 전이가 부족한 것으로 기록하고 ZIP을 만들지 않는다.

## 8. 기록할 지표

각 후보와 기준값에 대해 JSON에서 자동 생성한다.

- 시즌별 Brier Score와 Skill Score
- 평균 Skill과 최저 Skill
- 예측 평균, 실제 성공률, mean gap
- calibration slope와 intercept
- `game_type` R/F 성능
- 투수 현재 시즌 표본 구간 `0`, `1~19`, `20~99`, `100~499`, `500+`
- 투수·타자 신규/기존 조합
- 월별 성능과 최저 월
- 보조 label별 유효 pair 수, 양성률, season coverage
- 보조 model의 source season과 validation season
- success 증분 QA mismatch
- singleton/batch/행 순열 parity

## 9. 채택과 패키징

다음 조건을 모두 만족할 때만 EXP-022를 채택한다.

```text
Skill_2022 >= 1100
Skill_2023 >= 1100
Skill_2024 >= 1100
모든 시즌 Skill >= EXP021 strict
현재 fold label로 후보 선택/보정하지 않음
테스트 행 간 집계 없음
```

통과하면 2019~2024 source 상태와 보조 모델을 frozen serialization하고 최종 제출 ZIP 하나를 만든다. 통과하지 않으면 `validation_metrics.json`과 실패 원인만 기록하고, 기존 EXP-021 strict를 리더보드 선택으로 유지한다.

## 10. 예상 위험

- 누적 rate는 반올림된 값이므로 일부 pair에서 count 복원이 불가능할 수 있다. 불확실 pair를 억지로 label하지 않는다.
- reverse와 middle의 중첩을 잘못 multinomial로 다루면 정보가 손실된다.
- 보조 target 예측이 같은 공식 rate 피처를 재표현하는 데 그치면 새 residual 신호가 작을 수 있다.
- same-fold ceiling이 높아도 2023 구조 변화에 시간 전이되지 않을 수 있다.
- 대회 규칙이 train 내부 파생 보조 label까지 제한한다고 명시돼 있다면 구현 전에 이 branch를 폐기한다. 현재 설계는 train.csv를 이용한 학습 label 파생은 허용된다는 해석을 전제로 한다.

## 11. 산출물 계획

- `experiments/train_exp022_outcome_taxonomy_multitask.py`
- `artifacts/EXP-022/outcome_taxonomy_multitask/validation_metrics.json`
- Git ignore 대상 OOF auxiliary prediction/target arrays
- gate 통과 시에만 final builder, inference script, ZIP
