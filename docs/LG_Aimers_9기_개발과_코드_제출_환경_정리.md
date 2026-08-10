# LG Aimers 9기 프로젝트 : 모델 개발 환경과 코드 제출 방식 정리

> 작성 기준일: 2026년 8월 10일<br>
> 프로젝트 유형: 야구 투구별 제구 성공 확률 예측<br>
> 평가 지표: Brier Skill Score
> 제출 방식: `submit.zip` 코드 제출

## 1. 이 글을 작성한 이유

LG Aimers 9기 프로젝트를 진행하면서 모델 점수를 높이는 것만큼 중요한 것이 개발 환경과 제출 방식을 정확히 관리하는 일이었다. 이번 대회는 예측 결과인 CSV 파일을 직접 제출하는 일반 대회가 아니라, 모델 파일과 추론 코드를 ZIP 파일로 제출하면 평가 서버가 실제 비공개 테스트 데이터에서 코드를 실행하는 방식이다.

따라서 다음 내용을 팀원들과 공유하고, 프로젝트가 끝난 뒤에도 당시 환경을 다시 확인할 수 있도록 기록한다.

- 어떤 데이터와 모델로 시작했는지
- 로컬에서 학습과 검증을 어떻게 진행하는지
- 학습 코드와 제출용 추론 코드를 어떻게 분리하는지
- 평가 서버의 사양과 제한 사항은 무엇인지
- 제출 ZIP 파일을 어떤 구조로 만들어야 하는지
- 제출 전에 무엇을 확인해야 하는지

## 2. 프로젝트 목표와 평가 지표

이번 프로젝트의 목표는 각 투구에 대해 `control_success = 1`일 확률을 예측하는 것이다. 단순히 성공과 실패를 맞히는 분류 정확도 문제가 아니라, 실제 정답에 가까운 확률을 출력해야 하는 확률 예측 문제다.

평가에는 Brier Skill Score가 사용된다.

```text
Brier Score = mean((p_i - y_i)^2)

평균 제구율 Brier Score = r × (1 - r)

Score = max(
    0,
    100000 × (1 - Brier Score / 평균 제구율 Brier Score)
)
```

- `p_i`: 모델이 예측한 제구 성공 확률
- `y_i`: 실제 정답(0 또는 1)
- `r`: 비공개 평가 데이터 전체의 평균 제구 성공률

Brier Score는 낮을수록 좋다. 예를 들어 실제 정답이 1인 샘플에 0.9를 예측한 모델은 0.6을 예측한 모델보다 좋은 평가를 받는다. 반대로 틀린 답을 지나치게 확신해 0.99나 0.01처럼 예측하면 손실이 크게 증가한다.

따라서 이번 프로젝트에서는 다음 요소가 중요하다.

- `predict()` 결과가 아니라 `predict_proba()`의 성공 확률을 제출한다.
- 모델의 분류 성능뿐 아니라 확률 보정 상태도 확인한다.
- 예측값이 반드시 0 이상 1 이하인지 검사한다.
- 검증 데이터의 실제 성공률과 평균 예측 확률이 지나치게 다른지 확인한다.

## 3. 제공 데이터

현재 로컬 프로젝트에는 다음 데이터가 있다.

```text
data/
├── train.csv
├── test.csv
├── sample_submission.csv
└── trackman_history.csv
```

### `train.csv`

- 1,475,092행
- 정답을 포함해 49개 컬럼
- 2019~2024년 데이터
- 정답 컬럼: `control_success`

### `test.csv`

- 로컬 배포본에는 형식 확인용 샘플 5건만 포함
- 실제 평가 서버에서는 245,789개의 비공개 테스트 샘플로 교체
- 정답 컬럼은 포함되지 않음

따라서 로컬의 `test.csv`만으로 실제 리더보드 성능을 추정할 수는 없다. 로컬 테스트 파일은 제출 코드의 입력 형식과 실행 여부를 확인하는 용도로 사용한다.

### `sample_submission.csv`

제출 결과의 컬럼과 `row_id` 순서를 확인하기 위한 파일이다.

```text
row_id,control_success
```

### `trackman_history.csv`

- 2019~2024년 Trackman 과거 로그
- 메인 학습·평가 데이터와 1:1로 직접 연결되는 테이블이 아님
- 구속, 회전수, 무브먼트, 구종 등의 정보 포함

선수 식별자를 확실하게 연결할 수 있는지와 시점 누수 여부를 먼저 검토한 뒤 사용해야 한다. 매핑이 불확실한 상태에서 바로 결합하면 성능 향상보다 노이즈나 규정 위반 위험이 커질 수 있다.

## 4. 시작점인 베이스라인

운영진 베이스라인은 Random Forest 모델을 사용한다.

주요 설정은 다음과 같다.

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=200,
    n_jobs=-1,
    random_state=42,
)
```

`top_bottom`, `game_type`, `base_state`는 범주형으로 변환하며, 수치형 결측값은 중앙값으로 대체한다. 학습 시 2019~2023년을 학습 데이터로, 2024년을 검증 데이터로 사용한다.

현재 베이스라인의 장점은 구조가 단순하고 추론 속도가 빠르며, 제출 파일 용량도 작다는 점이다. 반면 다음과 같은 개선 여지가 있다.

- 투수·타자·팀 ID가 실질적인 범주형 변수인데 숫자형처럼 처리될 수 있음
- Random Forest 설정이 보수적이어서 복잡한 상호작용을 충분히 학습하지 못할 수 있음
- Brier Score를 직접 고려한 확률 보정이 없음
- 카운트, 주자, 점수 차, 경기 중요도 사이의 조합 피처를 더 만들 수 있음

## 5. 학습 코드와 제출 코드 분리

이번 대회에서는 학습 코드와 추론 코드의 역할을 명확히 구분해야 한다.

```text
train_v1.py + train.csv
        │
        │ 로컬에서 학습·검증
        ▼
model/model.pkl
        │
        │ 제출 ZIP에 포함
        ▼
script.py가 비공개 test.csv를 읽어 추론
        │
        ▼
output/submission.csv 생성
```

### 학습 코드

학습 코드는 로컬에서 다음 작업을 담당한다.

- `train.csv` 로드
- 피처 생성 및 전처리
- 시간 기준 학습·검증 분리
- 모델 학습
- Brier Score와 Brier Skill Score 계산
- 최종 모델 재학습
- 모델 파일 저장

학습 코드는 제출 ZIP에 넣지 않는다.

### 추론 코드

제출용 `script.py`는 평가 서버에서 다음 작업만 수행한다.

- `./data/test.csv` 로드
- `./data/sample_submission.csv` 로드
- 저장된 모델 로드
- 학습 때와 동일한 피처 생성
- 성공 확률 예측
- `row_id` 순서에 맞게 결과 구성
- `./output/submission.csv` 저장

평가 서버의 추론 시간 제한이 있으므로 `script.py`에서 모델을 다시 학습하는 방식은 피하는 것이 좋다.

## 6. 권장 프로젝트 구조

베이스라인을 직접 덮어쓰지 않고 실험별로 분리한다.

```text
open/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── trackman_history.csv
│
├── experiments/
│   ├── train_v1.py
│   ├── train_v2.py
│   └── results.csv
│
├── submissions/
│   ├── v1/
│   │   ├── model/
│   │   │   └── model.pkl
│   │   ├── script.py
│   │   └── requirements.txt
│   └── v2/
│       ├── model/
│       ├── script.py
│       └── requirements.txt
│
└── baseline_submit/
```

실험 결과도 함께 기록하면 어떤 변경이 실제로 유효했는지 추적하기 쉽다.

| 버전 | 변경 사항 | 검증 기간 | Brier Score | Skill Score | 제출 점수 |
| --- | --- | --- | ---: | ---: | ---: |
| baseline | Random Forest 기본 설정 | 2024 | 기록 필요 | 기록 필요 | 기준 점수 |
| v1 | 범주형 ID 처리 | 2024 |  |  |  |
| v2 | v1 + 조합 피처 | 2024 |  |  |  |
| v3 | v2 + 확률 보정 | 2024 |  |  |  |

## 7. 검증 전략

이번 데이터는 시간에 따른 정답 비율 변화가 크다.

| 시즌 | 제구 성공률 |
| --- | ---: |
| 2019 | 0.5647 |
| 2020 | 0.5327 |
| 2021 | 0.5328 |
| 2022 | 0.5289 |
| 2023 | 0.5000 |
| 2024 | 0.4861 |

랜덤 분할을 사용하면 과거와 미래 데이터가 섞여 실제 2025년 평가 상황보다 점수가 낙관적으로 나올 수 있다. 기본 검증은 다음과 같이 시간 기준으로 진행한다.

```python
is_valid = train["season"] == 2024

X_train = train.loc[~is_valid, FEATURES]
y_train = train.loc[~is_valid, TARGET]

X_valid = train.loc[is_valid, FEATURES]
y_valid = train.loc[is_valid, TARGET]
```

모델 비교 시에는 모든 실험에서 같은 2024년 검증 데이터를 사용해야 한다. 그래야 피처나 모델 변경 효과를 공정하게 비교할 수 있다.

보다 안정적으로 평가하려면 다음 시간 분할도 추가할 수 있다.

- 2019~2022년 학습 → 2023년 검증
- 2019~2023년 학습 → 2024년 검증

두 기간 모두 개선되는 변경을 우선 채택한다. 2024년 한 기간에만 과도하게 맞는 실험은 비공개 2025년 데이터에서 성능이 떨어질 수 있다.

## 8. 첫 번째 성능 개선 실험

첫 실험에서는 많은 변경을 한꺼번에 적용하지 않는다. 한 번에 하나씩 변경해야 어떤 요소가 점수에 영향을 줬는지 확인할 수 있다.

### 1단계: 범주형 변수 처리 개선

기존 범주형 컬럼 외에 다음 ID와 코드도 범주형 후보로 본다.

```python
CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]
```

ID 값의 숫자 크기는 선수의 실력이나 순서를 의미하지 않는다. 따라서 ID를 단순한 연속 수치로 처리하는 것보다 범주로 처리할 수 있는 모델을 비교할 가치가 있다.

### 2단계: 상황 조합 피처

다음과 같은 조합을 검토한다.

```python
df["count_code"] = df["balls_before"] * 4 + df["strikes_before"]

df["same_hand"] = (
    df["pitcher_hand"] == df["batter_hand"]
).astype("int8")

df["rate_matchup_gap"] = (
    df["asof_pitcher_success_rate"]
    - df["asof_batter_success_rate"]
)

df["score_pressure"] = (
    df["li"] * df["score_diff_pitcher_team"].abs()
)
```

피처는 반드시 현재 투구 직전에 알 수 있는 정보만 사용해야 한다.

### 3단계: 확률 보정

Brier Score에서는 확률 보정이 직접적으로 중요하다. 기본 모델, sigmoid 보정, isotonic 보정을 동일한 검증 데이터에서 비교한다.

다만 검증 데이터로 보정기를 학습한 뒤 같은 데이터에서 성능을 보고하면 과적합된 결과가 된다. 모델 학습용, 보정용, 최종 평가용 기간을 분리하거나 시간 순서 기반 out-of-fold 예측을 사용하는 것이 안전하다.

## 9. 테스트 데이터 사용 시 금지 사항

평가 데이터의 각 행은 독립적으로 예측해야 한다. 평가 서버에서 전체 `test.csv`를 읽을 수 있더라도 다른 테스트 행을 이용해 현재 행의 피처를 만들면 안 된다.

금지되는 예시는 다음과 같다.

- 테스트 데이터 전체의 평균, 빈도 또는 분포를 이용한 피처
- 테스트 데이터 내부 선수별·팀별 집계
- 테스트 데이터에서 만든 target encoding
- 테스트 행 순서를 이용한 rolling 또는 expanding 피처
- 현재 투구 이후에 확정되는 결과 정보 사용
- 2025년 Trackman 데이터 사용
- 외부 서버에서 데이터나 모델 다운로드

운영진이 제공한 `asof_*` 피처는 현재 투구 직전까지의 이력으로 계산된 공식 피처이므로 사용할 수 있다.

부정행위 탐지 시스템이 모든 제출물을 모니터링하며 규정 위반에 연관된 팀 전체가 탈락할 수 있으므로, 피처의 생성 시점과 데이터 출처를 코드와 실험 기록에 남겨야 한다.

## 10. 평가 서버 환경

평가 서버의 주요 사양과 제한은 다음과 같다.

| 항목 | 환경 |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.11.15 |
| CPU | 6 vCPU |
| RAM | 28GB |
| GPU | NVIDIA L4, VRAM 22.4GiB |
| CUDA | 12.8 |
| 실제 테스트 샘플 | 245,789개 |
| 패키지 설치 제한 | 10분 이하 |
| 추론 실행 제한 | 10분 이하 |
| 인터넷 | 패키지 설치 외 사용 불가 |
| ZIP 크기 | 10GB 이하 |
| 압축 해제 후 크기 | 32GB 이하 |

주요 기본 패키지는 다음과 같다.

```text
torch==2.7.1+cu128
pandas==2.0.3
numpy==1.26.4
scipy==1.15.3
scikit-learn==1.8.0
joblib==1.5.3
```

모델 직렬화 파일은 라이브러리 버전에 영향을 받을 수 있다. 특히 scikit-learn 모델은 학습 환경과 추론 환경의 버전을 일치시키는 것이 안전하다.

현재 확인한 로컬 `.venv` 환경은 다음과 같았다.

```text
Python 3.12.10
scikit-learn 1.9.0
pandas 3.0.5
numpy 2.5.1
joblib 1.5.3
```

평가 서버와 버전 차이가 있으므로 새 모델은 평가 서버에 맞춘 별도 Python 3.11 환경에서 학습하는 것이 좋다.

권장 환경은 다음과 같다.

```text
Python 3.11
scikit-learn 1.8.0
pandas 2.0.3
numpy 1.26.4
joblib 1.5.3
```

## 11. `requirements.txt` 관리

평가 서버에 이미 설치된 패키지는 가급적 다시 설치하지 않는 것이 좋다. 다른 버전을 강제로 설치하면 설치 시간이 증가하거나 의존성 충돌이 발생할 수 있다.

추가 라이브러리를 사용하지 않는다면 `requirements.txt` 파일은 유지하되 기본 패키지를 불필요하게 재설치하지 않도록 구성한다. 추가 패키지를 사용한다면 다음을 확인한다.

- Python 3.11과 호환되는가
- Ubuntu 22.04에서 설치 가능한가
- CUDA 12.8 환경과 호환되는가
- 설치가 10분 안에 끝나는가
- 설치 이후 인터넷 다운로드가 필요한 패키지는 아닌가
- 학습에만 필요하고 추론에는 필요 없는 패키지를 제외했는가

현재 베이스라인 `requirements.txt`에는 `pandas==2.3.3`이 명시되어 있지만 평가 서버 기본 버전은 `2.0.3`이다. 특별한 이유가 없다면 서버 기본 버전을 활용하는 방향으로 정리할 예정이다.

## 12. 제출 ZIP 만들기

제출 파일은 ZIP 하나만 업로드할 수 있으며 내부 구조가 정확해야 한다.

```text
submit_v1.zip
├── model/
│   └── model.pkl
├── script.py
└── requirements.txt
```

`submissions/v1` 폴더 안에서 다음 명령으로 압축할 수 있다.

```bash
cd submissions/v1

zip -r ../../submit_v1.zip \
  model \
  script.py \
  requirements.txt \
  -x "*.DS_Store" "__MACOSX/*"
```

압축 후에는 내부 구조를 확인한다.

```bash
unzip -l ../../submit_v1.zip
```

다음처럼 추가 최상위 폴더가 포함되면 안 된다.

```text
submit_v1.zip
└── submissions/
    └── v1/
        ├── model/
        ├── script.py
        └── requirements.txt
```

macOS에서 생성될 수 있는 `.DS_Store`와 `__MACOSX`도 제외한다.

## 13. 제출 전 로컬 점검

제출 오류는 일일 제출 횟수에 반영될 수 있으므로 로컬 샘플 데이터로 먼저 실행한다.

```bash
cd submissions/v1
python script.py
```

실행 후 다음 파일이 생성되어야 한다.

```text
output/submission.csv
```

다음 항목을 자동 검사하는 것이 좋다.

```python
import pandas as pd

test = pd.read_csv("data/test.csv")
submission = pd.read_csv("output/submission.csv")

assert list(submission.columns) == ["row_id", "control_success"]
assert len(submission) == len(test)
assert submission["row_id"].tolist() == test["row_id"].tolist()
assert submission["row_id"].is_unique
assert submission["control_success"].notna().all()
assert submission["control_success"].between(0, 1).all()
```

실제 제출 전 최종 체크리스트는 다음과 같다.

- [ ] 2024년 시간 분할 검증 점수를 기록했는가?
- [ ] 최종 모델을 2019~2024년 전체 데이터로 다시 학습했는가?
- [ ] 학습과 추론의 피처 생성 코드가 동일한가?
- [ ] 모델을 평가 서버와 호환되는 버전으로 저장했는가?
- [ ] `script.py`가 외부 인터넷에 접속하지 않는가?
- [ ] `script.py`에서 재학습하지 않는가?
- [ ] 테스트 데이터의 다른 행을 이용한 집계가 없는가?
- [ ] `./data/test.csv`와 `sample_submission.csv`를 정상적으로 읽는가?
- [ ] 결과가 `./output/submission.csv`에 저장되는가?
- [ ] 예측값에 결측치나 무한대가 없는가?
- [ ] 모든 예측 확률이 0~1 범위인가?
- [ ] 샘플 데이터에서 처음부터 끝까지 실행되는가?
- [ ] 추론 시간이 10분 이내인가?
- [ ] ZIP 최상위 구조가 정확한가?
- [ ] ZIP에 데이터, 노트북, 불필요한 캐시 파일이 없는가?

## 14. 오류 유형 이해하기

운영 안내에서는 오류를 크게 두 가지로 구분한다.

### 설치 오류

- ZIP 내부 구조 불일치
- `requirements.txt` 설치 실패

설치 오류는 일일 제출 횟수에 반영되지 않는다.

### 제출 오류

- `script.py` 실행 실패
- 모델 로드 실패
- 입력 파일 경로 오류
- 추론 중 메모리 또는 시간 초과
- `submission.csv` 미생성
- 출력 컬럼이나 행 개수 불일치

`script.py` 실행 이후 발생하는 오류는 제출 횟수에 반영되므로 로컬 재현 검사가 중요하다.

## 15. 팀 협업 규칙

팀원 간 실험 결과를 비교하기 위해 다음 항목을 함께 기록한다.

```text
실험 버전:
작성자:
학습 데이터 기간:
검증 데이터 기간:
사용 피처:
모델 및 주요 파라미터:
확률 보정 방식:
로컬 Brier Score:
로컬 Skill Score:
학습 시간:
추론 시간:
모델 파일 크기:
리더보드 점수:
비고:
```

한 번에 여러 변경을 적용하면 점수 변화의 원인을 찾기 어렵다. 가능하면 다음처럼 단계적으로 실험한다.

```text
baseline
  → 범주형 처리 변경
  → 조합 피처 추가
  → 모델 파라미터 변경
  → 확률 보정
  → 앙상블
```

각 단계의 모델과 제출 ZIP을 별도 버전으로 보관하면 리더보드 점수가 떨어졌을 때 이전 버전으로 쉽게 돌아갈 수 있다.

## 16. 현재 결론과 다음 작업

현재 가장 먼저 진행할 작업은 다음과 같다.

1. 운영진 베이스라인은 그대로 보존한다.
2. `train_v1.py`와 `submissions/v1/`을 새로 만든다.
3. 평가 서버와 동일한 Python 및 라이브러리 환경을 준비한다.
4. 2024년 검증 점수를 정확히 재현한다.
5. 투수·타자·팀 ID의 범주형 처리 개선을 첫 실험으로 진행한다.
6. 조합 피처와 확률 보정은 별도 실험으로 추가한다.
7. 로컬 검증과 제출 ZIP 테스트를 통과한 모델만 리더보드에 제출한다.

이번 대회에서 중요한 것은 가장 복잡한 모델을 만드는 것보다, 시간 순서에 맞게 검증하고 확률을 잘 보정하며 동일한 전처리를 제한 시간 안에 안정적으로 재현하는 것이다. 모델 점수와 함께 실행 환경, 피처 생성 시점, 제출 파일 구조를 꾸준히 기록하는 것이 최종 코드 검증까지 통과하는 데 도움이 될 것이다.

## 참고 링크

- [대회 평가 안내](https://dacon.io/competitions/official/236743/overview/evaluation)
- [데이콘 코드 제출 가이드](https://cfiles.dacon.co.kr/competitions/236564/guide.html)

> 대회 규칙과 평가 환경은 변경될 수 있으므로 실제 제출 전 데이콘 공지와 평가 탭을 다시 확인한다.
