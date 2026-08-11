# 개발 및 평가 환경

모델 학습과 코드 제출의 재현성을 위해 환경 정보를 기록한다. 모델을 새로 만들거나 패키지를 변경하면 이 문서도 함께 수정한다.

## 평가 서버

대회 안내 기준 환경은 다음과 같다.

| 항목 | 값 |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.11.15 |
| CPU | 6 vCPU |
| RAM | 28GB |
| GPU | NVIDIA L4, VRAM 22.4GiB |
| CUDA | 12.8 |
| 실제 테스트 샘플 | 245,789행 |
| 패키지 설치 제한 | 10분 이하 |
| 추론 실행 제한 | 10분 이하 |
| 인터넷 | 패키지 설치 외 사용 불가 |
| ZIP 크기 | 최대 10GB |
| 압축 해제 후 크기 | 최대 32GB |

### 평가 서버 기본 패키지

| 패키지 | 버전 |
| --- | --- |
| torch | 2.7.1+cu128 |
| pandas | 2.0.3 |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| scikit-learn | 1.8.0 |
| joblib | 1.5.3 |

## 현재 로컬 `.venv`

2026년 8월 10일 확인 결과다.

| 패키지 | 버전 |
| --- | --- |
| Python | 3.12.10 |
| pandas | 3.0.5 |
| numpy | 2.5.1 |
| scikit-learn | 1.9.0 |
| joblib | 1.5.3 |
| catboost | 1.2.8 |
| lightgbm | 4.6.0 |
| xgboost | 3.0.2 |

## 확인된 호환성 사항

- 기존 `baseline_submit/model/rf.pkl`은 scikit-learn 1.8.0에서 생성됐다.
- 2026년 8월 10일 학습 노트북을 실행해 새로 저장한 `model/rf.pkl`은 현재 로컬 scikit-learn 1.9.0에서 생성됐다.
- 현재 로컬 scikit-learn 1.9.0에서 기존 모델을 불러오면 `InconsistentVersionWarning`이 발생한다.
- 기존 `baseline_submit/requirements.txt`에는 `pandas==2.3.3`이 적혀 있지만 평가 서버 기본 버전은 2.0.3이다.
- 새 모델을 현재 로컬 scikit-learn 1.9.0으로 저장한 뒤 평가 서버 1.8.0에서 불러오는 방식은 피한다.
- EXP-013은 scikit-learn pickle을 사용하지 않고 CatBoost `.cbm`과 LightGBM `.txt` 네이티브 형식으로 저장해 Python 3.12 학습 환경과 Python 3.11 평가 환경의 pickle 호환 문제를 피한다.
- EXP-018은 LightGBM `.txt` 네이티브 모델과 JSON 상태만 사용한다. 추론 의존성은 `lightgbm==4.6.0` 하나이며 scikit-learn, scipy, pickle을 사용하지 않는다.
- EXP-018 최종 모델 디렉터리는 약 0.5MB이고 5행 격리 샘플 추론은 로컬에서 약 1초 이내에 완료됐다. 실제 245,789행 시간은 평가 서버에서 확인해야 한다.
- EXP-021은 LightGBM `.txt`, JSON 상태와 HistGradientBoosting 트리의 JSON 내보내기를 사용한다. 원본 HistGradientBoosting과 JSON-tree NumPy 추론은 4,096행에서 최대 절대 오차 `0.0`으로 일치했다.
- EXP-021 제출 패키지는 평가 서버 기본 NumPy·pandas·scikit-learn·joblib을 재설치하지 않고 `lightgbm==4.6.0`만 설치한다. 최종 strict와 aggressive 제출은 평가 서버에서 각각 8초에 실행됐다.
- EXP-021 로컬 생성 환경은 Python `3.12.10`, pandas `3.0.5`, NumPy `2.5.1`, scikit-learn `1.9.0`, joblib `1.5.3`, LightGBM `4.6.0`이다. 모델 직렬화 호환 문제를 피하기 위해 최종 ZIP에는 pickle·joblib 모델을 넣지 않았다.

## 권장 학습 환경

새 모델은 가능하면 평가 서버와 같은 버전으로 학습한다.

```text
Python 3.11
pandas 2.0.3
numpy 1.26.4
scipy 1.15.3
scikit-learn 1.8.0
joblib 1.5.3
```

## 이 Mac에서 제출 호환 환경 만들기

현재 이 Mac에는 Python 3.11이 설치되어 있지 않다. 기존 `.venv`의 패키지를 직접 내리면 다른 작업까지 영향을 받을 수 있으므로, 기존 환경은 유지하고 새 가상환경을 만든다.

### 1. Python 3.11 설치

```bash
brew install python@3.11
```

### 2. 프로젝트 전용 가상환경 생성

Apple Silicon Homebrew 기본 경로를 사용한다.

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 \
  -m venv .venv-py311-sklearn18
```

### 3. 가상환경 활성화

```bash
source .venv-py311-sklearn18/bin/activate
```

터미널 앞에 `(.venv-py311-sklearn18)`이 표시되는지 확인한다.

### 4. 학습에 필요한 버전 설치

```bash
python -m pip install --upgrade pip

python -m pip install \
  pandas==2.0.3 \
  numpy==1.26.4 \
  scipy==1.15.3 \
  scikit-learn==1.8.0 \
  joblib==1.5.3 \
  ipykernel
```

`ipykernel`은 로컬에서 노트북 커널을 선택하기 위한 패키지이며 제출용 `requirements.txt`에는 넣지 않는다.

### 5. 설치 버전 확인

```bash
python -c "import platform, pandas, numpy, sklearn, joblib; print(platform.python_version(), pandas.__version__, numpy.__version__, sklearn.__version__, joblib.__version__)"
```

예상 결과는 다음 버전 조합이다.

```text
Python 3.11.x
pandas 2.0.3
numpy 1.26.4
scikit-learn 1.8.0
joblib 1.5.3
```

### 6. VS Code/Jupyter에서 커널 선택

EXP-002 노트북을 연 뒤 오른쪽 위의 커널 선택 메뉴에서 다음 인터프리터를 선택한다.

```text
.venv-py311-sklearn18/bin/python
```

### 7. EXP-002 검증 실행

터미널에서는 다음처럼 실행할 수 있다.

```bash
.venv-py311-sklearn18/bin/python experiments/train_exp002.py
```

평가 서버 호환 환경에서도 EXP-001보다 점수가 좋으면 전체 모델을 저장한다.

```bash
.venv-py311-sklearn18/bin/python \
  experiments/train_exp002.py --save-final
```

저장 경로는 다음과 같다.

```text
artifacts/EXP-002/rf_exp002.pkl
```

### 기존 환경으로 돌아가기

활성화된 가상환경을 종료할 때는 다음 명령을 사용한다.

```bash
deactivate
```

기존 `.venv`를 삭제하거나 그 안의 scikit-learn 버전을 바꾸지 않는다.

## 환경 변경 기록

| 날짜 | 변경 내용 | 이유 | 확인 결과 |
| --- | --- | --- | --- |
| 2026-08-10 | 로컬과 평가 서버 버전 비교 | 모델 직렬화 호환성 확인 | 별도 Python 3.11 환경 필요 |
| 2026-08-10 | 학습 노트북으로 `model/rf.pkl` 재생성 | 베이스라인 재현 | scikit-learn 1.9.0 모델이므로 그대로 제출하지 않음 |
| 2026-08-10 | EXP-018 LightGBM 네이티브 모델 + JSON 이력 상태 | Python pickle 제거 및 행별 현재 시즌 복원 | 로컬 저장·재로드·격리 샘플 추론 통과 |

## 새 환경을 만들 때 확인할 항목

- [ ] Python 3.11을 사용한다.
- [ ] scikit-learn 1.8.0을 사용한다.
- [ ] 모델 학습 시 실제 패키지 버전을 기록한다.
- [ ] 저장된 모델을 새 프로세스에서 다시 불러온다.
- [ ] 로컬 샘플 `test.csv`로 추론한다.
- [ ] 평가 서버 기본 패키지를 불필요하게 재설치하지 않는다.
- [ ] 추가 패키지 설치가 10분 안에 가능한지 확인한다.
- [ ] 추론 중 외부 다운로드가 발생하지 않는지 확인한다.

## 실험별 환경 기록 템플릿

```text
실험 ID:
실행 날짜:
OS:
Python:
pandas:
numpy:
scikit-learn:
joblib:
추가 패키지:
CPU/GPU:
특이사항:
```
