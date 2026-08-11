# 코드 제출 기록

실제로 생성하거나 제출한 ZIP과 결과를 기록한다. 운영진이 제공한 베이스라인 패키지와 개인 제출물을 구분한다.

## 제출 규칙 요약

제출 ZIP 최상위 구조는 다음과 같아야 한다.

```text
submit.zip
├── model/
│   └── 모델 파일
├── script.py
└── requirements.txt
```

평가 서버가 추가하는 항목은 다음과 같다.

```text
data/
output/
```

`script.py`는 `./data/test.csv`를 읽고 반드시 `./output/submission.csv`를 생성해야 한다.

## 제출 요약

| 제출 ID | 날짜 | 연결 실험 | ZIP | 상태 | Public Score |
| --- | --- | --- | --- | --- | ---: |
| REF-BASELINE | 2026-08-10 확인 | EXP-000 | `baseline_submit.zip` | 운영진 참고 패키지 | 개인 제출 여부 미기록 |
| SUB-001 | 예정 | EXP-001 | 생성 전 | 제출 보류: 모델 버전 확인 필요 | - |
| SUB-002 | 예정 | EXP-002 | 생성 전 | 2024년 검증 개선, 호환 모델 준비 전 | - |
| SUB-013 | 2026-08-10 | EXP-013 | `submit_exp013.zip` | 제출 완료, 현재 리더보드 선택 | 935.810810 |
| SUB-015 | 2026-08-10 | EXP-015 | `submit_exp015_best.zip` | 제출 완료, EXP-013보다 하락 | 927.712979 |
| SUB-016 | 2026-08-10 | EXP-016 | `submit_exp016_no_season_adjustment.zip` | 진단 후보, 제출 보류 | - |
| SUB-018 | 2026-08-10 | EXP-018 | `exp018_multiscale.zip` | 제출 완료, EXP-013보다 하락해 비채택 | 895.836877 |
| SUB-021-STRICT | 2026-08-11 | EXP-021 | `submit_exp021_strict.zip` | 제출 완료, 현재 최종 리더보드 선택 | **1043.607420** |
| SUB-021-AGGR | 2026-08-11 | EXP-021-AGGR | `submit_exp021_aggr.zip` | 제출 완료, strict보다 낮아 비채택 | 1043.187131 |

## REF-BASELINE — 운영진 제공 베이스라인 패키지

이 항목은 개인 제출 결과가 아니라 현재 작업 폴더에 제공된 참고 ZIP을 확인한 기록이다.

### ZIP 내부 구조

```text
model/
model/rf.pkl
requirements.txt
script.py
```

### 파일 크기

- ZIP: 약 3.8MB
- 모델: 약 3.8MB

### 확인 사항

- 모델은 scikit-learn 1.8.0에서 생성됐다.
- `script.py`는 `./data/test.csv`와 `sample_submission.csv`를 읽는다.
- 결과를 `./output/submission.csv`에 저장한다.
- 기존 `requirements.txt`의 pandas 버전은 평가 서버 기본 버전과 다르다.

### 상태

- 개인 제출 여부: 기록 없음
- 리더보드 점수: 기록 없음
- 목적: 제출 구조 참고 및 베이스라인 보존

## SUB-001 — 첫 개인 제출

### 기본 정보

- 제출 날짜: 예정
- 연결 실험: EXP-001
- ZIP 파일: 생성 예정
- 모델 파일: 생성 예정
- 제출 상태: 제출 보류

### 로컬 검증

- 검증 기간: 2024년
- Brier Score: 0.248767
- Brier Skill Score: 416.18
- 로컬 샘플 추론 시간: 측정 전
- 모델 크기: 3,957,793바이트(약 3.8MiB)

### 제출 전 검사

- [x] `script.py` 로컬 실행 성공
- [x] `output/submission.csv` 생성
- [x] 제출 컬럼이 `row_id`, `control_success` 순서
- [x] 테스트 데이터와 결과 행 개수 일치
- [x] `row_id` 순서 일치
- [x] 중복 `row_id` 없음
- [x] 결측 예측값 없음
- [x] 모든 예측값이 0~1 범위
- [x] 학습과 추론 피처 생성 방식 일치
- [x] 외부 인터넷 접속 및 다운로드 없음
- [x] 테스트 데이터 내부 집계 없음
- [ ] 전체 추론 시간 10분 이내 예상
- [ ] ZIP 최상위 구조 확인
- [ ] `.DS_Store`, `__MACOSX` 제외
- [ ] 데이터와 학습 노트북 제외

### 제출 결과

- 설치 상태: 제출 후 작성
- 실행 상태: 제출 후 작성
- Public Score: 제출 후 작성
- 순위: 제출 후 작성
- 평가 서버 실행 시간: 확인 가능 시 작성

### 현재 제출을 보류한 이유

새 `model/rf.pkl`은 로컬 scikit-learn 1.9.0에서 생성됐다. 평가 서버 기본 버전은 1.8.0이므로 모델 직렬화 호환성을 보장할 수 없다. Python 3.11과 scikit-learn 1.8.0 환경에서 다시 학습한 모델로 ZIP을 만들어야 한다.

### 결과 해석

제출 후 로컬 검증 결과와 리더보드 결과의 차이를 작성한다.

## SUB-013 — CatBoost + LightGBM 보정 앙상블

### 기본 정보

- 준비 날짜: 2026-08-10
- 연결 실험: EXP-013
- ZIP 파일: `submit_exp013.zip`
- 모델: CatBoost `.cbm` + LightGBM `.txt`
- 제출 상태: 제출 완료, 현재 리더보드 선택

### 로컬 검증

- 검증 기간: 2024년
- Brier Score: 0.247862497
- Skill Score: 778.37
- 지표 출처: `artifacts/EXP-013/2024/validation_metrics.json`
- ZIP 크기: 약 17MB
- ZIP SHA-256: `791aa743b3d70b476a67c34d4cdda396b3271a25a508303fde6d6fe4f399b0d0`
- 압축 전 모델 및 코드: 약 47.2MB
- 평가 규모와 비슷한 253,507행 모델 파이프라인 시간: 약 1.4초

### 제출 전 검사

- [x] `script.py` 문법 검사 통과
- [x] 샘플 5행 추론 성공
- [x] `output/submission.csv` 생성
- [x] 행 개수와 `row_id` 순서 일치
- [x] 중복 `row_id` 없음
- [x] 예측값 결측 없음
- [x] 예측값 0~1 범위
- [x] ZIP 최상위에 `model/`, `script.py`, `requirements.txt` 배치
- [x] 데이터와 학습 노트북 제외
- [x] `.DS_Store`, `__MACOSX` 제외
- [x] 추론 중 외부 인터넷 사용 없음
- [x] 테스트 데이터 내부 집계 없음
- [x] 10분 추론 제한 대비 충분한 여유 확인

### 패키지

```text
catboost==1.2.8
lightgbm==4.6.0
```

두 패키지는 Python 3.11 Linux wheel 제공 여부를 확인했다. 실제 평가 서버의 설치 성공 여부는 제출 후 확인한다.

### 제출 결과

- 설치 상태: 성공
- 실행 상태: 성공
- Public Score: `935.8108097065`
- 현재 상태: EXP-015보다 높아 리더보드 선택 유지

## SUB-015~016 — 변경 원인 분리 미완료

- EXP-015 Public Score: `927.7129792368`
- 엔지니어드 LightGBM과 2025 고정 `-0.005`를 동시에 변경해 하락 원인을 완전히 분리하지 못했다.
- EXP-016은 고정 보정만 제거했지만 1000점 이상 기대가 낮아 제출하지 않았다.

## SUB-018 — constrained multiscale 제출 결과

### 기본 정보

- 연결 실험: EXP-018
- 실제 제출 ZIP: `exp018_multiscale.zip`
- 로컬 정리 후 파일명: `submit_exp018_multiscale.zip`
- ZIP 크기: `141,811`바이트
- SHA-256: `04ea8ef4e18eb23c01d87b0e13769f76086f7c17e714ad9a399c97466dadbd3a`
- 모델: LightGBM native text + JSON 이력 상태·그룹 효과
- 확률 보정: identity

### rolling 검증

| 시즌 | Brier | Skill |
| ---: | ---: | ---: |
| 2022 | 0.244537937 | 1856.48 |
| 2023 | 0.248075365 | 769.85 |
| 2024 | 0.247820261 | 795.28 |

- 평균 Skill: `1140.54`
- 최저 Skill: `769.85`
- EXP-013 raw rolling 평균/최저: `582.06` / `-1303.57`

### 최종 검사

- [x] ZIP CRC 통과
- [x] 최상위 `script.py`, `requirements.txt`, `model/`만 포함
- [x] CSV, NPY, pickle, 로그, 데이터 파일 미포함
- [x] 별도 임시 디렉터리 압축 해제 후 샘플 추론 성공
- [x] `output/submission.csv` 생성
- [x] 5행, row_id 순서, 중복, 결측, 0~1 범위 통과
- [x] 샘플 추론 `1.058`초
- [x] 학습용·제출용 피처 113열 parity 통과

### 제출 메모

`공식 asof 누적값과 2019~2024 종료 상태의 차이로 현재 시즌 투수·타자 기록을 행별 복원하고, 계층적 shrinkage 기준값에 과거 3시즌 count/손 조합 효과와 직전 1시즌 residual LightGBM 15%를 결합했습니다. 테스트 행 간 집계와 고정 2025 offset은 사용하지 않았습니다. 2022/2023/2024 rolling Skill은 1856.48/769.85/795.28, 평균 1140.54입니다.`

### Public 결과

- 제출 일시: `2026-08-10 16:39:00`
- Public Score: `895.8368767677`
- EXP-013 Public 대비: `-39.9739329388`
- EXP-015 Public `927.7129792368`보다도 낮음
- 채택 여부: 비채택
- 리더보드 선택: EXP-013 유지
- 해석: 2022 고점이 rolling 평균을 끌어올렸고, 현재 시즌 복원·계층적 기준값이 2025의 확률 수준과 초기 표본 구조에 충분히 일반화되지 못했다.

---

## SUB-021 — strict·aggressive 최종 제출 결과

### 기본 정보

| 항목 | strict | aggressive |
| --- | --- | --- |
| 연결 실험 | EXP-021 strict rank-6 | EXP-021 aggressive R/F gate |
| 실제 제출 ZIP | `submit_exp021_strict.zip` | `submit_exp021_aggr.zip` |
| ZIP 크기 | `1,942,657`바이트 | `1,942,498`바이트 |
| SHA-256 | `e4b1cd4868551df0ec9886bd5dae9c6e3f9707029c9cad31ebd9bba5bb7a8be5` | `68fb18791010794cc4670403c56e24848629ea74cfff02d803010cb01c583191` |
| 제출 기록 ID | `42698` | `42700` |
| 제출 일시 | `2026-08-11 08:58:48` | `2026-08-11 09:03:41` |
| 실행 시간 | `8초` | `8초` |

### 제출 과정에서 수정한 호환성 문제

- 첫 패키지는 로컬 학습 환경의 `numpy==2.5.1` 등을 강제 설치해 평가 서버 Python 3.11에서 설치에 실패했다.
- 평가 이미지에 있는 NumPy·pandas·scikit-learn·joblib은 재설치하지 않고 `lightgbm==4.6.0`만 설치하도록 수정했다.
- 다음 실행에서는 NumPy 2.5 환경에서 저장한 HistGradientBoosting Joblib을 서버 NumPy 1.26이 역직렬화하지 못했다.
- 최종 패키지는 HGB 160개 수치형 트리를 JSON으로 내보내 NumPy로 직접 추론하며, 원본 모델과 최대 절대 오차 `0.0` parity를 확인했다.

### 최종 Public 결과

| 후보 | Public Score | EXP-013 대비 | strict 대비 | 채택 |
| --- | ---: | ---: | ---: | --- |
| strict rank-6 | **1043.6074197937** | **+107.7966100872** | - | **최종 선택** |
| aggressive R/F gate | 1043.1871309639 | +107.3763212574 | -0.4202888298 | 비채택 |

### 로컬 검증과 리더보드 비교

- aggressive는 2023·2024 rolling Skill이 strict보다 각각 `+28.78`, `+4.36` 높았지만, 2022는 `-130.77` 낮고 후보 정의가 post-hoc이었다.
- 실제 Public에서는 과거 OOF만으로 smoothing과 rank를 고정한 strict가 aggressive를 `0.4202888298` 앞섰다.
- 두 후보 모두 기존 최고 EXP-013을 107점 이상 개선해 source-season Team EB와 투수 문맥 효과의 2025 전이 가능성을 확인했다.
- 최신 두 fold의 작은 우위보다 여러 시즌 하방 안정성, 선택 절차의 독립성, 사후 선택 위험을 함께 본 strict 선택이 최종 결과에서도 더 높았다.

### 최종 상태

- 리더보드 선택: EXP-021 strict
- aggressive 용도: 진단 결과로 보존
- 제출 실행 상태: 두 후보 모두 성공
- 추가 제출 오류: 없음

---

## 새 제출 템플릿

```markdown
## SUB-000 — 제출 이름

### 기본 정보

- 제출 날짜:
- 작성자:
- 연결 실험:
- Git commit:
- ZIP 파일:
- 모델 파일:
- 제출 상태: 제출 전 / 성공 / 설치 오류 / 제출 오류

### 로컬 검증

- 검증 기간:
- Brier Score:
- Brier Skill Score:
- 샘플 추론 시간:
- 모델 크기:
- ZIP 크기:

### 제출 전 검사

- [ ] script.py 실행 성공
- [ ] output/submission.csv 생성
- [ ] 행 개수와 row_id 순서 일치
- [ ] 예측값 결측 없음
- [ ] 예측값 0~1 범위
- [ ] ZIP 구조 확인
- [ ] 불필요한 파일 제외
- [ ] 인터넷 접속 코드 없음
- [ ] 테스트 데이터 내부 집계 없음

### 제출 결과

- 설치 상태:
- 실행 상태:
- Public Score:
- 순위:
- 실행 시간:
- 오류 메시지:

### 로컬과 리더보드 비교

차이와 원인에 대한 가설을 작성한다.

### 다음 제출에서 변경할 내용

다음 작업을 작성한다.
```
