# 데이터 배치 안내

대회에서 제공받은 원본 데이터는 저장소에 포함하지 않습니다.

노트북과 실험 코드를 실행하려면 이 디렉터리에 다음 파일을 직접 배치합니다.

```text
data/
├── train.csv
├── test.csv
├── sample_submission.csv
└── trackman_history.csv
```

- `train.csv`: 학습 입력과 Target `control_success`
- `test.csv`: 추론 입력
- `sample_submission.csv`: 제출 형식
- `trackman_history.csv`: 과거 Trackman 기록

데이터의 재배포 가능 여부는 대회 규정을 따르며, Git 커밋에는 포함하지 않습니다.
