# EXP-105 Correction Geometry Audit

## 범위와 정합성

공통 기준은 EXP-051 `trackman_direct_recent_w010`이다. 저장된 OOF에서 다음을 정의했다.

```text
c063 = p063 - p051
c064 = p064 - p051
c071 = p071 - p051
c072 = p072 - p051
r0   = y - p051
```

- 2021~2024에서 EXP-063/064/072의 EXP-051 base는 최대 절대 차이 `0.0`으로 일치했다.
- EXP-071이 존재하는 2022~2024에서도 base 최대 절대 차이는 `0.0`이었다.
- 네 시즌 target은 공식 `train.csv` 행 순서와 모두 bitwise 일치했다.
- EXP-071은 prior EXP-051 OOF residual이 필요한 구조라 2021 prediction을 적법하게 만들 수 없다. 따라서 2021 `c071`은 “mechanism unavailable”을 뜻하는 정확한 `0`으로 두었다. 2021 EXP-071 확률을 새로 추정하거나 역산하지 않았다.

Authoritative EXP-071 재계산값은 다음과 같다.

| Season | Rows | Brier | Skill |
|---:|---:|---:|---:|
| 2022 | 247,472 | 0.244900654279 | 1710.9054 |
| 2023 | 245,525 | 0.247660513870 | 935.7937 |
| 2024 | 253,507 | 0.247604672468 | 881.5826 |

## Correction끼리의 상관

2021~2024 pooled OOF correction Pearson correlation이다.

| Pair | Correlation |
|---|---:|
| c063–c064 | 0.018034 |
| c063–c071 | 0.049182 |
| c063–c072 | 0.042816 |
| c064–c071 | 0.006818 |
| c064–c072 | 0.028325 |
| c071–c072 | 0.110432 |

가장 큰 값도 `0.110432`다. 네 correction은 완성 확률 수준에서 보였던 유사성과 달리 서로 거의 직교에 가까운 방향이다.

## 완성 prediction error 상관

같은 행에서 `(p051 + cj - y)`의 상관을 계산하면 결과가 정반대다.

| Pair | Complete-error correlation |
|---|---:|
| p063–p064 | 0.99996454 |
| p063–p071 | 0.99997601 |
| p063–p072 | 0.99998413 |
| p064–p071 | 0.99996360 |
| p064–p072 | 0.99997268 |
| p071–p072 | 0.99998458 |

즉 완성 prediction error correlation이 높았던 주된 이유는 공통 EXP-051 오차가 지배했기 때문이다. EXP-105의 핵심 가설인 “완성 모델이 아니라 correction field로 보면 다른 basis가 드러난다”는 geometry 수준에서는 확인됐다.

## EXP-051 residual과의 관계

| Correction | corr(c, r0) | cov(c, r0) | Mean abs correction | Non-zero rate |
|---|---:|---:|---:|---:|
| c063 | 0.007383 | 9.0392e-6 | 0.001512 | 0.5877 |
| c064 | 0.007779 | 1.3212e-5 | 0.002018 | 0.4754 |
| c071 | 0.001032 | 1.2915e-6 | 0.001593 | 0.5936 |
| c072 | 0.004813 | 3.4142e-6 | 0.000572 | 0.8281 |

전체 상관은 모두 작지만 covariance는 양수다. 매우 큰 행 수 때문에 작은 전체 상관 자체를 강한 증거로 해석하지 않았다. 중요한 문제는 방향의 계절 안정성이다.

## Segment audit

Segment는 gate 학습이 아니라 진단에만 사용했다. 주요 관찰은 다음과 같다.

- `c063`은 2022 `game_type=F`에서 residual correlation `+0.0791`이었지만, 2023 batter history `1~19`에서는 `-0.0404`였다.
- `c064`는 2022 batter history `1~19`에서 `+0.0694`였지만, 같은 시즌 `game_type=F`에서 `-0.0670`, 2024 batter history `1~19`에서 `-0.0662`였다.
- `c071`은 2022 batter history `1~19`에서 `-0.0600`이었다. count `12`는 2022 `-0.0276`, 2024 `+0.0311`로 방향이 바뀌었다.
- `c072`는 AR reliability `0.5~1.0`인 2022 행에서 `+0.0578`로 가장 명확했고, 2024 batter history `20~99`에서도 `+0.0365`였다.
- physical lookup availability, pitcher/batter history, count, runners, game type 전반에서 한 mechanism이 모든 계절에 같은 우위를 보이지 않았다.

이 값들은 다수 segment를 훑은 사후 진단이며 multiple-comparison 보정이 없다. 새로운 threshold 또는 validation-specific gate를 만드는 데 사용하지 않았다.

## Reliability 정의

EXP-108에 사용한 reliability는 validation label로 학습하지 않았다.

- EXP-063: 기존 `abs(p051 - 0.5) < 0.06` eligibility.
- EXP-064: prior-season stable cell의 총 source count와 기존 smoothing `500`으로 만든 `count / (count + 500 × source seasons)`.
- EXP-071: frozen TrackMan pitcher/context lookup support와 기존 context smoothing으로 만든 support reliability.
- EXP-072: latest prior-season pitcher count posterior reliability와 current-season `k=30` prior retention의 곱.

Correction 자체가 이미 gating/smoothing된 상태이므로 EXP-108 reliability는 추가 보수화다.

## Orthogonal basis

2021~2024 final correction covariance의 singular-value 제곱 비율은 다음과 같다.

| Basis | Variance share |
|---:|---:|
| z1 | 44.75% |
| z2 | 25.11% |
| z3 | 22.48% |
| z4 | 7.66% |

첫 두 basis가 약 `69.86%`를 설명하지만, 세 번째도 `22.48%`로 무시할 수 없다. 즉 correction matrix가 사실상 rank-1이라서 조합이 실패한 것은 아니다. EXP-109가 실패한 원인은 독립 방향의 부재보다 target alignment와 계절 안정성 부족이다.

## 결론

Geometry 가설은 확인됐다. 네 correction은 서로 다르다. 그러나 “서로 다름”만으로 Brier 개선이 보장되지는 않았다. correction-to-residual alignment가 작고 segment 방향이 계절마다 바뀌어, 자유도가 있는 ridge/SVD 조합은 2023 또는 2024에서 EXP-071을 안정적으로 이기지 못했다. 설명 가능한 원래 mechanism을 유지한 EXP-110만 recent pooled에서 `-6.8061e-7`의 매우 작은 개선을 남겼다.

Authoritative machine-readable artifact: `artifacts/EXP-105/correction_geometry/report.json` (SHA256 `ca038546dcc9460755636d2dac088fdd76772c6c80a1a06e4739f8ef3068962d`).
