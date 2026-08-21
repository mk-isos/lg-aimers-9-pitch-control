# Research Report: EXP-105+

## 결론

**1100+를 로컬에서 뒷받침하는 후보는 발견하지 못했다.** Public best `1053.8615519684`를 넘어설 가능성을 제한적으로 시험할 후보는 EXP-110 하나뿐이다. EXP-110은 2024와 2022를 개선했지만 2023은 악화했고, 2023~2024 pooled EXP-071 대비 개선은 Brier `-6.8061e-7`에 불과하다. 이를 강한 local evidence로 과장하지 않는다.

EXP-106/107/108/109는 모두 tier C로 폐기했다. 결과를 본 뒤 weight, threshold, rule을 추가 탐색하지 않았고, 두 번째 신규 ZIP을 억지로 만들지 않았다. 기존 EXP-071 ZIP은 reference/control로 이미 보존돼 있다.

## Protocol

- Common baseline: EXP-051 `trackman_direct_recent_w010`.
- Correction basis: EXP-063 `close060_last_w025`, EXP-064 `stable_count_runners_pbin_w050`, EXP-071 `playerphys_resid_w025`, EXP-072 `ar_k30_w050`.
- Meta target: `y - p051`; intercept `0`.
- Ridge objective: season-equal weighted SSE + `lambda × ||w||²`.
- Lambda grid: `[1, 10, 100, 1000]`; outer fold보다 과거인 OOF season의 leave-one-season-out으로만 선택.
- 2021만 존재하는 초기 fit은 사전 고정 fallback `lambda=100`.
- Public score, 2025 label, canonical test distribution은 numeric fitting에 사용하지 않았다.
- 2021 EXP-071 correction은 mechanism unavailable을 나타내는 `0`; 가짜 prediction을 만들지 않았다.

## Correction geometry

가장 중요한 발견은 완성 모델과 correction space의 차이다.

- Correction pairwise correlation: `0.006818~0.110432`.
- Complete prediction error correlation: `0.99996360~0.99998458`.
- SVD variance share: `44.75% / 25.11% / 22.48% / 7.66%`.

따라서 네 mechanism은 실제로 다른 residual direction이다. 그러나 각 correction과 `y-p051`의 전체 correlation은 `0.0010~0.0078`로 작고, 주요 segment의 부호가 season 사이에서 자주 바뀌었다. 자세한 표는 `docs/CORRECTION_GEOMETRY_EXP105.md`에 있다.

## EXP-106~111

### EXP-106 — Brier-optimal ridge correction stack

2025 final historical fit은 `lambda=10`, coefficient `[c063,c064,c071,c072] = [0.57438, 0.49919, 0.01470, 0.26677]`을 선택했다. 그러나 outer 2023 coefficient는 `[2.46551, 1.04327, -1.05421, 1.31077]`, outer 2024는 거의 0으로 수축했다. sign과 scale이 불안정하고 recent pooled Brier가 EXP-071보다 `+6.9615e-5` 악화해 폐기했다.

### EXP-107 — Nonnegative constrained stack

`wj >= 0`, `sum(wj) <= 2`를 적용했다. Final constraint는 비활성이라 EXP-106과 같은 coefficient가 나왔지만, outer 2023에서는 `[1.41144, 0.58856, 0, 0]`으로 sum bound에 닿았다. Recent pooled delta는 `+2.0813e-5`로 악화해 폐기했다.

### EXP-108 — Reliability-scaled stack

Source-only support reliability를 각 correction에 곱했다. Final coefficient는 `[0.57826, 0.28550, 0.06890, 0.15484]`였다. 2023 악화가 `+1.0793e-4`, recent pooled 악화가 `+5.4729e-5`라 폐기했다. 이미 내부적으로 shrink된 correction을 다시 reliability-scale한 것이 지나치게 보수적이었다.

### EXP-109 — Orthogonal residual basis

Final inner LOSO는 rank `2`, `lambda=1`, basis coefficient `[-1.06539, 0.76232]`를 선택했다. Correction rank는 충분했지만 2023과 2024가 모두 EXP-071보다 나빴다. Recent pooled delta `+9.3478e-6`로 tier C다.

### EXP-110 — Mechanism-preserving rule composition

Formula는 다음과 같다.

```text
base = p051 + c071
aux  = mean(active c063,
            active c064,
            c072 only when EXP-071 physical lookup is unavailable)
p110 = clip(base + alpha * aux, 0, 1)
```

`alpha`는 매 outer fold와 final 2021~2024 historical OOF에서 모두 `[0,1]` 경계 내 closed-form residual fit으로 구했고, 모든 경우 `1.0`이었다. 새 threshold는 만들지 않았다. 2023은 Brier `+1.2209e-5` 악화했지만 2024는 `-1.3165e-5`, 2022는 `-6.9750e-5` 개선했다. Recent pooled delta `-6.8061e-7`, 전체 2022~2024 pooled delta `-2.3578e-5`로 tier B다.

### EXP-111 — Temporal-shift control

실행하지 않았다. Cai and Ye의 ICML 2025 방법은 continuous timestamp의 trend와 year/month/week/day Fourier periods를 사용하며, official implementation은 fixed period prior로 frequency를 초기화한 뒤 temporal embedding을 학습한다. [PMLR paper](https://proceedings.mlr.press/v267/cai25j.html), [official code](https://github.com/LAMDA-Tabular/Tabular-Temporal-Shift), [temporal embedding implementation](https://raw.githubusercontent.com/LAMDA-Tabular/Tabular-Temporal-Shift/main/model/lib/temporal_embeddings.py).

우리 공식 row에는 `season`, `game_month`, `game_dayofweek`만 있고 exact timestamp가 없다. Row order로 날짜를 복원하면 test-row independence를 위반한다. 사용 가능한 month/weekday harmonic만 넣으면 EXP-057 discrete calendar EB의 smooth reparameterization에 가까워 논문의 핵심 차이를 보존하지 못한다. Strict rolling source는 validation보다 최소 한 season 뒤처지지만 exact within-season training lag도 식별할 수 없다. 따라서 optional EXP-111을 정직하게 skip했다.

## EXP-071 대비 결과표

`Pooled Δ`는 2023~2024 행 수 가중 Brier delta이며 음수가 개선이다. Correction correlation은 EXP-105 전체 pairwise 범위다.

| EXP | Formula | 2022 Brier / Skill | 2023 Brier / Skill | 2024 Brier / Skill | Pooled Δ vs EXP-071 | Coefficient | Correction correlation | Tier | ZIP path | SHA256 |
|---|---|---:|---:|---:|---:|---|---|:---:|---|---|
| EXP-071 | `p051+c071` | 0.244900654279 / 1710.91 | 0.247660513870 / 935.79 | 0.247604672468 / 881.58 | 0 | `c071=1` | 0.0068~0.1104 | Reference | `ready_to_submit/2026-08-20-post58/EXP-071-PLAYERPHYS-RESID.zip` | `27ab6adbc4e2f460d705438dc5b6cbd0d9dffd9a0d519354a47d26aa6d240ad3` |
| EXP-106 | ridge `p051+Cw` | 0.244862344469 / 1726.28 | 0.247798810609 / 880.48 | 0.247607768979 / 880.34 | +6.9615e-5 | `[.57438,.49919,.01470,.26677]` | 0.0068~0.1104 | C | — | — |
| EXP-107 | nonnegative, `sum(w)<=2` | 0.244862344469 / 1726.28 | 0.247701994399 / 919.20 | 0.247605469100 / 881.26 | +2.0813e-5 | `[.57438,.49919,.01470,.26677]` | 0.0068~0.1104 | C | — | — |
| EXP-108 | reliability-scaled ridge | 0.244862436906 / 1726.24 | 0.247768441609 / 892.62 | 0.247607878046 / 880.30 | +5.4729e-5 | `[.57826,.28550,.06890,.15484]` | 0.0068~0.1104 | C | — | — |
| EXP-109 | centered SVD rank 2 + ridge | 0.244862314411 / 1726.29 | 0.247676290045 / 929.48 | 0.247607794312 / 880.33 | +9.3478e-6 | `z=[-1.06539,.76232]` | 0.0068~0.1104 | C | — | — |
| EXP-110 | physical-first deterministic rule | 0.244830904204 / 1738.90 | 0.247672723215 / 930.91 | 0.247591507760 / 886.85 | **-6.8061e-7** | `alpha=1.0` | 0.0068~0.1104 | **B** | `ready_to_submit/2026-08-21-correction-composition/EXP-110-MECHANISM-COMPOSITION.zip` | `12b27e9ae50ae3e4d93eac107c8ae413fe0c67d63f22978f7065a75dd596552e` |
| EXP-111 | fixed Fourier temporal control | — | — | — | — | — | — | Skipped | — | — |

## 왜 이 coefficient/component/ensemble인가

### 왜 이 coefficient인가?

Public score가 아니라 strict historical nested OOF에서 결정했다. EXP-110 `alpha=1.0`도 2021~2024 OOF closed-form residual fit 결과이며, Public 제출 전에 formula/state/ZIP을 freeze했다.

### 왜 이 component인가?

EXP-063/064/071/072는 각각 EXP-051 대비 별도로 사전 고정 제출되어 실제 2025 Public에서 개선된 historical mechanism family다. EXP-105는 이들이 완성 prediction error 관점에서는 거의 같아 보여도 correction field 관점에서는 다름을 확인했다.

### 왜 이 ensemble인가?

Brier Score는 squared probability loss이므로 `y-p051` residual을 직접 최소화했다. EXP-110은 자유로운 meta model이 불안정했던 결과를 받아들여 원래 mechanism의 eligibility를 보존하고 overlap만 deterministic mean으로 축소한다. Public 결과를 이용한 weight tuning은 하지 않았다.

## Public submission candidates

1. **Candidate A — EXP-110**: 유일한 신규 tier-B 후보. Upside는 Public-positive mechanism composition이고, downside는 recent pooled local gain이 사실상 neutral이라는 점이다.
2. **Control — EXP-071**: 현재 Public best이며 이미 제출·보존됨. 새 정보가 없으므로 재제출 우선순위는 낮다.

EXP-106/107/108/109는 C reject라 ZIP을 만들지 않았다. 목표 문서의 “명백히 나쁜 후보는 ZIP을 만들지 않는다”를 우선해 신규 ZIP 수는 1개로 제한했다.

## ZIP / SHA256

- EXP-110 ZIP: `ready_to_submit/2026-08-21-correction-composition/EXP-110-MECHANISM-COMPOSITION.zip`
- Bytes: `4,989,667`
- SHA256: `12b27e9ae50ae3e4d93eac107c8ae413fe0c67d63f22978f7065a75dd596552e`
- CRC: passed.
- Independence: batch, singleton, reverse, random permutation seed 42, split batch, duplicate batch 모두 passed; max absolute difference `0.0` at tolerance `1e-12`.
- Canonical `data/test.csv`와 `data/sample_submission.csv`는 build/QA에서 열지 않았다.

## 위험

- EXP-110의 2023~2024 pooled 개선 `6.8e-7`은 noise 규모일 수 있다.
- 2023 손실과 2024 개선이 거의 상쇄돼 local/Public mismatch 방향을 예측할 수 없다.
- EXP-071이 Public에서는 최고지만 historical residual ridge가 final `c071` weight를 `0.0147`까지 낮춘 것은 local selection과 2025 transfer 사이 regime mismatch가 크다는 증거다.
- Correction basis는 독립적이지만 target alignment가 작고 season별 sign이 바뀐다. 자유로운 coefficient는 과적합 위험이 높다.
- EXP-110의 auxiliary shrinkage가 경계 `1.0`에 닿았으므로 더 큰 weight를 시험하고 싶은 유혹이 생기지만, 그것은 사전 bound를 깨는 post-result sweep이므로 금지한다.
- 실제 DACON 실행 환경 smoke는 source-derived synthetic row로 통과했다. Codex는 실제 제출을 수행하지 않았다.

## 권장 제출 순서

1. **EXP-110** — 신규 정보 가치가 있는 유일한 candidate. 한 번만 제출하고 score를 확인한다.
2. **추가 신규 제출 없음** — 결과를 보고 weight/threshold/interpolation을 수정하지 않는다.
3. **EXP-071 유지** — EXP-110이 `1053.8615519684`를 넘지 못하면 현재 best를 그대로 유지한다.

Public `1100+` 또는 Top-100은 현재 evidence로 주장할 수 없다. EXP-110이 Public best를 넘는다면 mechanism-preserving composition의 전이 증거가 되지만, 그 뒤에도 같은 tranche 안에서 numeric retuning은 하지 않는다.
