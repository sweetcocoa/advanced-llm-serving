---
title: "Linear Attention과 Hybrid 모델: DeltaNet에서 Kimi Linear까지"
module: "LLM 모델 이론"
chapter: "Linear Attention과 Hybrid 모델: DeltaNet에서 Kimi Linear까지"
format: "dialogue-lecture"
dialogue_tone: "tutoring"
estimated_reading_time: "20-30 min"
prerequisites: ["Transformer와 Autoregressive Language Model"]
visuals:
  formulas: 1
  mermaid: 0
  local_images: 0
updated_at: "2026-11-20"
source_count: 3
---

# Linear Attention과 Hybrid 모델: DeltaNet에서 Kimi Linear까지

## 수업 개요

문서에 한 줄이 붙을 때마다 과거 위치의 목록도 길어지는 설계가 있고, 기존 요약 상태에 새 정보를 써 넣는 설계가 있다. causal linear attention의 기본형과 Gated DeltaNet, Kimi Linear를 따라가며 각 층이 과거에 접근하는 방식을 비교한다. [S1][S2][S3]

## 토큰을 저장하는 대신 상태에 쓰기

**학습자:** DSA처럼 과거 토큰 수천 개를 골라 읽는 대신, 처음부터 과거를 요약해 둘 수 있나요?

**교수자:** 기본 causal linear attention은 feature map을 적용한 key와 value의 외적을 누적하고, query feature로 상태를 읽습니다. 아래 식은 읽기 전에 현재 토큰을 포함하며 정규화 누산기까지 표시한 기본형입니다. feature 차원과 value 차원이 고정돼 있으면 층 상태 크기는 문맥 길이에 따라 증가하지 않습니다. [S1]

$$
S_t=S_{t-1}+\phi(k_t)v_t^\top,\qquad
z_t=z_{t-1}+\phi(k_t),\qquad
y_t=\frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}
$$

**교수자:** 긴 문서에서 “37번 토큰만 읽어 보자”는 요구를 생각해 보세요. 위 상태에는 그 위치의 KV를 독립적으로 꺼내는 주소가 없습니다. 어떤 정보가 살아남는지는 학습한 상태 갱신과 읽기 방식에 달려 있습니다. 반면 DSA는 후보를 골라 특정 토큰 위치를 참조합니다. 여기서 linear는 고정 차원 기준으로 길이에 따른 계산 스케일링을 설명합니다. [S1]

## 단순 누적에서 선택적 갱신으로

- **Delta rule**은 현재 상태의 key 연관 예측과 새 value의 차이를 사용해 갱신한다. 이미 기억에 있는 연관을 덮어쓰는 상황을 생각하면 단순 외적 누적과의 차이가 드러난다. Gated DeltaNet은 이전 상태를 얼마나 남길지 조절하는 gate와 delta 갱신을 결합한다. [S2]
- **Kimi Delta Attention(KDA)**은 Gated DeltaNet 계열에 더 세분된 gate를 적용한다. Kimi Linear는 KDA 층과 MLA 층을 섞는다. 이 경우 KDA 층은 반복 상태를, MLA 층은 토큰별 cache를 유지하므로 모델 전체의 메모리 추정에는 두 비용이 모두 들어간다. [S3]

## 같은 계약서에서 무엇을 기억할까

| 층의 방식 | 긴 문서의 앞부분에 접근하는 경로 |
| --- | --- |
| Sliding window | 현재 위치의 창에 포함된 토큰을 직접 참조 |
| DSA | 현재 쿼리가 고른 과거 위치의 KV를 직접 참조 |
| Causal linear attention | 상태 갱신에 보존된 정보를 query로 읽음 |
| KDA + MLA hybrid | 층마다 반복 상태 또는 토큰별 KV를 사용 |

기존 [DSA 장](../02-long-context-sparse-attention/README.md)의 토큰 선택 예제와 함께 읽고, 정확한 문자열 회수와 문서 전체 요약처럼 성격이 다른 과제를 각각 평가해 보자. 고정 크기 상태라는 사실만으로 개별 과제의 품질이나 속도는 정해지지 않는다. [S1][S3]

## 출처

| ID | 출처 | 발행처 | 날짜 | URL | 사용 이유 |
| --- | --- | --- | --- | --- | --- |
| [S1] | Transformers are RNNs §3 | Katharopoulos et al. | 2020 | https://arxiv.org/abs/2006.16236 | kernel feature map과 causal recurrence |
| [S2] | Gated Delta Networks | Yang et al. | 2024 | https://arxiv.org/abs/2412.06464 | gate와 delta rule이 맡는 서로 다른 상태 갱신 역할 |
| [S3] | Kimi Linear | Moonshot AI | 2025 | https://arxiv.org/abs/2510.26692 | KDA의 gate 설계와 MLA 혼합 모델 |
