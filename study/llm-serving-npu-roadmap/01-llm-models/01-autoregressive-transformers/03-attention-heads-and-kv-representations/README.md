---
title: "어텐션 헤드와 KV 표현: MHA, MQA, GQA, MLA"
module: "LLM 모델 이론"
chapter: "어텐션 헤드와 KV 표현: MHA, MQA, GQA, MLA"
format: "dialogue-lecture"
dialogue_tone: "tutoring"
estimated_reading_time: "20-30 min"
prerequisites: ["Transformer와 Autoregressive Language Model"]
visuals:
  formulas: 1
  mermaid: 0
  local_images: 0
updated_at: "2026-11-20"
source_count: 4
---

# 어텐션 헤드와 KV 표현: MHA, MQA, GQA, MLA

## 수업 개요

체크포인트에 query head가 32개라고 쓰여 있다. 이 숫자만으로는 토큰 하나를 추가할 때 KV cache가 얼마나 커지는지 계산할 수 없다. MHA, MQA, GQA는 **몇 개의 query head가 K와 V를 공유할지**를 정한다. MLA는 이후 계산에 쓸 K와 V 정보를 **어떤 표현으로 저장할지** 바꾼다. [S1][S2][S3][S4]

## Query head와 KV head를 따로 세기

**교수자:** query head가 32개이고 KV head가 8개면, 연속된 query head 네 개씩 같은 K/V head를 읽는 구성을 생각할 수 있습니다. 예를 들어 첫 네 head는 KV head 0을 읽습니다. 각 query head의 Q는 여전히 다르므로 어텐션 점수와 출력도 달라질 수 있습니다. [S3]

| 방식 | KV head 수 | 층당 토큰당 KV, BF16·head dim 128 |
| --- | ---: | ---: |
| MHA | 32 | 16 KiB |
| GQA | 8 | 4 KiB |
| MQA | 1 | 512 B |

BF16 요소가 2바이트이고 K와 V의 head dimension이 같다는 예다. 패딩, 페이지 여유분, 런타임의 head 복제는 빼고 논리적 크기만 센다. [S1][S2][S3]

$$
B_{\mathrm{KV/token/layer}} = 2 H_{\mathrm{KV}} d_{\mathrm{head}} b
$$

**학습자:** 16 KiB가 4 KiB면 decode도 네 배 빨라지나요?

**교수자:** 줄어든 것은 논리적 KV 용량입니다. 커널이 공유 KV를 실제로 재사용하는지, KV 읽기가 전체 지연에서 얼마나 큰지까지 봐야 합니다. query head 수를 그대로 뒀다면 독립적으로 계산할 query 쪽 출력도 그대로 남습니다. [S2][S3]

## MLA는 무엇을 캐시하는가

DeepSeek-V2의 MLA는 각 토큰에서 **공동 KV 잠재 벡터**와 **위치 정보를 담는 decoupled RoPE key**를 계산한다. 이후 쿼리가 읽을 K/V 정보를 이 표현으로 보존한다. 구현은 업프로젝션을 인접한 연산에 흡수해 토큰마다 모든 head의 K/V를 물리적으로 재구성하는 비용을 피할 수 있다. KV cache를 비교할 때는 KV head 개수 대신 잠재 벡터와 위치 key의 차원을 각각 확인한다. [S4]

| 체크포인트에서 확인할 항목 | 답할 수 있는 질문 |
| --- | --- |
| Query head와 KV head 수 | MHA·MQA·GQA에서 몇 head가 캐시를 공유하나? |
| KV latent 차원과 RoPE key 차원 | MLA의 토큰당 캐시 항목은 얼마나 큰가? |
| Attention mask와 선택 규칙 | 저장된 항목 중 쿼리가 어떤 위치를 읽나? |

세 번째 질문은 [DSA와 KPool](../02-long-context-sparse-attention/README.md)에서, 물리적 캐시와 실행 비용은 [KV Cache and PagedAttention](../../../02-serving-systems/02-cache-batching-and-generation/01-kv-cache-and-paged-attention/README.md)에서 이어서 다룬다.

## 출처

| ID | 출처 | 발행처 | 날짜 | URL | 사용 이유 |
| --- | --- | --- | --- | --- | --- |
| [S1] | Attention Is All You Need §3.2.2 | Vaswani et al. | 2017 | https://arxiv.org/abs/1706.03762 | MHA의 독립적인 projection과 head 출력 |
| [S2] | Fast Transformer Decoding | Shazeer | 2019 | https://arxiv.org/abs/1911.02150 | MQA의 K/V 공유와 decode 메모리 대역폭 |
| [S3] | GQA | Ainslie et al. | 2023 | https://arxiv.org/abs/2305.13245 | 그룹별 K/V 공유와 MHA·MQA 비교 |
| [S4] | DeepSeek-V2 §2.1 | DeepSeek-AI | 2024 | https://arxiv.org/abs/2405.04434 | MLA 공동 KV 압축, decoupled RoPE, projection 흡수 |
