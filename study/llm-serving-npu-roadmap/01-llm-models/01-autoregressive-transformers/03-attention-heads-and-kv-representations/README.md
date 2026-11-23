---
title: "어텐션 헤드와 KV 표현: MHA, MQA, GQA, MLA"
module: "LLM 모델 이론"
chapter: "어텐션 헤드와 KV 표현: MHA, MQA, GQA, MLA"
format: "dialogue-lecture"
dialogue_tone: "tutoring"
estimated_reading_time: "45-60 min"
prerequisites: ["Transformer와 Autoregressive Language Model"]
visuals:
  formulas: 5
  mermaid: 0
  local_images: 0
updated_at: "2026-11-23"
source_count: 4
---

# 어텐션 헤드와 KV 표현: MHA, MQA, GQA, MLA

## 수업 개요

체크포인트에 query head가 32개라고 쓰여 있다. 이 숫자만으로는 토큰 하나를 추가할 때 KV cache가 얼마나 커지는지 계산할 수 없다. MHA, MQA, GQA는 **몇 개의 query head가 K와 V를 공유할지**를 정한다. MLA는 이후 계산에 쓸 K와 V 정보를 **어떤 표현으로 저장할지** 바꾼다. [S1][S2][S3][S4]

## 학습 목표

- 배치, 시퀀스 길이, head 개수, head 차원으로 Q·K·V의 모양을 적고, prefill과 decode에서 각각 어떤 축이 달라지는지 추적한다.
- MHA·MQA·GQA의 head 배정과 KV cache 크기를 직접 계산하고, head를 공유해도 query마다 읽은 결과가 달라지는 이유를 설명한다.
- MLA에서 latent와 위치용 key를 캐시하는 이유를 projection 이동과 RoPE의 위치 의존성으로 설명한다.
- 체크포인트의 논리적 캐시 크기와 런타임에서 관찰한 메모리·지연을 구분해 해석한다.

## 한 토큰에서 여러 헤드가 나오는 과정

**학습자:** head가 32개라면 입력 문장을 32조각으로 나눠서 각각 읽는 건가요?

**교수자:** 각 head는 같은 위치의 hidden vector를 자신의 projection으로 읽습니다. 한 토큰의 hidden vector를 Q, K, V로 변환하고 각 projection의 결과를 head 축으로 나눕니다. 여러 head의 읽기 결과를 이어 붙인 뒤 output projection이 다음 층으로 보낼 벡터를 만듭니다. 따라서 head 수는 시퀀스를 나누는 방법과 무관합니다. [S1]

세 토큰으로 구성된 `[계약, 갱신, 조건]`을 생각해 보자. 마지막 위치에서 query head 하나는 앞선 각 토큰의 key와 점수를 계산하고, causal mask가 허용한 위치의 value를 가중합한다. 각 head는 같은 세 위치를 볼 수 있으면서 서로 다른 가중치를 줄 수 있다. 어떤 head가 실제로 문법이나 특정 의미를 전담하는지는 이 구조만으로 알 수 없다.

아래는 배치 크기 2, query head 4개, head dimension 8인 작은 GQA 모델을 예로 든다. KV head는 2개다. 이 표는 설명용 축 순서이며 실제 엔진의 메모리 배치는 다를 수 있다. prefill에는 토큰 6개, 다음 decode에는 토큰 1개를 넣고 새 KV까지 포함해 계산한다.

| 텐서 | 6토큰 prefill | 다음 1토큰 decode |
| --- | --- | --- |
| Q, `[batch, heads, queries, dim]` | `[2, 4, 6, 8]` | `[2, 4, 1, 8]` |
| 사용할 K 또는 V, `[batch, KV heads, keys, dim]` | `[2, 2, 6, 8]` | `[2, 2, 7, 8]` |
| head별 논리적 점수 | `[2, 4, 6, 6]` | `[2, 4, 1, 7]` |
| output projection 전의 head 출력 | `[2, 4, 6, 8]` | `[2, 4, 1, 8]` |

decode에서 길이 7인 축은 과거와 현재의 key를 가리킨다. 해당 단계에 과거 query 여섯 개를 다시 만들 필요는 없다. 또 표의 점수 모양은 수학적으로 계산할 쿼리–키 조합을 나타낸다. 전체 점수 행렬을 실제로 할당하는지는 [커널과 백엔드 장](../../../02-serving-systems/02-cache-batching-and-generation/08-attention-kernels-and-backends/README.md)의 타일링 예제에서 확인할 수 있다.

## Query head와 KV head를 따로 세기

**교수자:** query head가 32개이고 KV head가 8개면, 연속된 query head 네 개씩 같은 K/V head를 읽는 구성을 생각할 수 있습니다. 예를 들어 첫 네 head는 KV head 0을 읽습니다. 각 query head의 Q는 여전히 다르므로 어텐션 점수와 출력도 달라질 수 있습니다. [S3]

### 같은 KV를 읽어도 답이 달라지는 이유

앞의 작은 예에서 query head `0, 1`은 KV head 0에, `2, 3`은 KV head 1에 배정된다. head 수를 4/4로 두면 MHA, 4/1로 두면 MQA다. 균등하게 배정한다면 query head 수가 KV head 수로 나누어떨어져야 한다. [S2][S3]

$$
g(i)=\left\lfloor\frac{i}{H_Q/H_{\mathrm{KV}}}\right\rfloor,\qquad
o_{t,i}=\sum_{s\le t}\operatorname{softmax}_{s}
\left(\frac{q_{t,i}^{\top} k_{s,g(i)}}{\sqrt{d_k}}\right)v_{s,g(i)}
$$

두 위치만 포함한 한 그룹에서 첫 위치의 value가 10, 둘째 위치의 value가 30이라고 하자. query head 0이 가중치 `[0.9, 0.1]`을 내면 출력은 12다. 같은 두 value를 읽는 head 1이 `[0.2, 0.8]`을 내면 출력은 26이다. 이 숫자는 softmax 이후를 보여 주는 학습용 예다. KV를 한 벌만 저장해도 각 query는 자신의 점수 분포를 만든다.

### 토큰당 수치를 요청 하나의 예산으로 바꾸기

| 방식 | KV head 수 | 층당 토큰당 KV, BF16·head dim 128 |
| --- | ---: | ---: |
| MHA | 32 | 16 KiB |
| GQA | 8 | 4 KiB |
| MQA | 1 | 512 B |

BF16 요소가 2바이트이고 K와 V의 head dimension이 같다는 예다. 패딩, 페이지 여유분, 런타임의 head 복제는 빼고 논리적 크기만 센다. [S1][S2][S3]

$$
B_{\mathrm{KV/token/layer}} = 2 H_{\mathrm{KV}} d_{\mathrm{head}} b
$$

첫 번째 2는 K와 V 두 텐서를 뜻하고, 마지막의 $b$는 요소당 저장 바이트다. K·V 차원이 다르면 `2 × head dimension` 자리에 `key dimension + value dimension`을 넣는다. 층 수나 dtype이 다른 모델을 비교할 때는 층별로 계산하고 합한다. 아래는 32개 층이 같은 방식을 쓰고, 각 층에서 8,192토큰을 모두 보관하는 요청 하나의 예다.

| 방식 | 계산 | 요청당 논리적 KV |
| --- | --- | ---: |
| MHA | 16 KiB × 32 × 8,192 | 4 GiB |
| GQA | 4 KiB × 32 × 8,192 | 1 GiB |
| MQA | 512 B × 32 × 8,192 | 128 MiB |

같은 GQA 요청 8개가 각자의 캐시를 가지면 8 GiB다. 이 값은 모델 가중치, 페이지의 빈 공간, 실행 중간 버퍼와는 별도다. 두 요청이 같은 prefix를 물리적으로 공유하면 실제 캐시의 중복은 줄 수 있고, 런타임이 KV head를 여러 rank에 복제하면 각 장치의 합은 논리적 크기보다 커질 수 있다. 용량 추정은 한 요청의 논리적 바이트로 시작하고, 배포 산정에서는 엔진의 캐시 레이아웃을 확인한다.

**학습자:** 16 KiB가 4 KiB면 decode도 네 배 빨라지나요?

**교수자:** 줄어든 것은 논리적 KV 용량입니다. 커널이 공유 KV를 실제로 재사용하는지, KV 읽기가 전체 지연에서 얼마나 큰지까지 봐야 합니다. query head 수를 그대로 뒀다면 독립적으로 계산할 query 쪽 출력도 그대로 남습니다. [S2][S3]

예를 들어 한 decode 단계에 10 ms가 들고 그중 KV 읽기와 직접 연관된 부분이 4 ms라고 가정하자. 그 구간이 이상적으로 1 ms로 줄어도 나머지 6 ms가 그대로라면 총 7 ms, 약 1.43배다. 실제 측정값을 예측하는 공식이라기보다 부분 최적화와 전체 지연 사이의 차이를 보여 주는 계산이다. KV cache 감소는 동일 메모리에서 동시 요청을 늘리는 데도 쓰일 수 있으므로, 토큰 간 지연과 동시 처리량을 함께 본다.

### 체크포인트를 바꿀 때

**학습자:** 이미 학습한 MHA의 설정 파일에서 KV head만 줄이면 GQA 실험을 할 수 있나요?

**교수자:** 원래 체크포인트에는 head별 K/V projection 가중치가 있습니다. 그룹 수만 바꾸면 가중치 모양부터 맞지 않을 수 있습니다. GQA 논문에서는 기존 K·V projection을 그룹별로 평균 내어 초기화하고 추가 사전학습을 진행했습니다. 그룹별 평균은 학습 가중치에 적용하는 변환입니다. DSA의 KPool이 실행 중 여러 **토큰 위치**의 선택용 key를 묶는 과정과는 다른 축입니다. [S3]

## MLA는 무엇을 캐시하는가

DeepSeek-V2의 MLA는 각 토큰에서 **공동 KV 잠재 벡터**와 **위치 정보를 담는 decoupled RoPE key**를 계산한다. 이후 쿼리가 읽을 K/V 정보를 이 표현으로 보존한다. 구현은 업프로젝션을 인접한 연산에 흡수해 토큰마다 모든 head의 K/V를 물리적으로 재구성하는 비용을 피할 수 있다. KV cache를 비교할 때는 KV head 개수 대신 잠재 벡터와 위치 key의 차원을 각각 확인한다. [S4]

### 1. 공통 latent에서 head별 K와 V가 나온다

토큰 $s$의 hidden vector를 $h_s$라고 하자. down-projection은 이를 길이 $d_c$인 $c_s$로 줄인다. head $i$는 그 latent를 자신의 K·V up-projection으로 읽는다. 같은 latent를 공유해도 서로 다른 projection을 거친 K·V는 head별로 달라질 수 있다. 이 구조의 표현력과 cache 크기를 비교할 때는 latent의 폭과 head마다 어떤 변환을 할 수 있는지를 함께 봐야 한다. [S4]

$$
c_s=W^{DKV}h_s,\qquad k_{s,i}^{C}=U_i^Kc_s,\qquad v_{s,i}^{C}=U_i^Vc_s
$$

### 2. 과거 위치마다 K·V를 펼치지 않고 계산하기

**학습자:** decode에서 과거 토큰이 8천 개라면 그때마다 모든 head의 K·V를 복원하나요?

**교수자:** 내용 점수에서는 query와 행렬을 먼저 곱할 수 있습니다. 아래 첫 식의 괄호를 보면 head별로 이미 변환한 query를 여러 과거 latent에 그대로 적용할 수 있습니다. value 쪽의 선형 변환도 가중합 뒤로 옮길 수 있습니다. 실제 커널마다 계산 순서와 최적화는 달라질 수 있습니다. [S4]

$$
(q_{t,i}^{C})^\top U_i^K c_s
=\big((U_i^K)^\top q_{t,i}^{C}\big)^\top c_s,\qquad
\sum_s a_{t,s,i}U_i^V c_s=U_i^V\sum_s a_{t,s,i}c_s
$$

작은 스칼라 예를 들어 head의 value 변환이 `×3`, 두 latent가 `2, 8`, 이미 구한 가중치가 `0.25, 0.75`라면 먼저 값을 변환한 뒤 가중합해도 19.5다. latent를 먼저 가중합하면 6.5이고 여기에 3을 곱해도 19.5다. 실제로는 벡터와 행렬에서 같은 선형성이 성립한다.

### 3. 위치는 따로 읽는다

내용 key 전체에 위치마다 다른 RoPE 회전을 끼우면, key의 up-projection을 앞 식처럼 위치와 무관한 query 쪽 변환으로 옮기기 어렵다. DeepSeek-V2는 위치 정보를 위한 rotary query를 head별로 만들고, 토큰마다 공유 rotary key를 하나 둔다. 실제 점수는 내용 부분과 rotary 부분의 내적을 합한 뒤 논문의 차원 스케일과 softmax를 적용한다. 캐시에는 **공동 KV latent와 공유 rotary key**가 남는다. query를 낮은 차원으로 투영하는 별도 경로도 있지만 과거 query를 매 토큰 저장하는 계산에는 들어가지 않는다. [S4]

$$
B_{\mathrm{MLA/token/layer}}=d_c b_c+d_R b_R
$$

DeepSeek-V2의 $d_c=512$, $d_R=64$를 넣고 두 성분을 모두 BF16으로 저장한다고 가정하면 `512×2 + 64×2 = 1,152 B`다. 같은 저장 가정을 앞의 32층·8,192토큰 예제에 적용하면 288 MiB다. 이는 비교를 위해 층 수를 맞춘 산술 예이고, 실제 DeepSeek-V2 배포 메모리 수치와는 별도다. 이 예에서 MQA는 512 B로 더 작고, GQA는 4,096 B로 더 크다. MLA와 GQA의 순서는 비교하는 head 구성, latent 차원, 저장 dtype에 따라 결정된다. [S4]

## 모델 설정에서 무엇을 확인할까

긴 계약서의 앞부분을 읽는 서비스가 느릴 때는 먼저 요청당 몇 바이트가 쌓이는지 계산하자. 그다음 현재 쿼리가 저장된 후보 전체를 읽는지 일부만 고르는지 확인한다. 마지막으로 어떤 캐시가 실제 가속기 메모리에 올라가고 어떤 방식으로 읽히는지 살펴보면, 용량과 토큰 지연을 같은 이유로 설명하는 실수를 줄일 수 있다.

| 체크포인트에서 확인할 항목 | 답할 수 있는 질문 |
| --- | --- |
| Query head와 KV head 수 | MHA·MQA·GQA에서 몇 head가 캐시를 공유하나? |
| KV latent 차원과 RoPE key 차원 | MLA의 토큰당 캐시 항목은 얼마나 큰가? |
| Attention mask와 선택 규칙 | 저장된 항목 중 쿼리가 어떤 위치를 읽나? |
| 캐시 dtype과 tensor parallel 배치 | 계산한 논리적 KV는 장치별로 얼마나 분할되거나 복제되나? |

**교수자:** [DeepSeek-V3.2의 MLA와 DSA 사례](../02-long-context-sparse-attention/README.md)에서는 토큰별 cache 표현과 참조 위치 선택이 각각 비용을 바꿉니다. 토큰당 캐시 바이트와 쿼리당 읽은 항목 수를 각각 적어 보세요. 물리적 캐시와 실행 비용은 [KV Cache and PagedAttention](../../../02-serving-systems/02-cache-batching-and-generation/01-kv-cache-and-paged-attention/README.md)에서 이어서 다룹니다.

## 출처

| ID | 출처 | 발행처 | 날짜 | URL | 사용 이유 |
| --- | --- | --- | --- | --- | --- |
| [S1] | Attention Is All You Need §3.2.2 | Vaswani et al. | 2017 | https://arxiv.org/abs/1706.03762 | MHA의 독립적인 projection과 head 출력 |
| [S2] | Fast Transformer Decoding | Shazeer | 2019 | https://arxiv.org/abs/1911.02150 | MQA의 K/V 공유와 decode 메모리 대역폭 |
| [S3] | GQA §2 | Ainslie et al. | 2023 | https://arxiv.org/html/2305.13245 | 그룹별 K/V 공유와 MHA 체크포인트 변환 절차 |
| [S4] | DeepSeek-V2 §2.1, §3.1.2 | DeepSeek-AI | 2024 | https://arxiv.org/html/2405.04434 | MLA 공동 KV 압축, decoupled RoPE, projection 흡수와 실제 모델 차원 |
