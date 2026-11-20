---
title: "어텐션 커널과 백엔드: FlashAttention, FlashInfer, FlexAttention"
module: "서빙 시스템 설계"
chapter: "어텐션 커널과 백엔드: FlashAttention, FlashInfer, FlexAttention"
format: "dialogue-lecture"
dialogue_tone: "tutoring"
estimated_reading_time: "20-30 min"
prerequisites: ["KV Cache and PagedAttention"]
visuals:
  formulas: 0
  mermaid: 0
  local_images: 0
updated_at: "2026-11-20"
source_count: 8
---

# 어텐션 커널과 백엔드: FlashAttention, FlashInfer, FlexAttention

## 수업 개요

같은 모델에서도 긴 prompt를 한 번에 읽을 때와 토큰 하나를 decode할 때는 쿼리 개수와 캐시 읽기 패턴이 다르다. 이 장은 softmax attention을 실행하는 커널, 이를 호출하는 API, 엔진이 선택하는 백엔드를 구분한 뒤 같은 입력으로 정확도와 성능을 비교한다. [S1][S5][S6][S8]

## FlashAttention은 어디서 메모리를 아끼는가

일반적인 기준 구현은 점수 행렬을 만든 뒤 softmax와 V 곱을 실행한다. FlashAttention은 Q와 KV를 타일로 나누고, softmax의 행별 최댓값과 정규화 합, 가중 출력 누산값을 갱신합니다. 다음 타일에서 더 큰 점수가 나오면 이전 누산값을 새 최댓값에 맞춰 재조정합니다. 전체 점수 행렬을 HBM에 써 두지 않고도 같은 dense softmax 결과를 수치 오차 범위에서 얻습니다. dense prefill의 쿼리–키 쌍 수는 여전히 길이의 제곱에 비례합니다. [S1]

- **FA2**는 thread block과 warp 사이의 작업 분할을 개선한다. [S2]
- **FA3**는 Hopper의 비동기 실행을 활용해 연산을 겹치고 저정밀도 경로를 다룬다. [S3]
- **FA4**는 Blackwell에서 행렬 곱과 softmax의 처리량 차이를 고려해 파이프라인을 다시 설계한다. [S4]

## 코드에서 만나는 이름들

| 이름 | 역할 | 확인할 조건 |
| --- | --- | --- |
| PyTorch SDPA | 표준 scaled dot-product attention API. 입력과 환경에 맞는 구현을 선택 | mask, dtype, head dimension, 사용 가능한 커널 [S6] |
| FlexAttention | 점수 변경 함수와 BlockMask를 전달할 수 있는 PyTorch API | 고정 창처럼 통째로 건너뛸 블록을 표현할 수 있는지, 컴파일 비용 [S7] |
| FlashInfer | paged KV 등을 읽는 prefill·decode 커널을 제공하는 서빙 라이브러리 | 요청 길이 분포, KV 레이아웃, prefill·decode 경로 [S5] |
| vLLM attention backend | 커널과 캐시 레이아웃을 모델 요청에 연결하는 엔진 경로 | 장치, dtype, KV block size, head dimension과 기능 지원표 [S8] |

**학습자:** FlexAttention으로 sliding window를 만들면 길이 증가에 따라 계산량도 바로 줄어드나요?

**교수자:** 점수 변경 함수에서 창 밖 점수만 가리는 것과 BlockMask로 빈 블록을 건너뛰는 것은 실행 경로가 다릅니다. 창 경계에 걸친 블록은 내부 원소를 추가로 가려야 합니다. 설정한 마스크를 먼저 짧은 입력으로 검증하고, 실제로 건너뛴 블록과 실행 시간을 확인하세요. [S7]

## 실습 1. 동일한 Q, K, V로 결과 맞춰 보기

CUDA와 PyTorch SDPA Flash backend를 사용할 수 있는 환경에서 아래 코드를 실행한다. 짧은 causal self-attention을 FP32 math 결과와 비교하는 smoke test다. 허용 오차는 이 입력과 dtype을 위한 예시이며, 배포 모델의 정확도 기준은 별도로 정한다. 순수 GPU 커널 시간 비교에는 워밍업과 CUDA event를 사용한다. [S6]

```python
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

torch.manual_seed(7)
q, k, v = [
    torch.randn(2, 8, 256, 64, device="cuda", dtype=torch.float16)
    for _ in range(3)
]

with sdpa_kernel(SDPBackend.MATH):
    expected = scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=True, dropout_p=0.0
    )
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    actual = scaled_dot_product_attention(
        q, k, v, is_causal=True, dropout_p=0.0
    )
torch.testing.assert_close(actual.float(), expected, atol=2e-3, rtol=2e-2)
```

강제 선택이 실패하면 PyTorch가 보고하는 지원 조건을 확인한다. 이 코드는 정사각 causal 마스크를 쓰는 prefill 예시다. KV cache가 붙는 decode 검증에서는 실제 토큰 위치에 맞는 참조 범위를 별도로 확인한다. [S6]

## 실습 2. 같은 체크포인트로 서빙 백엔드 비교하기

같은 GQA 체크포인트를 두 백엔드가 모두 지원하는 장비에서 비교한다. 설치한 vLLM 버전의 지원표를 기준으로 백엔드를 지정하고 실제 선택을 시작 로그에서 확인한다. 모델, dtype, KV dtype, page size, 출력 길이, 동시 요청 수를 고정한다. [S8]

- **긴 prefill:** 같은 길이 분포에서 워밍업 후 TTFT와 피크 메모리를 기록한다.
- **다중 요청 decode:** 같은 동시성에서 TPOT, 출력 토큰 처리량과 피크 메모리를 기록한다.
- **정확도 확인:** 고정된 입력에서 가능하면 logits나 logprobs를 허용 오차로 비교한다. 장문 생성이 달라지면 첫 차이가 나온 단계의 마스크와 출력 분포를 확인한다.

[GQA 장](../../../01-llm-models/01-autoregressive-transformers/03-attention-heads-and-kv-representations/README.md)에서 구한 논리적 KV 용량은 이 실습의 기준선이다. [PagedAttention 장](../01-kv-cache-and-paged-attention/README.md)은 물리적 페이지 배치를, [Long-Context Serving 장](../06-long-context-serving-and-kv-memory/README.md)은 sparse indexer의 비용을 추적한다.

## 출처

| ID | 출처 | 발행처 | 날짜 | URL | 사용 이유 |
| --- | --- | --- | --- | --- | --- |
| [S1] | FlashAttention | Dao et al. | 2022 | https://arxiv.org/abs/2205.14135 | 타일링과 HBM 이동, 정확한 dense attention |
| [S2] | FlashAttention-2 | Dao | 2023 | https://arxiv.org/abs/2307.08691 | thread block과 warp의 작업 분할 |
| [S3] | FlashAttention-3 | Shah et al. | 2024 | https://arxiv.org/abs/2407.08608 | Hopper의 비동기 실행과 저정밀 경로 |
| [S4] | FlashAttention-4 | Zadouri et al. | 2026 | https://arxiv.org/abs/2603.05451 | Blackwell의 계산과 메모리 파이프라인 |
| [S5] | FlashInfer | Ye et al. | 2025 | https://arxiv.org/abs/2501.01005 | paged KV와 서빙용 커널 |
| [S6] | scaled_dot_product_attention | PyTorch | 2026-11-20 확인 | https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html | SDPA dispatch, 강제 백엔드 선택, causal 마스크 |
| [S7] | FlexAttention | PyTorch | 2026-11-20 확인 | https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html | score_mod와 BlockMask API |
| [S8] | Attention Backend Feature Support | vLLM | 2026-11-20 확인 | https://docs.vllm.ai/en/stable/design/attention_backends/ | 백엔드 선택, 기능·장치·레이아웃 호환성 |
