# 반복 생성 모델: Diffusion LLM과 Recurrent Depth

이 모듈은 autoregressive decoder-only Transformer 다음에 나타난 두 개의 새로운 LLM 계보를 독립적으로 다룬다. Diffusion LLM은 token sequence를 반복적으로 복원하고, looped 또는 recurrent-depth Transformer는 hidden state를 반복적으로 계산한다. 두 계열은 출발점과 학습 목표가 다르지만 `공유 전이 함수에 의한 iterative refinement`라는 계산 구조로 수렴하고 있다.

이 주제를 단일 트렌드로 다루지 않는 이유는 serving 계약까지 달라지기 때문이다. Diffusion LLM은 block generation, bidirectional attention, rollback, custom sampler를 요구한다. Recurrent-depth LM은 token마다 다른 loop depth와 hidden-state exit를 처리해야 한다. 2026년에는 vLLM의 native diffusion model 지원, Ouro inference 지원, dLLM 전용 cache와 scheduler, continuous depth batching까지 등장해 모델 연구와 serving 연구가 직접 만나는 단계에 들어섰다.

## 이 모듈을 마치면 설명할 수 있어야 하는 것
- D3PM에서 LLaDA, Dream, DiffusionGemma, 2026년 8월 연구까지 이어지는 diffusion LLM 계보를 설명할 수 있다.
- Universal Transformer에서 Huginn, Ouro, elastic/adaptive-depth 모델로 이어지는 recurrent-depth 계보를 설명할 수 있다.
- 두 계열을 같은 반복 상태 전이 식으로 묶되, 왜 동일한 확률 모델은 아닌지 반박할 수 있다.
- Fast-dLLM, dKV-Cache, Sangam, vLLM ModelState가 AR serving contract를 어떻게 바꾸는지 설명할 수 있다.
- Ouro와 Huginn의 adaptive depth가 일반 continuous batching을 왜 깨뜨리는지 설명할 수 있다.
- 논문 benchmark와 실제 엔진 지원, production readiness를 구분할 수 있다.

## 챕터 순서
- [Diffusion LLM의 계보와 생성 원리](./01-diffusion-llm-lineage-and-principles/README.md): 연속 diffusion에서 discrete/masked diffusion, 대규모 dLLM으로 이어지는 흐름을 배운다.
- [Looped Transformer와 Recurrent Depth](./02-looped-transformers-and-recurrent-depth/README.md): weight tying, latent reasoning, adaptive depth의 계보와 한계를 배운다.
- [두 계보가 만나는 Iterative Refinement](./03-convergence-as-iterative-refinement/README.md): 공통 상태 전이, 차이점, sampler 이동, 결합 모델을 통해 "비슷하다"의 정확한 범위를 정한다.
- [현재 Serving 시스템과 Runtime](./04-serving-systems-and-runtime/README.md): vLLM native 지원, cache, batching, scheduling, 배포 성숙도를 실제 구현 기준으로 비교한다.
- [2026 Frontier와 평가 방법](./05-2026-frontier-and-evaluation/README.md): 2026년 8월 최신 논문, 반례, benchmark 설계를 정리하고 다음 연구 질문을 도출한다.

## 선행 관계
- `01-foundations/01-transformer-inference`, `02-llm-serving-core/01-prefill-vs-decode`, `02-llm-serving-core/02-kv-cache-and-paged-attention`을 먼저 읽는 편이 좋다.
- 1, 2 챕터는 서로 독립적으로 읽을 수 있다.
- 3 챕터는 1, 2 챕터의 개념을 결합한다.
- 4 챕터는 모델 원리보다 serving 구현과 운영 판단에 초점을 둔다.
- 5 챕터는 2026년 8월 12일 기준 최신 상태를 다루므로 출처 날짜와 저자 보고 범위를 함께 확인한다.
