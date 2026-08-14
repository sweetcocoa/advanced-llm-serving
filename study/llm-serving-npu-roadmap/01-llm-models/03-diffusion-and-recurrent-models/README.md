# Diffusion and Recurrent Models

Autoregressive next-token generation과 다른 두 계보를 다룬다. Diffusion LLM은 token sequence를 반복 복원하고, recurrent-depth Transformer는 hidden state를 반복 계산한다. 공통 반복 구조와 서로 다른 objective를 함께 본다.

## 챕터 순서
- [Diffusion LLM의 계보와 생성 원리](./01-diffusion-llm-lineage-and-principles/README.md)
- [Looped Transformer와 Recurrent Depth](./02-looped-transformers-and-recurrent-depth/README.md)
- [수렴으로 보는 반복 정제](./03-convergence-as-iterative-refinement/README.md)
- [Iterative Model Frontier](./04-iterative-model-frontier/README.md)

## 경계
Diffusion cache, rollback, scheduler와 continuous-depth batching은 서빙 트랙에서, 모델과 runtime의 공정한 비교는 종합 설계 트랙에서 다룬다.
