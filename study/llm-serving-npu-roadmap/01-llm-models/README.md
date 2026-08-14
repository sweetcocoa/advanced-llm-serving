# LLM 모델 이론

이 트랙은 모델이 어떤 확률 분해와 상태 전이를 학습하고, 어떤 순서로 출력을 생성하는지 다룬다. 하드웨어 배치나 scheduler보다 architecture, objective, causality, routing, iterative refinement를 먼저 설명한다.

## 이 트랙을 마치면 설명할 수 있어야 하는 것
- autoregressive LM의 causal factorization과 Transformer block의 역할을 설명할 수 있다.
- MoE의 conditional computation을 expert-parallel runtime과 구분할 수 있다.
- Diffusion LLM과 recurrent-depth Transformer의 계보와 차이를 설명할 수 있다.
- 모델이 serving system에 넘기는 상태, 순서, 종료 조건의 contract를 적을 수 있다.

## 섹션 순서
- [Autoregressive Transformer](./01-autoregressive-transformers/README.md)
- [Mixture of Experts](./02-mixture-of-experts/README.md)
- [Diffusion and Recurrent Models](./03-diffusion-and-recurrent-models/README.md)

## 다음 트랙과의 경계
모델 구조가 정의한 contract를 실제 cache, batching, parallelism, runtime으로 구현하는 문제는 `02-serving-systems`에서 다룬다.
