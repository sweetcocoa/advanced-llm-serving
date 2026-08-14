# Autoregressive Transformer

Transformer block, causal mask, next-token objective와 autoregressive generation을 모델 관점에서 정리한다. prefill/decode의 성능 차이는 여기서 결론 내리지 않고 서빙 트랙으로 넘긴다.

## 챕터 순서
- [Transformer and Autoregressive LM](./01-transformer-and-autoregressive-lm/README.md): causal factorization과 Transformer가 serving에 넘기는 모델 contract를 배운다.

## 경계
KV cache, TTFT, TPOT과 batching은 `02-serving-systems/01-inference-foundations`에서 다룬다.
