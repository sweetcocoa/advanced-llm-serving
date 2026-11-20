# Autoregressive Transformer

Transformer block, causal mask, next-token objective와 autoregressive generation을 모델 관점에서 정리한다. prefill/decode의 성능 차이는 여기서 결론 내리지 않고 서빙 트랙으로 넘긴다.

## 챕터 순서
- [Transformer and Autoregressive LM](./01-transformer-and-autoregressive-lm/README.md): causal factorization과 Transformer가 serving에 넘기는 모델 contract를 배운다.
- [긴 문맥을 읽는 방식: DSA, Lightning Indexer, KPool](./02-long-context-sparse-attention/README.md): 토큰 선택, 선택용 key pooling, 본 어텐션의 KV 압축을 비교한다.
- [어텐션 헤드와 KV 표현: MHA, MQA, GQA, MLA](./03-attention-heads-and-kv-representations/README.md): head 공유와 잠재 표현 압축의 차이를 계산한다.
- [Linear Attention과 Hybrid 모델](./04-linear-and-hybrid-attention/README.md): 과거를 요약하는 반복 상태와 토큰별 KV를 비교한다.

## 경계
KV cache, TTFT, TPOT과 batching은 `02-serving-systems/01-inference-foundations`에서 다룬다.
