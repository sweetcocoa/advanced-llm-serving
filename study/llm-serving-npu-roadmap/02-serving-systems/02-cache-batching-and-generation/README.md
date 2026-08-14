# Cache, Batching, and Generation

단일 요청의 생성 loop와 여러 요청의 동시 실행을 연결한다. AR의 KV cache뿐 아니라 diffusion의 mutable sequence state와 recurrent-depth의 variable loop depth까지 같은 runtime 질문으로 비교한다.

## 챕터 순서
- [KV Cache and PagedAttention](./01-kv-cache-and-paged-attention/README.md)
- [Continuous Batching](./02-continuous-batching/README.md)
- [Prefix Caching](./03-prefix-caching/README.md)
- [Speculative Decoding](./04-speculative-decoding/README.md)
- [Structured Outputs and Tool Calling](./05-structured-outputs-and-tool-calling/README.md)
- [Long-Context Serving and KV Memory](./06-long-context-serving-and-kv-memory/README.md)
- [Iterative Generation Serving](./07-iterative-generation-serving/README.md)
