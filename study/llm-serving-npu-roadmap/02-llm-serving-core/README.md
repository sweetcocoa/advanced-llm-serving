# LLM Serving 핵심

서빙 엔진이 요청을 받아 토큰을 생성하기까지 필요한 cache, batching, quantization, stack 구성을 본다.

## 이 모듈을 마치면 설명할 수 있어야 하는 것
- continuous batching과 PagedAttention의 역할을 연결해서 설명할 수 있다.
- prefill과 decode에서 자원이 다르게 소모되는 이유를 설명할 수 있다.
- vLLM, TensorRT-LLM, ONNX Runtime/TGI의 포지션 차이를 설명할 수 있다.

## 챕터 순서
- [Prefill vs Decode](./01-prefill-vs-decode/README.md): 한 요청이 왜 두 개의 다른 workload로 나뉘는지 이해한다.
- [KV Cache and PagedAttention](./02-kv-cache-and-paged-attention/README.md): KV cache가 왜 필수인지, 그리고 PagedAttention이 fragmentation을 어떻게 낮추는지 본다.
- [Continuous Batching](./03-continuous-batching/README.md): continuous batching이 throughput과 latency를 동시에 개선하는 이유를 본다.
- [Quantization Basics](./04-quantization-basics/README.md): INT8, INT4, FP8 양자화가 latency와 품질에 주는 영향을 정리한다.
- [Serving Stack Overview](./05-serving-stack-overview/README.md): API layer부터 runtime, kernel, observability까지 serving stack의 계층을 정리한다.

## 선행 관계
- 같은 모듈에서는 위 순서를 따르는 것이 좋다.
- 다음 모듈로 넘어가기 전에 각 챕터 퀴즈를 먼저 풀어 본다.
