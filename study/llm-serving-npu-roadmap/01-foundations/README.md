# 기초와 성능 직관

Transformer inference의 기본 연산과 메모리 병목을 이해해 이후 serving/NPU 문서를 읽을 수 있는 공통 언어를 만든다.

## 이 모듈을 마치면 설명할 수 있어야 하는 것
- prefill과 decode를 분리해 설명할 수 있다.
- memory bandwidth와 arithmetic intensity가 latency를 어떻게 제한하는지 설명할 수 있다.
- GPU와 NPU를 같은 추론 시스템의 다른 실행 장치로 비교할 수 있다.

## 챕터 순서
- [Transformer Inference](./01-transformer-inference/README.md): Transformer 추론 경로를 prefill, decode, attention, projection 관점에서 읽는다.
- [Memory, Bandwidth, Latency](./02-memory-bandwidth-latency/README.md): latency가 FLOPs보다 memory movement에 더 민감해지는 이유를 정리한다.
- [GPU vs NPU Mental Model](./03-gpu-vs-npu-mental-model/README.md): GPU와 NPU를 연산 장치가 아니라 실행 모델 차이로 비교한다.
- [LLM Inference Metrics](./04-llm-inference-metrics/README.md): TTFT, TPOT, throughput, utilization을 같은 시스템 지표 체계로 묶는다.

## 선행 관계
- 같은 모듈에서는 위 순서를 따르는 것이 좋다.
- 다음 모듈로 넘어가기 전에 각 챕터 퀴즈를 먼저 풀어 본다.
