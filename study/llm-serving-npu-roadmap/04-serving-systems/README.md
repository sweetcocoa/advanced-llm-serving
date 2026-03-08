# Serving 시스템 설계

한 개 기능이 아니라 시스템 전체 관점에서 스케줄링, 병렬화, 관측 가능성, 프레임워크 비교를 본다.

## 이 모듈을 마치면 설명할 수 있어야 하는 것
- scheduler와 admission control을 queueing 관점에서 설명할 수 있다.
- tensor/pipeline/expert parallelism의 목적 차이를 설명할 수 있다.
- benchmark와 observability가 같은 문제가 아닌 이유를 설명할 수 있다.

## 챕터 순서
- [Scheduling and Admission Control](./01-scheduling-and-admission-control/README.md): request scheduler가 어떤 workload를 먼저 처리할지 결정하는 기준을 본다.
- [Model Parallelism: TP, PP, EP](./02-model-parallelism-tp-pp-ep/README.md): tensor parallelism, pipeline parallelism, expert parallelism을 추론 관점으로 비교한다.
- [Benchmarking and Observability](./03-benchmarking-and-observability/README.md): 성능 측정과 운영 관찰을 구분하면서 연결한다.
- [vLLM, TensorRT-LLM, Neuron Comparison](./04-vllm-tensorrt-llm-neuron-comparison/README.md): 대표 서빙 프레임워크의 강점과 제약을 비교한다.

## 선행 관계
- 같은 모듈에서는 위 순서를 따르는 것이 좋다.
- 다음 모듈로 넘어가기 전에 각 챕터 퀴즈를 먼저 풀어 본다.
