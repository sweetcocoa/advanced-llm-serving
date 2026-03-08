# NPU 소프트웨어 스택

NPU를 칩이 아니라 graph lowering, quantization, runtime partitioning 관점으로 이해한다.

## 이 모듈을 마치면 설명할 수 있어야 하는 것
- NPU 친화적인 graph 형태와 비친화적 graph 형태를 구분할 수 있다.
- QDQ/QOperator 관점에서 양자화 흐름을 설명할 수 있다.
- execution provider가 offload 경계를 어떻게 만드는지 설명할 수 있다.
- 대표 NPU 벤더들의 공통점과 차이점을 실제 배포 사례로 비교할 수 있다.

## 챕터 순서
- [NPU Architecture Basics](./01-npu-architecture-basics/README.md): NPU의 SRAM, DMA, operator fusion 중심 사고방식을 만든다.
- [ONNX Graph and Lowering](./02-onnx-graph-and-lowering/README.md): 모델이 ONNX graph를 거쳐 backend-specific lowering으로 내려가는 흐름을 본다.
- [NPU-Friendly Quantization](./03-npu-friendly-quantization/README.md): QDQ/QOperator, calibration, import 가능한 quantized artifact 관점으로 NPU-friendly quantization을 본다.
- [Runtime and Execution Provider](./04-runtime-and-execution-provider/README.md): execution provider와 runtime이 session creation, device target, artifact reuse를 어떻게 결정하는지 정리한다.
- [Profiling, Debugging, and Bottlenecks](./05-profiling-debugging-and-bottlenecks/README.md): NPU workload에서 trace와 profiler로 실제 병목을 좁혀 가는 방법을 본다.
- [Vendor Case Studies and Comparison](./06-vendor-case-studies-and-comparison/README.md): 대표 NPU 벤더들의 실제 배포 경로를 비교하면서 공통 개념과 차이점을 정리한다.

## 선행 관계
- 같은 모듈에서는 위 순서를 따르는 것이 좋다.
- 다음 모듈로 넘어가기 전에 각 챕터 퀴즈를 먼저 풀어 본다.
