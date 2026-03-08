# 런타임과 컴파일러

실제 제품 스택과 MLIR 계열 컴파일러를 연결해 온디바이스 추론과 backend 특화를 이해한다.

## 이 모듈을 마치면 설명할 수 있어야 하는 것
- ONNX Runtime QNN과 Windows ML의 역할 차이를 설명할 수 있다.
- Ryzen AI, OpenVINO, Intel NPU stack의 포지션을 비교할 수 있다.
- MLIR, StableHLO, IREE가 lowering pipeline에서 맡는 역할을 설명할 수 있다.

## 챕터 순서
- [ONNX Runtime QNN](./01-onnx-runtime-qnn/README.md): QNN EP가 graph를 Qualcomm backend로 넘기는 구조를 본다.
- [Windows ML](./02-windows-ml/README.md): Windows ML이 ONNX Runtime 기반으로 on-device AI 경험을 묶는 방식을 본다.
- [Ryzen AI OGA](./03-ryzen-ai-oga/README.md): AMD의 hybrid OGA 흐름과 NPU/GPU 협업 시나리오를 정리한다.
- [OpenVINO NPU](./04-openvino-npu/README.md): OpenVINO에서 NPU target을 사용하는 경로와 한계를 본다.
- [MLIR, StableHLO, IREE](./05-mlir-stablehlo-iree/README.md): 컴파일러 중간 표현이 왜 필요한지와 lowering이 어떻게 연결되는지 본다.

## 선행 관계
- 같은 모듈에서는 위 순서를 따르는 것이 좋다.
- 다음 모듈로 넘어가기 전에 각 챕터 퀴즈를 먼저 풀어 본다.
