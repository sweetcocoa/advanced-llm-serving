# ONNX Graph and Lowering 퀴즈

## 객관식 1
ONNX graph를 NPU 배포의 "중간 표현"으로 이해할 때 가장 적절한 설명은 무엇인가?
1. ONNX로 export되면 backend별 차이는 사실상 사라진다.
2. ONNX는 공통 graph를 제공하지만, 실제 실행 단위는 partition과 lowering을 거치며 backend마다 달라진다.
3. ONNX는 layout 변환 정보를 완전히 고정하므로 compiler가 추가 결정을 할 필요가 없다.
4. ONNX는 unsupported op를 자동으로 모두 제거한다.

## 객관식 2
QNN Execution Provider 같은 execution provider를 볼 때, supported op 개수만 세는 것이 부족한 가장 큰 이유는 무엇인가?
1. execution provider는 오직 CPU fallback만 지원하기 때문이다.
2. graph가 몇 개의 subgraph로 분할되는지에 따라 boundary handoff 비용이 크게 달라지기 때문이다.
3. execution provider는 lowering과 무관한 로깅 계층이기 때문이다.
4. supported op 수가 많으면 항상 layout 변환 비용이 0이 되기 때문이다.

## 객관식 3
MLIR/IREE 관점에서 lowering을 가장 잘 설명한 것은 무엇인가?
1. ONNX op 이름을 backend op 이름으로 한 번 바꾸는 단일 치환 과정
2. unsupported op를 삭제하는 정리 단계
3. 여러 수준의 IR를 거치며 legalize, rewrite, layout 결정, bufferization, codegen을 수행하는 과정
4. runtime이 끝난 뒤 결과를 시각화하는 단계

## 짧은 서술형 1
`Benefit_offload = T_host_only - (T_npu_segment + N_boundary * T_handoff + T_layout)` 식이 실무에서 무엇을 판단하는 데 쓰이는지 3문장 이내로 설명하라.

## 짧은 서술형 2
layout 변환이 compiler 내부 구현 세부가 아니라 end-to-end latency 문제라고 말하는 이유를 설명하라.

## 심화 설명형 1
다음 두 경로를 비교해 설명하라.
- ONNX Runtime + QNN EP 경로
- MLIR/IREE 중심 lowering 경로

각 경로에서 실무자가 가장 먼저 확인해야 할 항목을 쓰고, 왜 질문 순서가 달라지는지 서술하라.

## 심화 설명형 2
transformer 계열 모델을 예로 들어, 다음 네 요소를 하나의 흐름으로 연결해 설명하라.
- ONNX graph
- graph partition
- layout 변환
- unsupported op fallback

답안에는 "왜 ONNX export 성공이 배포 성공과 같지 않은가"를 반드시 포함하라.

## 정답 및 해설
- 객관식 1 정답: 2. ONNX는 공통 표현을 제공하지만 실제 실행 경로는 execution provider의 partition과 backend-specific lowering 결과에 따라 달라진다.
- 객관식 2 정답: 2. supported op 수가 많아 보여도 graph가 잘게 쪼개지면 boundary handoff, 복사, 동기화, layout 재배치 비용이 커질 수 있다.
- 객관식 3 정답: 3. MLIR/IREE 문맥의 lowering은 다단계 IR 변환이며, legalize와 rewrite뿐 아니라 layout 결정과 bufferization까지 포함하는 과정이다.
- 짧은 서술형 1 예시: 이 식은 특정 segment를 NPU에 내리는 것이 실제로 이득인지 판단할 때 쓴다. NPU 계산 시간이 짧아도 boundary handoff와 layout 변환 비용이 크면 전체 이득이 사라질 수 있다. 따라서 offload 여부는 compute만이 아니라 경계 비용까지 포함해 봐야 한다.
- 짧은 서술형 2 예시: layout 변환은 tensor 배치를 다시 맞추기 위해 추가 복사나 재배치를 유발할 수 있다. 이 비용이 partition 경계와 겹치면 NPU compute 절감보다 더 큰 지연을 만들 수 있다. 그래서 layout은 compiler 내부 세부가 아니라 end-to-end latency를 바꾸는 성능 변수다.
- 심화 설명형 1 해설 포인트: QNN EP 경로에서는 먼저 partition 경계, provider assignment, fallback 위치를 본다. MLIR/IREE 경로에서는 dialect conversion, legalize 단계, layout 및 bufferization이 어디서 결정되는지 먼저 본다. 질문 순서가 다른 이유는 전자는 runtime 위의 subgraph offload 품질이 핵심이고, 후자는 compiler pipeline 자체가 실행 단위를 규정하기 때문이다.
- 심화 설명형 2 해설 포인트: ONNX graph는 공통 출발점이지만, execution provider가 지원 구간만 partition으로 묶고 나머지는 fallback으로 남긴다. 이 과정에서 layout 변환이 끼면 segment 이득이 줄어들 수 있다. unsupported op 하나가 큰 subgraph를 끊으면 handoff가 늘어 배포 품질이 나빠질 수 있으므로, ONNX export 성공만으로는 실제 NPU lowering 성공을 보장하지 못한다.
