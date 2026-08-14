# ONNX Runtime QNN 퀴즈

## 객관식 1
Snapdragon PC에서 `QNNExecutionProvider` 로그는 보이지만 latency 개선이 거의 없다. README 기준으로 가장 먼저 확인할 항목은 무엇인가?
1. 앱 UI 프레임 드롭 여부
2. QNN 파티션의 수와 각 파티션의 크기
3. 모델 파일의 압축률
4. CPU의 최대 클럭

## 객관식 2
QNN EP에서 첫 실행은 느리지만 반복 실행은 빨라지는 현상을 가장 잘 설명하는 것은 무엇인가?
1. ONNX 파일 크기가 자동으로 줄어들기 때문이다.
2. Windows ML이 백그라운드에서 자동 최적화를 대신하기 때문이다.
3. session/context 준비 비용과 steady-state 실행 비용을 구분해야 하기 때문이다.
4. CPU fallback이 반복 실행에서 항상 사라지기 때문이다.

## 객관식 3
다음 중 README의 비교 섹션에서 QNN EP 대신 다른 선택지를 우선 보게 만드는 결정적 질문으로 맞는 것은 무엇인가?
1. "모델 파일 이름이 ONNX인가?"
2. "Qualcomm NPU를 직접 겨냥하는가?"
3. "Python 대신 C++를 쓰는가?"
4. "앱 아이콘이 Windows 스타일인가?"

## 짧은 서술형 1
QNN EP에서 `가중 QNN 오프로드 비율`이 단순 노드 수 비율보다 유용한 이유를 2~3문장으로 설명하라.

## 짧은 서술형 2
모델을 QNN-friendly하게 수정하는 선택이 왜 portability 비용으로 이어질 수 있는지 설명하라.

## 심화 설명형 1
로그에 NPU 사용이 찍혔지만 latency가 개선되지 않았다. README의 진단 순서를 따라 `partition 수/크기`, `boundary copy`, `CPU fallback`, `init vs steady-state`를 어떤 순서로 확인할지 서술하라. 각 단계에서 왜 그 순서인지도 함께 설명하라.

## 심화 설명형 2
다음 두 상황을 비교해 QNN EP를 계속 밀어붙일지 판단하라.

1. Snapdragon PC 상주형 음성 앱: 첫 실행은 900ms지만 이후 반복 호출은 120ms.
2. Snapdragon PC 단발성 유틸리티: 첫 실행 900ms, 반복 호출 기회가 거의 없음.

답안에는 `cold-start`, `steady-state`, `context 준비`, `제품 사용 패턴`을 모두 포함하라.

## 정답 및 해설
- 객관식 1 정답: 2. QNN EP는 "사용했다/안 했다"보다 어떤 크기의 subgraph가 QNN으로 갔는지, 그리고 그 조각 수가 몇 개인지가 더 중요하다.
- 객관식 2 정답: 3. README는 QNN EP에서 session/context 준비 비용과 반복 실행 비용을 분리해 읽어야 한다고 설명한다.
- 객관식 3 정답: 2. QNN EP의 가장 직접적인 선택 질문은 "Qualcomm NPU를 직접 겨냥하는가?"다. 다른 선택지는 각기 다른 질문에서 출발한다.
- 짧은 서술형 1 예시: 단순 노드 수는 비싼 연산과 가벼운 연산을 구분하지 못한다. 가중 오프로드 비율은 실제 비용이 큰 블록이 QNN으로 갔는지 판단하게 해 주므로, fragmentation이 심한 모델의 체감 성능을 더 잘 설명한다.
- 짧은 서술형 2 예시: QNN 지원 범위에 맞추기 위해 모델 표현이나 양자화 방식을 바꾸면 Qualcomm 경로에서는 큰 파티션을 얻을 수 있다. 하지만 같은 ONNX 모델을 Windows ML, OpenVINO, AMD 경로에도 유지해야 한다면 별도 검증과 분기 관리가 필요해져 portability 비용이 생긴다.
- 심화 설명형 1 예시: 먼저 QNN 파티션 수와 크기를 본다. 큰 파티션이 아니라 작은 파티션 여러 개라면 그 자체가 가장 큰 경고다. 다음으로 각 경계에서 boundary copy나 layout 변환이 있는지 본다. 그 뒤 CPU fallback이 어느 구간에서 생기는지 확인해 실제 병목이 unsupported op인지 구분한다. 마지막으로 첫 실행과 반복 실행을 분리해 기록한다. partition 구조가 이미 나쁘면 cold-start를 줄여도 효과가 제한적이기 때문에 이 순서가 맞다.
- 심화 설명형 2 예시: 상주형 음성 앱은 context 준비 비용을 한 번 치른 뒤 steady-state 이득을 오래 재사용하므로 QNN EP를 유지할 가치가 있다. 반대로 단발성 유틸리티는 cold-start가 거의 전체 UX를 결정하므로 같은 숫자라도 불리하다. 즉 판단 기준은 절대 숫자 하나가 아니라 제품 사용 패턴과 context 준비 비용의 amortization 가능성이다.
