# Windows ML 퀴즈

## 객관식 1
회의 요약 기능을 Windows 앱에 넣으려는 팀이 있다. 우선순위는 "Windows 배포 일관성"과 "로컬 실행"이고, 아직 특정 NPU vendor에 강하게 묶이고 싶지는 않다. 이때 가장 자연스러운 첫 선택은 무엇인가?

1. 처음부터 장치별 execution provider를 직접 붙이며 vendor별 튜닝부터 시작한다.
2. Windows ML을 출발점으로 삼고, ONNX 기반 실행 경로와 fallback을 먼저 점검한다.
3. ONNX export만 되면 성능은 자동으로 해결된다고 보고 별도 검증 없이 출시한다.
4. NPU가 있는 PC만 지원한다고 공지하고 latency 측정은 생략한다.

## 객관식 2
"이 기능은 NPU를 쓴다"는 보고를 받았는데도 사용자는 앱이 느리다고 말한다. 이때 가장 먼저 확인해야 할 것은 무엇인가?

1. 그래프의 주요 node가 실제로 가속 장치에서 실행되는지와 fallback 구간이 어디인지
2. 마케팅 자료에 Copilot+ PC 문구가 들어갔는지
3. 모델 파일 확장자가 `.onnx`인지 아닌지만
4. GPU와 CPU의 이론 TOPS 수치

## 객관식 3
다음 중 Windows ML 챕터의 tradeoff를 가장 정확히 요약한 것은 무엇인가?

1. 정확도와 데이터셋 크기의 tradeoff
2. 클라우드 비용과 프롬프트 길이의 tradeoff
3. OS 통합성과 세밀한 튜닝의 tradeoff
4. 양자화와 파인튜닝의 tradeoff

## 짧은 서술형 1
다음 식이 Windows ML 성능 분석에서 왜 중요한지 설명하라.

$$
T_{\mathrm{user}} = T_{\mathrm{load}} + T_{\mathrm{bind}} + T_{\mathrm{dispatch}} + T_{\mathrm{fallback}}
$$

## 짧은 서술형 2
ONNX export 성공과 "실제로 장치 친화적으로 잘 실행된다"는 판단이 왜 다른지, Windows ML 관점에서 2~3문장으로 설명하라.

## 심화 설명형 1
카메라 보조 캡션 기능을 예로 들어, 왜 짧은 burst성 워크로드에서는 단순한 peak throughput 수치보다 `load`, `bind`, `dispatch` 비용이 더 중요할 수 있는지 설명하라. 답변에는 Windows ML을 먼저 써 볼 이유와, 어느 시점에 직접 ONNX Runtime + execution provider 경로를 검토할지를 함께 포함하라.

## 심화 설명형 2
Windows ML과 직접 ONNX Runtime + execution provider 경로를 비교하라. 다음 세 요소를 반드시 포함하라.

1. 어떤 팀에게 Windows ML이 더 자연스러운 출발점인지
2. 어떤 조건에서 직접 provider 수준 튜닝이 필요해지는지
3. "NPU 사용"이라는 문장이 왜 의사결정 근거로 불충분한지

## 정답 및 해설
- 객관식 1 정답: 2. Windows ML은 Windows 앱 관점의 on-device AI 통합 경로로 이해해야 하며, ONNX 기반 실행 흐름을 통해 빠른 제품 통합에 유리하다. 다만 이후에는 fallback과 graph coverage를 반드시 봐야 한다. [S1] [S2] [S3]
- 객관식 2 정답: 1. 성능 문제는 장치 존재 여부보다 실제 graph 실행 비율과 fallback 위치를 봐야 드러난다. QNN execution provider 문맥에서도 핵심은 어떤 graph가 backend로 내려가는가이다. [S4]
- 객관식 3 정답: 3. 이 챕터의 핵심 tradeoff는 Windows 차원의 통합성과 vendor/장치 수준 세밀 튜닝 사이의 선택이다. [S1] [S4]
- 짧은 서술형 1 예시: 이 식은 사용자가 느끼는 지연이 순수 추론 시간 하나가 아니라 모델 로드, 입력 바인딩, 장치 dispatch, fallback 비용의 합으로 결정된다는 점을 보여 준다. 그래서 회의 요약이나 카메라 보조 기능처럼 짧고 자주 호출되는 앱에서는 `T_load`와 `T_bind`가 특히 중요하다. [S1] [S2]
- 짧은 서술형 2 예시: ONNX는 배포용 공용 계약이지만, export 성공이 곧바로 높은 장치 활용률을 보장하지는 않는다. 실제 실행에서는 operator 지원 범위와 graph partition에 따라 일부가 CPU로 fallback될 수 있으므로, Windows ML에서도 ONNX graph의 실효 offload 비율을 따로 봐야 한다. [S3] [S4]
- 심화 설명형 1 해설 포인트: burst성 기능은 최고 처리량보다 시작 반응성이 중요하므로 `T_load`, `T_bind`, `T_dispatch`를 먼저 분석해야 한다. Windows 통합과 로컬 배포가 우선이면 Windows ML이 자연스러운 출발점이지만, graph coverage가 낮고 fallback이 심하면 직접 ONNX Runtime + execution provider 경로로 내려가 저수준 튜닝을 검토해야 한다. [S1] [S2] [S4]
- 심화 설명형 2 해설 포인트: Windows ML은 Windows 앱 통합과 배포 일관성에 강점이 있고, 직접 ONNX Runtime + provider 경로는 backend 중심 세밀 제어에 강점이 있다. 후자가 항상 빠른 것은 아니며, "NPU 사용"만으로는 충분하지 않다. 실제로는 ONNX graph 중 얼마나 많은 부분이 원하는 장치에서 실행되는지와 fallback 비용을 함께 판단해야 한다. [S1] [S3] [S4]
