# Windows ML 퀴즈

## 객관식 1

`ExecutionProviderCatalog`의 역할을 가장 정확히 설명한 것은 무엇인가?

1. ONNX graph를 자동으로 다시 학습한다.
2. 장치와 호환되는 EP를 발견·설치·등록한다.
3. 모든 operator를 반드시 NPU에서 실행한다.
4. BYO EP의 버전 호환성을 자동 보증한다.

## 객관식 2

BYO EP를 먼저 검토할 조건은 무엇인가?

1. 소비자 장치에서 앱 크기를 최소화하고 자동 업데이트를 받고 싶을 때
2. Windows Update가 허용되고 첫 실행 네트워크를 사용할 수 있을 때
3. 오프라인·관리형 장치에서 EP 버전을 엄격히 고정해야 할 때
4. graph coverage를 측정하지 않아도 될 때

## 객관식 3

catalog와 BYO EP의 관계를 올바르게 설명한 것은 무엇인가?

1. 하나를 쓰면 다른 하나는 같은 앱에서 사용할 수 없다.
2. catalog를 우선하고 실패 시 bundled EP로 넘어갈 수 있다.
3. BYO EP는 Windows ML의 ONNX Runtime과 무관하다.
4. catalog는 EP를 앱 패키지에 정적으로 포함한다.

## 짧은 서술형 1

첫 실행 지연을 `EP acquire`, `EP register`, `model load`, `session`, `infer`로 나눠야 하는 이유를 설명하라.

## 짧은 서술형 2

EP catalog에서 NPU용 provider를 찾았다는 사실만으로 성능을 보장할 수 없는 이유를 설명하라.

## 심화 설명형

소비자용 앱과 폐쇄망 기업용 앱을 모두 지원해야 한다. catalog 우선·BYO EP fallback 정책을 설계하고, 다음 항목을 포함해 설명하라.

- EP 획득과 등록
- 버전 및 호환성 책임
- 첫 실행 실패 처리
- 실제 선택된 EP와 graph fallback의 관측

## 정답 및 해설

- 객관식 1 정답: 2. catalog는 호환 EP의 발견·다운로드·설치·등록을 담당하며, 실제 graph 실행은 등록된 EP와 ONNX Runtime session의 일이다. [S2]
- 객관식 2 정답: 3. Windows Update가 제한되거나 오프라인이고 정확한 버전 통제가 필요하면 앱이 EP를 포함하는 BYO 경로가 적합하다. [S4]
- 객관식 3 정답: 2. 공식 문서는 catalog를 선호 경로로 쓰고 실패 시 같은 target의 bundled EP로 넘어가는 혼합 전략을 허용한다. [S4]
- 짧은 서술형 1 예시: catalog가 EP를 내려받아야 하는 첫 실행은 수 초에서 수 분이 걸릴 수 있다. 이 비용을 모델 inference와 분리해야 네트워크·등록 문제를 모델 성능 문제로 오진하지 않는다. [S2]
- 짧은 서술형 2 예시: catalog는 EP를 준비하지만 모델의 operator·shape·precision이 그 EP에서 얼마나 실행될지는 결정하지 않는다. session 생성 결과와 graph partition, CPU fallback을 별도로 관측해야 한다. [S3]
- 심화 설명형 해설 포인트: 일반 소비자 장치에서는 catalog로 인증 EP를 설치·등록하고, 오프라인·정책 제한 환경에서는 bundled EP를 등록한다. catalog 조합은 Microsoft가 검증하지만 BYO 조합은 앱 팀이 Windows ML·ORT·EP 호환성을 시험한다. 첫 실행에는 준비 상태와 다운로드 실패를 다루고, 운영 시 실제 EP 이름·버전·fallback·latency를 함께 남겨야 한다. [S2] [S4]
