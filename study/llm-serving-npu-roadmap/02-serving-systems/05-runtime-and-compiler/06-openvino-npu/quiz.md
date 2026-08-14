# OpenVINO NPU 퀴즈

## 객관식 1

긴 prompt가 OpenVINO NPU에서 실패했을 때 가장 적절한 첫 진단은 무엇인가?

1. NPU는 가변 prompt를 지원하지 않으므로 CPU로 바꾼다
2. 일반 Runtime graph인지 GenAI LLM pipeline인지 구분하고 context와 prompt mode 설정을 확인한다
3. 모델 파일을 무조건 ONNX로 다시 변환한다
4. `AUTO`를 쓰면 길이 제한이 없어지므로 실행 모드만 바꾼다

## 객관식 2

어떤 팀이 같은 모델을 같은 장비에서 반복 실행했더니 첫 요청만 유난히 느리고, 이후 요청은 훨씬 안정적이었다. 이때 가장 먼저 떠올려야 할 해석은 무엇인가?

1. NPU는 첫 요청에서만 CPU로 동작한다
2. FEIL/FIL 또는 model caching과 연결된 first-run 비용을 steady-state와 분리해야 한다
3. ONNX 형식은 두 번째 요청부터만 최적화된다
4. `AUTO`를 쓰면 첫 요청 지연이 원칙적으로 사라진다

## 객관식 3

다음 중 OpenVINO에서 `"NPU"` 직접 지정과 `AUTO`/`HETERO`의 차이를 가장 정확히 설명한 것은 무엇인가?

1. `"NPU"`는 장치 고정이고, `AUTO`는 장치 선택 정책이며, `HETERO`는 장치 분할 실행이다
2. `"NPU"`와 `AUTO`는 완전한 동의어이고, `HETERO`만 별도 기능이다
3. `AUTO`는 항상 NPU를 우선 선택하고, `HETERO`는 항상 CPU를 우선 선택한다
4. `"NPU"`는 ONNX 전용이고, `AUTO`와 `HETERO`는 IR 전용이다

## 짧은 서술형 1

왜 OpenVINO NPU 챕터에서 `supported data types`와 `feature support`를 별도 확인 대상으로 두는지 2~3문장으로 설명하라.

## 짧은 서술형 2

Intel PC용 앱 팀이 OpenVINO NPU, Intel NPU Acceleration Library, Windows ML 중 무엇을 먼저 검토해야 하는지 2~3문장으로 설명하라.

답변 조건:

- `장치 고정` 또는 `AUTO/HETERO` 중 하나를 포함할 것
- `OS 통합`이라는 표현을 포함할 것

## 심화 설명형 1

로컬 챗 UI 팀이 NPU에서 긴 프롬프트만 실패한다고 한다. 본문에서 제시한 순서대로 디버깅 질문을 5단계 이상 쓰라. 답변에는 `LLMPipeline`, `MAX_PROMPT_LEN`, `PREFILL_HINT`, `supported data types`, `CPU/GPU 대안`이 모두 포함되어야 한다.

## 심화 설명형 2

문서 임베딩 서비스를 운영하는 팀이 "첫 배포 직후는 느린데 재실행은 빠르니, NPU 연산 자체가 불안정하다"라고 말한다. 이 주장에 반박하는 설명을 쓰라. 답변에는 `FEIL/FIL`, `model caching`, `first-run`, `steady-state`를 모두 포함하라.

## 정답 및 해설

- 객관식 1 정답: 2. 일반 graph의 shape 제약과 GenAI LLM pipeline의 가변 prompt 지원은 같은 문제가 아니다. OpenVINO 2025.3 이후 NPU LLM pipeline은 `DYNAMIC` prefill이 기본이므로 API 층, 모델 준비, context budget과 prompt mode부터 확인한다 [S1][S5].
- 객관식 2 정답: 2. 첫 요청만 느리고 이후가 안정적이면 FEIL/FIL이나 model caching 같은 first-run 비용을 steady-state와 분리해 읽어야 한다 [S1].
- 객관식 3 정답: 1. `"NPU"`는 장치 고정, `AUTO`는 장치 선택 정책, `HETERO`는 장치 분할 실행이므로 같은 실험으로 취급하면 안 된다 [S1].
- 짧은 서술형 1 예시: OpenVINO NPU는 모든 dtype과 feature를 무제한으로 받는 경로가 아니다. 그래서 모델이 요구하는 dtype이나 기능이 NPU supported data types, feature support 범위를 벗어나면 `"NPU"`를 줬더라도 CPU/GPU 경로를 다시 검토해야 한다 [S1].
- 짧은 서술형 2 예시: OpenVINO NPU를 먼저 볼 상황은 Intel PC에서 NPU를 장치 고정으로 검증하거나 `AUTO/HETERO`까지 같은 toolkit에서 비교하고 싶을 때다 [S1][합성]. Intel NPU Acceleration Library는 Intel NPU 활용을 직접 다루는 별도 라이브러리 경로이고 [S2], Windows ML은 OS 통합 경로이므로 Windows 앱 통합이 앞선 질문이면 그쪽이 먼저다 [S4][합성].
- 심화 설명형 1 해설 포인트: 먼저 일반 Runtime graph인지 `LLMPipeline`인지 구분한다 [S1][S5]. LLM pipeline이면 NPU용 모델 준비와 `MAX_PROMPT_LEN`·`MIN_RESPONSE_LEN` context budget을 확인하고, prefill chunk와 `PREFILL_HINT`를 본다 [S5]. 이어서 supported data types와 feature support 범위를 확인하고, 그래도 제약이 남으면 CPU/GPU 대안이나 다른 실행 모드를 비교한다 [S1].
- 심화 설명형 2 해설 포인트: 첫 배포 직후의 느림은 NPU 연산이 불안정해서가 아니라 first-run에 compile 성격의 비용이 앞에 붙었을 가능성이 크다 [S1]. 이때 FEIL/FIL은 첫 실행 지연을 steady-state와 분리해 해석하게 해 주고, model caching은 재실행에서 compile 재사용이 일어나 체감이 달라질 수 있음을 설명한다 [S1]. 따라서 첫 요청과 이후 요청을 같은 숫자로 뭉치면 원인을 잘못 붙이게 된다.
