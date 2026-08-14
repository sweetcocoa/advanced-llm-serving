# Runtime and Execution Provider 퀴즈

## 객관식 1
Snapdragon 앱에서 같은 ONNX 모델을 처음 열 때만 유난히 느리고, 실행 후 디렉터리에 `_ctx.onnx`와 `.bin`이 생겼다. 이 상황을 가장 정확하게 해석한 것은 무엇인가?

1. NPU kernel이 첫 실행에서만 느려지도록 설계되어 있다.
2. 세션 생성이 QNN context를 새로 만드는 compile path를 탔고, 그 artifact가 이후 재사용 후보가 되었다.
3. ONNX opset import가 자동으로 다시 export되었다.
4. AI Hub가 앱 내부에서 `precompiled_qnn_onnx`를 자동 생성했다.

## 객관식 2
QNN EP와 OpenVINO NPU의 runtime surface 비교로 맞는 것은 무엇인가?

1. 둘 다 동일한 `provider_options` 키로 장치를 고른다.
2. QNN EP는 provider/session option이 먼저 보이고, OpenVINO NPU는 device/property surface가 먼저 보인다.
3. OpenVINO NPU는 cache를 지원하지 않으므로 warm start 차이는 모두 kernel 차이다.
4. QNN EP는 quantized model 조건을 세션 생성과 분리해 나중에만 검사한다.

## 객관식 3
운영 판단 관점에서 가장 적절한 첫 대응은 무엇인가?

1. 첫 실행만 느린 OpenVINO 앱이면 곧바로 steady-state kernel profiler부터 본다.
2. 장치별 warm start 차이가 보이면 먼저 `ov::cache_dir`와 same compilation request 유지 여부를 확인한다.
3. AI Hub artifact를 쓰는 앱은 device target을 볼 필요가 없다.
4. QNN EP에서 dynamic shape는 warm start와만 관련 있고 admission과는 무관하다.

## 짧은 서술형 1
이 챕터가 execution provider를 단순 플러그인 설명으로 끝내지 않고 `graph partition 권한`과 `artifact reuse 경로`를 함께 다루는 이유를 2~3문장으로 설명하라. 답변에는 반드시 `session creation`이라는 표현을 포함하라.

## 짧은 서술형 2
OpenVINO NPU에서 두 번째 실행만 빨라졌을 때, 왜 이를 먼저 `cache miss`와 `cache hit`의 차이로 해석해야 하는지 설명하라. 답변에는 반드시 `ov::cache_dir`와 `same compilation request`를 포함하라.

## 심화 설명형 1
다음 두 앱을 비교 설명하라.

- 앱 A: ONNX Runtime + QNN EP로 세션 생성 시 context를 만든다.
- 앱 B: Qualcomm AI Hub에서 `precompiled_qnn_onnx`를 미리 받아 배포한다.

`compile path`, `import path`, `cold start`, `device target` 네 표현을 모두 사용해 두 앱의 차이를 설명하라.

## 심화 설명형 2
동일한 ONNX 모델을 두 런타임으로 배포했다.

- 경로 1: `providers=["QNNExecutionProvider"]`, `provider_options=[...]`
- 경로 2: `compile_model(model, "NPU")`, `ov::cache_dir`

이 둘이 `무엇을 먼저 확인하고`, `어떤 surface에서 문제를 드러내며`, `artifact reuse를 어떻게 기대하는지`를 비교 설명하라. 답변에는 반드시 `provider/session option`, `device/property surface`, `admission 조건`을 포함하라.

## 정답 및 해설
- 객관식 1 정답: 2. [문서 사실] QNN EP는 context binary cache와 관련 session config를 제공하고, 세션 준비 과정에서 `_ctx.onnx`와 `.bin` 같은 artifact가 생성될 수 있다 [S2]. 따라서 이 증상은 first-run compile path를 먼저 뜻한다.
- 객관식 2 정답: 2. QNN EP는 `QNNExecutionProvider`와 provider option을 통해 정책이 먼저 보이고 [S2], OpenVINO NPU는 `compile_model(..., "NPU")`와 `ov::cache_dir` 같은 device/property surface에서 정책이 드러난다 [S4].
- 객관식 3 정답: 2. [문서 사실] OpenVINO NPU는 `ov::cache_dir`가 켜져 있으면 같은 compilation request에 대해 plugin이 model을 import한다 [S4]. 운영 판단에서는 먼저 cache miss/hit 조건을 확인해야 한다.
- 짧은 서술형 1 예시: execution provider를 단순 플러그인으로만 보면 session creation에서 어떤 subgraph를 claim하고 어떤 artifact reuse 경로를 선택하는지 놓치게 된다. 이 챕터는 EP를 graph partition 권한을 가진 계층으로 읽고, compile path와 import path가 어디서 갈리는지를 함께 보도록 구성됐다. [S2][S4]
- 짧은 서술형 2 예시: OpenVINO NPU 문서는 `ov::cache_dir`가 compile 결과를 저장하고, same compilation request가 다시 들어오면 plugin이 model을 import한다고 설명한다 [S4]. 그래서 두 번째 실행만 빠르면 먼저 cache miss가 cache hit로 바뀌었는지 확인하는 것이 kernel 성능을 의심하는 것보다 우선이다.
- 심화 설명형 1 해설: 앱 A는 앱 내부 session creation에서 compile path를 타며 QNN context를 만든다 [S2]. 그래서 cold start 비용이 앱 시작 구간에 직접 나타난다. 앱 B는 AI Hub에서 device target과 target runtime을 지정해 `precompiled_qnn_onnx`를 만들어 두므로 [S3], 배포 후에는 import path 비중이 더 커지고 cold start 일부를 사전에 옮긴 셈이 된다.
- 심화 설명형 2 해설: QNN 경로에서는 provider/session option이 먼저 보이며, 여기서 backend/device와 context cache를 확인한다 [S2]. OpenVINO 경로에서는 device/property surface가 먼저 보이고, `compile_model(..., "NPU")`, `ov::cache_dir`, execution/performance mode를 본다 [S4]. 둘 다 artifact reuse를 기대하지만, QNN은 quantized model과 fixed shape 같은 admission 조건이 강하게 전면에 나오고 [S2], OpenVINO는 same compilation request에 대한 cache import 조건이 먼저 드러난다 [S4].
