# Vendor Case Studies and Comparison 퀴즈

## 객관식 1
이 챕터가 벤더 비교의 기본 축으로 제시한 조합은 무엇인가?

1. 가격, TOPS, 출시 연도
2. first action, next object, observability hook
3. 지원 프레임워크 수, 벤치마크 점수, 마케팅 슬로건
4. 전력, 메모리 용량, 시장 점유율

## 객관식 2
다음 중 `Windows ML에서 먼저 확인해야 하는 실패 모드`와 가장 잘 맞는 것은 무엇인가?

1. `tpu-info`에서 topology가 예상과 다르게 보인다.
2. `compile_model(model, "NPU")` 캐시 경로가 비어 있다.
3. standalone ONNX Runtime으로 검증한 가정을 shared ORT/version table 확인 없이 그대로 배포에 가져간다.
4. HailoRT driver 설치 전에 pipeline stdout만 본다.

## 객관식 3
다음 중 `공통 모델 계층 아래에서 infrastructure path로 가장 직접적으로 갈라지는 사례`는 무엇인가?

1. Apple Core ML의 Xcode integration
2. Google TPU의 TPU VM과 queued resource collection
3. AMD Hybrid OGA의 `genai_config.json` bundle
4. HailoRT의 runtime library와 CLI

## 짧은 서술형 1
Qualcomm의 precompiled package 또는 QNN context binary를 다른 벤더의 compiled artifact처럼 재사용하려는 가정이 왜 위험한지 2~3문장으로 설명하라.

## 짧은 서술형 2
AMD Hybrid OGA와 HailoRT가 모두 "빠르게 시작되는 경로"처럼 보이더라도, 같은 종류의 편의성으로 묶으면 안 되는 이유를 설명하라.

## 심화 설명형 1
한 팀이 다음 두 제품을 동시에 준비한다.

- Snapdragon 기반 모바일 앱
- Windows 노트북용 앱

Qualcomm 경로와 Windows ML 경로를 비교하여 다음을 설명하라.

1. 공통 계층은 어디까지 유지되는가?
2. vendor-specific object는 각각 무엇인가?
3. 첫 디버깅 문서와 실패 모드는 왜 달라지는가?

## 심화 설명형 2
다음 두 운영 상황을 비교하라.

- 상황 A: Google TPU에서 single-host serving을 먼저 검증한다.
- 상황 B: Tenstorrent에서 TT-Forge lowering과 toolchain 관찰을 먼저 검증한다.

두 상황의 first action, next object, observability hook, 운영 복잡도 증가 지점을 비교하라.

## 정답 및 해설
- 객관식 1 정답: 2. 이 챕터는 벤더를 성능표가 아니라 `first action -> next object -> observability hook`로 비교한다. 그래야 compile path, packaged runtime path, app/OS path, infra path를 구분할 수 있다. [S2] [S4] [S8] [S10] [S18]
- 객관식 2 정답: 3. Windows ML의 대표적인 실패 모드는 standalone ONNX Runtime에서 검증한 가정을 shared ORT와 shipped version table을 확인하지 않은 채 배포에 옮기는 것이다. 이 경우 문제는 모델 자체보다 runtime coupling에서 생긴다. [S8] [S9]
- 객관식 3 정답: 2. Google TPU는 TPU VM, topology, quota, queued resource collection이 serving 이전 단계에서 먼저 등장하는 infrastructure path의 대표 사례다. [S10] [S11] [S12] [S13]
- 짧은 서술형 1 예시: Qualcomm의 precompiled package와 QNN context binary는 Qualcomm runtime/QNN 문맥에 묶인 vendor-specific artifact다. 따라서 공통 모델 계층처럼 다른 벤더 stack으로 옮겨 재사용할 수 없고, Intel OpenVINO compiled model이나 Tenstorrent lowering artifact와는 별도의 생성 절차를 다시 거쳐야 한다. [S2] [S3] [S5] [S18]
- 짧은 서술형 2 예시: AMD Hybrid OGA는 pre-optimized folder, `genai_config.json`, sample runner, benchmark executable이 중심인 bundle path다. Hailo는 runtime library, CLI, driver, pipeline entry point가 중심인 runtime stack path이므로, 둘 다 시작이 빠르더라도 준비물과 운영 책임이 다르다. [S4] [S14] [S15]
- 심화 설명형 1 해설 포인트: 공통 계층은 모델 의미, tokenizer, 서비스 입출력 계약 수준까지는 유지된다. Qualcomm은 named-device compile 뒤에 precompiled package 또는 QNN context binary와 QNN EP load 경로가 핵심 object가 되고, Windows ML은 App SDK bootstrapper, shared ORT, shipped ORT version table이 핵심 object가 된다. 따라서 Qualcomm은 compile artifact와 EP 연결 로그를, Windows ML은 version coupling과 runtime distribution 상태를 먼저 본다. [S2] [S3] [S8] [S9]
- 심화 설명형 2 해설 포인트: Google TPU는 TPU VM 또는 queued resource collection을 먼저 만들고, topology와 `tpu-info`가 핵심 관측 훅이 된다. Tenstorrent는 TT-Forge lowering으로 시작해 TT-MLIR artifact와 `tt-smi`, TT-NN Visualizer가 핵심 관측 훅이 된다. TPU는 infra prerequisite가 운영 복잡도를 키우고, Tenstorrent는 compiler/toolchain artifact가 복잡도의 중심이 된다. [S10] [S11] [S12] [S13] [S17] [S18]
