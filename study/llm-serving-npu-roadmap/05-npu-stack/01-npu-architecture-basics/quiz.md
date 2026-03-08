# NPU Architecture Basics 퀴즈

## 객관식 1
다음 중 이 챕터의 기준으로 `NPU 특화 경로`를 가장 우선 검토할 만한 상황은 무엇인가?
1. 요청마다 제어 흐름이 크게 바뀌고, 중간 cache를 host가 자주 만지는 경우
2. 입력 shape가 거의 고정되고, 같은 연산 사슬이 프레임마다 반복되는 경우
3. reshape와 후처리가 자주 섞여 다음 tensor 크기를 미리 예측하기 어려운 경우
4. 모델 일부만 빨라지면 안 되고 모든 단계가 동일한 범용 디버깅 경로를 유지해야 하는 경우

## 객관식 2
이 장에서 [S2]와 [S4]를 함께 읽을 때 가장 적절한 해석은 무엇인가?
1. 두 문서는 벤더별 SRAM 크기와 DMA 채널 수를 직접 비교해 준다.
2. 두 문서는 NPU offload, device target, compile/cache 경로를 설명하므로 긴 resident 세그먼트의 가치로 해석하는 것이 적절하다.
3. 두 문서는 ONNX graph 없이도 모든 연산을 자동 fusion한다고 보장한다.
4. 두 문서는 unsupported op가 하나라도 있으면 NPU 실행이 완전히 불가능하다고 말한다.

## 객관식 3
다음 중 `이 챕터의 첫 질문`에 가장 정확히 대응하는 것은 무엇인가?
1. 같은 해상도의 영상 프레임이 일정하게 반복되는 경우
2. execution provider가 graph를 몇 개의 subgraph로 분할하는가를 먼저 본다.
3. export 결과가 QDQ인지 QOperator인지부터 고정한다.
4. 한 번 올린 텐서가 온칩에 얼마나 오래 머무르고 host handoff가 얼마나 잦은가를 먼저 본다.

## 짧은 서술형 1
이 source pack만으로는 벤더별 SRAM/DMA 세부 스펙을 단정하기 어렵다. 그런데도 이 장에서 SRAM, DMA, fusion을 계속 사용하는 이유를 3문장 이내로 설명하라.

## 짧은 서술형 2
`특화 효율과 범용성`의 tradeoff를 기준으로, 어떤 조건이면 NPU 쪽을 선택하고 어떤 조건이면 범용 경로를 선택할지 설명하라.

## 심화 설명형 1
다음 두 워크로드를 비교하라.
- 스마트폰 카메라 보정 파이프라인
- AI PC 회의 요약 보조 기능

어느 쪽이 더 NPU 친화적인지 쓰고, 답안에 `고정 shape`, `on-chip residency`, `DMA refill 리듬`, `host handoff`를 모두 포함하라.

## 심화 설명형 2
회의 요약 보조 기능에서 encoder 뒤에 irregular reshape와 cache handoff가 잦아 provider partition이 잘게 끊기고, compile artifact 재사용 이점도 약해졌다고 하자. 이 상황에서 왜 NPU 특화 경로보다 범용 경로가 더 나을 수 있는지 설명하라. 답안에는 `resident 구간`, `partition boundary`, `compile artifact`, `host handoff`를 반드시 포함하라.

## 정답 및 해설
- 객관식 1 정답: 2. 입력 shape가 거의 고정되고 같은 연산 사슬이 반복되면 resident 구간을 길게 만들기 쉽고 DMA refill도 예측 가능해져 NPU 특화 효율이 커진다.
- 객관식 2 정답: 2. [S2]와 [S4]는 runtime/provider 관점에서 device offload와 target, cache를 설명하므로, 이 장에서는 이를 `긴 세그먼트를 device에 오래 붙들 수 있는가`라는 기준으로 해석한다.
- 객관식 3 정답: 4. `resident 구간 길이`와 `host handoff`는 이 장의 첫 질문이다. 2번은 다음 장인 lowering/partition 관점에 가깝고, 3번은 quantization artifact 형식을 먼저 묻는 03장의 질문에 가깝다.
- 짧은 서술형 1 예시: [S2]와 [S4]는 세부 하드웨어 표보다 offload와 target 경로를 설명하는 문서다. 그래서 이 장에서 SRAM은 온칩 작업 공간, DMA는 그 공간을 다시 채우는 리듬, fusion은 resident 구간을 늘리는 방법으로 제한해 사용한다. 즉 세부 스펙 암기가 아니라 실행 모델 해석 도구로 쓰는 것이다.
- 짧은 서술형 2 예시: 입력 shape가 고정되고, 한 번 올라간 텐서가 온칩에서 여러 op를 거치며, host handoff가 적으면 NPU 쪽이 유리하다. 반대로 불규칙 제어 흐름, 잦은 reshape, cache 관리, host 왕복이 많으면 범용 경로가 더 자연스럽다. tradeoff의 핵심은 절대 성능이 아니라 특화 효율이 범용성 손실보다 큰지다.
- 심화 설명형 1 해설 포인트: 카메라 보정은 해상도와 패턴이 비교적 고정되어 on-chip residency를 길게 만들기 쉽고 DMA refill도 규칙적이다. 회의 요약 보조 기능은 일부 encoder/projection은 후보가 될 수 있지만, 전체 루프는 cache와 host 제어가 섞여 handoff가 많아지기 쉽다. 좋은 답안은 `NPU에 넣을 수 있는가`보다 `NPU에 오래 머무는가`를 중심으로 비교해야 한다.
- 심화 설명형 2 해설 포인트: 좋은 답안은 irregular reshape와 cache 경로 때문에 `partition boundary`가 늘어나고 `host handoff`가 잦아지면 `resident 구간`이 짧아진다고 설명해야 한다. 이때 [S2]가 보여 주는 provider offload 이점은 잘게 쪼개진 경계 비용에 잠식될 수 있고, [S3]의 `compile artifact` 재사용 이점도 안정적 경로가 무너질수록 약해진다. 따라서 이 장에서는 `NPU에 넣을 수 있는가`보다 `NPU에 오래 머무는가`를 먼저 따져 범용 경로가 더 안정적인 선택이 될 수 있다고 답해야 한다.
