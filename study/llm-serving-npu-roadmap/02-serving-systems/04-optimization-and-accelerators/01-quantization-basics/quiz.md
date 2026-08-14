# Quantization Basics 퀴즈

## 객관식 1
8B 모델을 서버 GPU에 올리려는데 현재 비어 있는 메모리가 10GB이고, 서비스는 고객지원 챗봇이라 품질 흔들림에 민감하다. 본문 기준으로 가장 먼저 둘 기본선은 무엇인가?
1. FP16, 품질 손실이 없으니 무조건 우선이다
2. INT8, [식 1]상 예산을 맞추면서 INT4보다 덜 공격적인 8비트 계열이기 때문이다
3. INT4, 가장 작으니 다른 조건을 볼 필요가 없다
4. 아무 형식이나 괜찮다. 서버에서는 backend를 보지 않아도 된다

## 객관식 2
다음 중 NPU 배포에서 형식 선택 순서를 가장 정확히 설명한 것은 무엇인가?
1. 먼저 가장 낮은 bit-width를 고르고, 안 되면 나중에 compile을 본다
2. 먼저 device target, execution provider, compile 산출물 경로를 확인하고, 그다음 통과한 형식끼리 메모리와 품질을 비교한다
3. FP8은 항상 INT8보다 좋으므로 backend 확인 없이 바로 채택한다
4. INT4는 메모리를 가장 줄이므로 fallback이 있어도 항상 가장 빠르다

## 객관식 3
같은 값 범위를 가정할 때 INT4가 INT8보다 품질 리스크가 더 크다고 설명하는 직접 근거로 가장 적절한 것은 무엇인가?
1. vLLM 문서가 INT4를 금지하기 때문이다
2. OpenVINO NPU 문서가 INT4 품질 수치를 제공하기 때문이다
3. [식 2]에서 INT4의 분모가 15, INT8의 분모가 255라서 INT4의 눈금이 훨씬 거칠기 때문이다
4. compile 예제가 INT4를 자동으로 FP16으로 바꿔 주기 때문이다

## 짧은 서술형 1
[식 1]을 이용해 8B 모델의 FP16, INT8, FP8, INT4 weight 예산을 각각 계산하고, 이 계산이 초기 배포 판단에 왜 유용한지 3문장 이내로 설명하라.

## 짧은 서술형 2
"INT4가 더 작으니 무조건 더 빠르다"가 왜 틀릴 수 있는지 설명하라. 답변에는 `dequantize`, `fallback`, `execution provider` 중 최소 두 개를 포함하라.

## 심화 설명형 1
다음 상황에서 어떤 형식을 우선 검토할지 쓰고 이유를 설명하라.

- 장치: 모바일 NPU
- 모델: 8B
- 메모리 예산: 9GB
- backend 상태: INT8은 offload와 compile이 안정적이고, INT4는 일부 연산이 fallback되며, FP8은 target 미지원
- 품질 요구: 요약문 누락이 자주 나면 안 됨

답변에는 [식 1] 또는 [식 2] 중 하나 이상, 그리고 `backend 호환성`을 모두 포함하라.

## 심화 설명형 2
한 팀이 "INT4 아티팩트를 만들었으니 NPU latency는 반드시 줄어든다"라고 주장한다. 본문 기준으로 이를 검증하는 절차를 4단계 이상으로 쓰라. 답변에는 `execution provider`, `compile 산출물`, `device target`, `실측 latency`, `fallback`을 모두 포함하라.

## 정답 및 해설
- 객관식 1 정답: 2. [식 1]상 8B 모델의 INT8은 약 8GB라 10GB 예산 안에 들어가고, [식 2] 관점에서는 INT4보다 덜 공격적인 8비트 계열이라 품질 민감 서비스의 기본선으로 두기 쉽다. 서버 엔진 문맥은 [S1]이 받친다.
- 객관식 2 정답: 2. NPU 배포에서는 형식 비교 전에 execution provider의 offload 경로 [S2], compile 산출물 [S3], device target [S4]을 먼저 확인해야 한다.
- 객관식 3 정답: 3. 품질 리스크 비교의 직접 근거는 backend 문서가 아니라 [식 2]다. 같은 범위를 가정하면 INT4의 눈금이 INT8보다 훨씬 거칠다.
- 짧은 서술형 1 예시 답안: 8B 모델이면 FP16은 약 16GB, INT8은 약 8GB, FP8도 약 8GB, INT4는 약 4GB다. 이 계산은 형식을 평가하기 전에 장치에 실을 수 있는 후보를 빠르게 거르는 데 유용하다. 그래서 실험 순서를 정할 때 첫 필터로 쓰기 좋다.
- 짧은 서술형 2 예시 답안: INT4는 weight 메모리를 줄이더라도 execution provider가 해당 경로를 충분히 offload하지 못하면 fallback이 생길 수 있다 [S2]. 또 dequantize 같은 추가 변환이 붙으면 kernel 이득이 end-to-end latency 개선으로 이어지지 않을 수 있다. 따라서 속도는 bit-width만으로 결정되지 않는다.
- 심화 설명형 1 해설 포인트: 좋은 답안은 INT8을 우선 검토한다. 이유는 [식 1]상 8GB로 예산을 맞추고, backend 문맥에서도 INT8만 offload와 compile이 안정적이기 때문이다 [S2][S3][S4]. 품질 요구가 높으므로, [식 2]상 더 공격적인 INT4를 fallback이 많은 상태로 쓰는 선택은 불리하다고 설명하면 좋다.
- 심화 설명형 2 해설 포인트: 좋은 답안은 1) execution provider가 실제로 어떤 연산을 맡는지 확인 [S2], 2) compile 산출물이 타깃 장치용으로 생성됐는지 확인 [S3], 3) device target과 실행 모드 확인 [S4], 4) fallback 발생 위치 추적, 5) kernel/dequant/fallback을 나눈 실측 latency 비교의 흐름을 가진다.
