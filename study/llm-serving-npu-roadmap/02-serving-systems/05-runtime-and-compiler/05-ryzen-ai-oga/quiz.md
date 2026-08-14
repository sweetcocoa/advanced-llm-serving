# Ryzen AI OGA 퀴즈

## 객관식 1

Ryzen AI 1.7.1 hybrid artifact의 공식 phase 배치는 무엇인가?

1. iGPU prefill + NPU token generation
2. NPU prefill + iGPU token generation
3. CPU prefill + NPU token generation
4. NPU prefill + NPU token generation

## 객관식 2

NPU-only 모드를 가장 정확히 설명한 것은 무엇인가?

1. hybrid artifact에서 iGPU 옵션만 끈다.
2. prefill은 iGPU, decode는 CPU에서 수행한다.
3. 별도 artifact를 사용해 prefill과 decode를 NPU에서 수행한다.
4. 모든 Ryzen AI 릴리스에서 같은 artifact를 사용한다.

## 객관식 3

Ryzen AI 릴리스를 갱신할 때 가장 적절한 절차는 무엇인가?

1. runtime만 갱신하고 이전 artifact를 무조건 재사용한다.
2. OGA와 artifact 호환성을 확인하고 모델을 다시 준비해 회귀 테스트한다.
3. 장치 이름이 같으므로 버전 기록을 생략한다.
4. TTFT만 재고 TPS와 에너지는 이전 값을 쓴다.

## 짧은 서술형 1

왜 `GPU prefill -> NPU decode`라는 일반 직관을 Ryzen AI 1.7.1 hybrid에 적용하면 안 되는지 설명하라.

## 짧은 서술형 2

긴 입력·짧은 출력과 짧은 입력·긴 출력에서 hybrid의 지배 phase가 어떻게 달라질 수 있는지 설명하라.

## 심화 설명형

hybrid와 NPU-only를 비교하는 실험을 설계하라. 답변에 다음 항목을 포함하라.

- artifact와 runtime 버전
- input/output token 길이
- TTFT와 TPS
- phase handoff
- 시스템 전력과 메모리

## 정답 및 해설

- 객관식 1 정답: 2. AMD의 1.7.1 모델 준비 문서는 hybrid를 `NPU prefill phase + GPU token phase`로 명시한다. [S1]
- 객관식 2 정답: 3. NPU-only는 별도 생성 경로를 사용하며 prefill과 decode를 NPU에서 수행한다. [S1]
- 객관식 3 정답: 2. 현재 문서는 1.8과 이전 릴리스 모델의 비호환을 명시하므로 OGA·artifact 조합을 확인하고 재검증해야 한다. [S3]
- 짧은 서술형 1 예시: vendor runtime의 실제 partition은 일반적인 하드웨어 직관과 다를 수 있다. 1.7.1에서는 공식 artifact 계약이 NPU prefill과 iGPU token phase이므로 그 계약과 phase 로그를 기준으로 판단해야 한다. [S1]
- 짧은 서술형 2 예시: 긴 입력·짧은 출력에서는 NPU prefill 시간이 상대적으로 커지고, 짧은 입력·긴 출력에서는 iGPU token generation 시간이 지배적일 수 있다. 따라서 hybrid의 성능과 에너지는 phase별로 측정해야 한다.
- 심화 설명형 해설 포인트: 같은 모델 계열과 품질 조건에서 각 실행 모드에 맞는 artifact를 준비하고 Ryzen AI·OGA·driver 버전을 기록한다. input/output 길이를 여러 조합으로 고정해 TTFT와 TPS를 따로 재며, hybrid handoff 시간과 전체 시스템 전력·메모리를 함께 기록한다. [S1] [S3]
