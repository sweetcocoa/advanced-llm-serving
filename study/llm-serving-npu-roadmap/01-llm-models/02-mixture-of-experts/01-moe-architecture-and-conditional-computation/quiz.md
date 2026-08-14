# MoE Architecture and Conditional Computation 퀴즈

## 객관식 1
Sparse MoE의 총 파라미터와 token당 활성 파라미터를 가장 정확히 설명한 것은 무엇인가?

1. 모든 expert를 실행하므로 두 값은 항상 같다.
2. 전체 expert weight는 총 파라미터에 포함되지만 한 token은 top-k expert만 실행하므로 활성 파라미터는 더 작을 수 있다.
3. 활성 파라미터는 router weight만 뜻한다.
4. Expert 수를 늘리면 활성 파라미터도 반드시 같은 비율로 늘어난다.

## 객관식 2
GShard top-2와 Switch Transformer top-1의 비교로 가장 적절한 것은 무엇인가 [S2][S3]?

1. Top-2에는 router가 없다.
2. Top-1은 모든 expert 출력을 평균한다.
3. Top-2는 두 expert 출력을 결합하고, top-1은 한 expert 경로로 routing을 단순화한다.
4. 두 방식의 forward graph는 완전히 같다.

## 객관식 3
Capacity overflow 규칙을 모델 contract에 포함해야 하는 이유는 무엇인가?

1. Drop, residual, reroute에 따라 모델 출력이 달라질 수 있기 때문이다.
2. Capacity는 문서 제목을 정하는 값이기 때문이다.
3. Capacity는 학습이나 추론과 무관하기 때문이다.
4. Overflow는 dense FFN에서만 발생하기 때문이다.

## 짧은 서술형 1
$E$와 $k$를 각각 늘릴 때 $P_{\mathrm{total}}$과 $P_{\mathrm{active/token}}$이 어떻게 변하는지 설명하라.

## 짧은 서술형 2
Load-balancing objective가 필요한 이유와, 이것이 추론 시 완전한 균형을 보장하지 않는 이유를 설명하라 [S1][S2][S3].

## 심화 설명형 1
Top-2 MoE checkpoint를 top-1만 지원하는 runtime에 배포할 때 두 번째 expert 호출을 단순히 생략하면 안 되는 이유를 출력 결합, capacity, 학습 분포 관점에서 설명하라.

## 심화 설명형 2
Expert layout, router와 top-k, capacity, overflow semantics, combine, precision을 포함한 model-to-runtime contract를 작성하고 각 항목이 빠졌을 때 생기는 오류를 설명하라.

## 정답 및 해설
- 객관식 1 정답: 2. Conditional computation은 전체 expert를 저장하되 입력마다 일부만 활성화해 모델 용량과 token당 계산을 분리한다 [S1][S2].
- 객관식 2 정답: 3. GShard는 top-2 출력을 결합하고 [S2], Switch Transformer는 top-1으로 경로를 단순화한다 [S3].
- 객관식 3 정답: 1. Overflow token의 처리는 forward 결과를 바꾸므로 모델 semantics다 [S2][S3].
- 짧은 서술형 1 예시: $E$만 늘리면 총 파라미터는 증가하지만 $k$가 고정된 한 token의 expert 계산은 그대로일 수 있다. $k$를 늘리면 활성 expert와 계산량이 증가한다.
- 짧은 서술형 2 예시: 보조 손실이 없으면 router가 소수 expert에 집중할 수 있다 [S1]. 그러나 이는 균형을 유도할 뿐이며 입력 분포 변화와 specialization 때문에 추론 routing은 다시 편중될 수 있다 [S1][S2][S3].
- 심화 설명형 1 포인트: Top-2 모델은 두 출력과 gate weight 결합을 전제로 학습됐다. 경로를 제거하면 함수가 바뀌고 top-2 기준 capacity와 router 분포도 맞지 않는다 [S2][S3].
- 심화 설명형 2 포인트: Contract는 weight shape뿐 아니라 token이 어떤 경로를 거쳐 어떤 규칙으로 합쳐지는지 재현해야 한다. Capacity와 overflow가 빠지면 같은 checkpoint도 runtime마다 다른 결과를 낼 수 있다.
