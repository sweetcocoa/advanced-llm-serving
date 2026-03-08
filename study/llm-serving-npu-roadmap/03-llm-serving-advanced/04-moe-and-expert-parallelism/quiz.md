# MoE and Expert Parallelism 퀴즈

## 객관식 1
MoE 서비스에서 평균 GPU utilization은 큰 변화가 없는데 p99만 급등했다. 가장 먼저 의심할 원인으로 가장 적절한 것은 무엇인가?

1. 특정 hot expert의 queue depth 증가와 capacity overflow
2. 모든 expert가 같은 속도로 조금씩 느려진 현상
3. KV cache hit율이 너무 높아진 현상
4. structured output validator가 항상 비활성화된 현상

## 객관식 2
인터랙티브 챗봇에서 batch가 매우 작은 상태로 운영 중인데 top-k를 1에서 2로 올렸다. 가장 직접적으로 예상되는 변화는 무엇인가?

1. dispatch와 combine 비용이 늘고, all-to-all 오버헤드가 amortize되지 않기 쉬워진다
2. router가 사라져 dense 모델과 거의 같은 통신 패턴이 된다
3. capacity factor를 바꾸지 않아도 overflow 위험이 자동으로 줄어든다
4. cross-node placement가 더 이상 중요하지 않게 된다

## 객관식 3
다음 중 expert placement 문제를 가장 직접적으로 시사하는 증상은 무엇인가?

1. 자주 같이 선택되는 expert 쌍이 서로 다른 노드에 있어 remote dispatch bytes가 지속적으로 높다
2. 총 파라미터 수가 dense 모델보다 크다
3. 공통 system prompt가 자주 반복된다
4. 출력 JSON의 필드 수가 많다

## 짧은 서술형 1
`활성 파라미터가 dense보다 적다`는 사실만으로 MoE latency를 예측하면 안 되는 이유를 3문장 이내로 설명하라.

## 짧은 서술형 2
MoE 서비스의 tail latency가 나빠졌을 때 점검할 순서를 4단계 이상으로 적어라. 답에는 `router histogram`, `capacity/overflow`, `placement`, `batch 또는 top-k`가 반드시 들어가야 한다.

## 심화 설명형 1
실시간 code assistant가 MoE로 전환된 뒤 사용자가 `가끔 2초 이상 멈췄다가 토큰이 몰아서 나온다`고 불평한다. 현재 조건은 다음과 같다.

- GPU당 동시 활성 요청 수는 낮다.
- top-k는 2다.
- 자주 같이 선택되는 expert 두 개가 서로 다른 노드에 있다.
- capacity factor는 공격적으로 낮게 잡혀 있다.

다음을 설명하라.
1. 이 상황에서 p99가 튀는 직접 원인 3가지
2. 조정 우선순위와 그 이유
3. 언제 dense 대안 재검토가 합리적인지

## 심화 설명형 2
야간 대량 문서 분류 파이프라인과 낮 시간 인터랙티브 상담 봇이 같은 MoE 클러스터를 공유한다. 운영자는 두 workload에 동일한 `top-k`, `capacity factor`, `expert placement` 정책을 쓰고 있다.

다음을 포함해 설명하라.
1. 왜 같은 정책이 두 workload에 모두 최적이 아닐 수 있는가
2. batch size와 skew가 두 workload에서 다르게 작동하는 방식
3. 어느 workload가 MoE의 장점을 더 잘 살리고, 어느 workload가 dense에 더 가까운 선택을 요구하는지

## 정답 및 해설
- 객관식 1 정답: 1. GShard와 Switch Transformer는 expert capacity와 load balancing을 핵심 장치로 설명한다 [S2][S3]. 그래서 평균 utilization이 멀쩡해도 hot expert의 queue depth와 overflow가 p99를 먼저 망가뜨릴 수 있다 [합성] [S2][S3].
- 객관식 2 정답: 1. GShard의 top-2 gating은 토큰이 두 expert 경로를 타게 만들고 [S2], Switch Transformer가 top-1을 택한 이유 중 하나도 이 복잡도를 줄이기 위해서다 [S3]. 작은 batch에서는 이 dispatch/combine 비용이 amortize되지 못해 latency가 바로 드러난다 [합성] [S2][S3].
- 객관식 3 정답: 1. placement 문제는 `어떤 expert가 어디에 놓였는가`와 `그 결과 remote dispatch bytes가 얼마나 생기는가`로 가장 직접적으로 드러난다 [S2].
- 짧은 서술형 1 예시: MoE latency는 활성 파라미터 수뿐 아니라 router, dispatch, combine, expert queue에 의해 결정된다 [S1][S2][S3]. 특히 토큰이 특정 expert로 쏠리면 평균 계산량이 작아도 가장 바쁜 expert가 전체 step을 늦춘다 [합성] [S1][S2]. 그래서 active parameter만 보면 overflow와 all-to-all tail 비용을 놓치게 된다.
- 짧은 서술형 2 예시: `router histogram 또는 balancedness 지표 확인 -> capacity overflow와 reroute 비율 확인 -> 자주 같이 선택되는 expert의 placement와 remote dispatch bytes 확인 -> EPLB나 재배치 정책 검토 -> batch size, top-k, microbatch 정책 확인 -> workload 자체가 low-batch interactive인지 재판단` 순서가 적절하다 [합성] [S1][S2][S3][S4].
- 심화 설명형 1 포인트: 낮은 동시성 때문에 dispatch 오버헤드가 amortize되지 않고 [합성] [S1][S2][S3], top-k=2라 remote all-to-all이 더 자주 일어나며 [S2], 낮은 capacity factor 때문에 overflow나 지연이 겹쳐 p99가 튈 수 있다 [S2][S3]. 우선순위는 `hot pair co-placement 또는 remote traffic 감소 -> capacity factor 재조정 -> top-k 재검토 -> dense 대안 재평가`가 합리적이다. 계속 low-batch interactive이고 routing 비용이 품질 이득보다 크면 dense가 더 실용적일 수 있다 [합성] [S1][S2][S3].
- 심화 설명형 2 포인트: 야간 대량 문서 분류는 batch가 크고 skew가 평균화되기 쉬워 MoE의 active parameter 절감이 throughput 개선으로 이어질 가능성이 높다 [합성] [S1][S2][S3]. 반면 낮 시간 상담 봇은 batch가 작아 top-k와 dispatch 고정 비용이 크게 보이고, 우연한 expert 편중도 바로 p99로 드러난다 [합성] [S1][S2][S3]. 따라서 두 workload에 같은 top-k와 capacity factor를 쓰면 하나에는 과보수, 다른 하나에는 과공격적 정책이 될 수 있다.
