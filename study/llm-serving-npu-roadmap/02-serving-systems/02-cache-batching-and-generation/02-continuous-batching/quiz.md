# Continuous Batching 퀴즈

## 객관식 1
다음 중 continuous batching이 static batching보다 throughput과 latency를 함께 개선할 수 있는 이유로 가장 적절한 것은 무엇인가?

1. 모든 요청의 prefill 길이를 동일하게 강제하기 때문이다.
2. decode 중 먼저 끝난 요청의 빈 slot을 step 경계에서 새 요청으로 채워 평균 활성 시퀀스 수를 높이기 때문이다.
3. 한 번에 더 큰 행렬 곱을 실행하므로 KV cache가 필요 없어지기 때문이다.
4. prefix reuse가 꺼져 있어도 모든 tail block을 자동 공유하기 때문이다.

## 객관식 2
다음 중 continuous batching 환경에서 "scheduler 문제"를 가장 직접적으로 의심하게 만드는 관찰은 무엇인가?

1. 모델 weight 적재 시간이 길다.
2. 평균 활성 시퀀스 수가 낮고, 먼저 끝난 요청 뒤 빈 slot이 여러 step 동안 채워지지 않는다.
3. 모든 요청이 같은 길이의 오프라인 배치 추론으로 들어온다.
4. GPU utilization이 100%에 가깝다.

## 객관식 3
문서 요약 요청이 갑자기 몰리는 시간대에, 이미 스트리밍 중인 채팅 응답이 끊기듯 느려진다. 이때 가장 먼저 검토할 대응으로 적절한 것은 무엇인가?

1. decode를 완전히 중단하고 prefill만 계속 실행한다.
2. static batching으로 되돌려 batch 경계를 고정한다.
3. prefill/decode 우선순위와 cache placement를 분리해 볼지 검토한다.
4. prefix reuse를 무조건 끈다.

## 짧은 서술형 1
다음 수식이 continuous batching의 어떤 문제를 보여 주는지 3~4문장으로 설명하라.

$$
S_{\mathrm{idle}} = \sum_{t=1}^{T}\left(B_{\max} - n_t\right)
$$

설명에는 `B_max`, `n_t`, static batching과 continuous batching의 차이를 포함하라.

## 짧은 서술형 2
다음 상황에서 PagedAttention이 왜 continuous batching의 실무 동반자로 자주 같이 언급되는지 3~4문장으로 설명하라.

- 상담 챗봇 서비스
- 짧게 끝나는 세션과 길게 이어지는 세션이 섞여 있음
- decode slot은 자주 비었다가 다시 채워져야 함

## 심화 설명형 1
8개 slot을 가진 GPU에서 6개의 decode step을 관찰했다. step별 활성 시퀀스 수는 아래와 같다.

- static batching: `8, 8, 7, 5, 5, 4`
- continuous batching: `8, 8, 8, 8, 7, 8`

또한 새 요청 3개가 step 3과 step 4 사이에 도착했다고 하자.

1. 두 방식의 `S_idle`을 계산하라.
2. 어떤 방식이 throughput에 유리한지 설명하라.
3. 새 요청의 queueing latency가 왜 달라지는지 6문장 이상으로 설명하라.

## 심화 설명형 2
다음 서비스 구성을 보고 scheduler 관점의 진단 순서를 6문장 이상으로 설명하라.

- 서비스 A: 긴 시스템 프롬프트를 가진 상담 챗봇
- 서비스 B: 같은 diff에 대해 4-way sampling을 수행하는 코드 리뷰 에이전트
- 서비스 C: 점심 시간에 몰리는 문서 요약 요청
- 공통 환경: 하나의 GPU 풀, continuous batching 사용
- 관측: GPU utilization은 아주 높지 않은데, TTFT와 streaming 안정성이 동시에 나빠짐

설명에는 `slot refill`, `prefix reuse hit`, `KV cache admission`, `prefill/decode 분리`를 모두 포함하라.

## 정답 및 해설
- 객관식 1 정답: 2. continuous batching은 먼저 끝난 요청의 빈 slot을 step 경계에서 새 요청으로 메워 평균 활성 시퀀스 수를 유지하려는 방식이다. 그래서 idle slot이 줄고, 대기열의 요청도 다음 큰 batch를 기다리지 않아도 된다 [S1][S2].
- 객관식 2 정답: 2. continuous batching의 핵심 지표는 slot이 비는지보다, 빈 slot이 얼마나 빨리 다시 채워지는지다. 평균 활성 시퀀스 수가 낮고 refill이 늦으면 scheduler, admission, KV cache 관리 쪽을 먼저 봐야 한다 [S2].
- 객관식 3 정답: 3. 문서 요약처럼 prefill-heavy 요청이 몰리면 decode-heavy 채팅 응답과 같은 자원 예산을 두고 경쟁할 수 있다. 이때는 batching 방식 자체보다 prefill/decode 우선순위, placement 분리, disaggregated serving 가능성을 보는 편이 맞다 [S4].
- 짧은 서술형 1 예시: `B_max`는 장비가 한 step에서 처리할 수 있는 최대 활성 시퀀스 수이고, `n_t`는 실제로 그 step에서 살아 있는 시퀀스 수다. 두 값의 차이를 시간축으로 더한 `S_idle`은 slot이 얼마나 놀았는지를 보여 준다. static batching은 중간에 요청이 끝나도 다음 batch 경계까지 빈칸이 남기 쉬워 `S_idle`이 커진다. continuous batching은 step 경계마다 새 요청을 넣어 이 누적 idle을 줄이려 한다 [S1][S2].
- 짧은 서술형 2 예시: continuous batching은 빈 decode slot을 자주 다시 채우는 구조라서, KV cache도 요청 길이 편차에 맞춰 유연하게 늘고 줄어야 한다. 상담 챗봇처럼 짧게 끝나는 세션과 긴 세션이 섞이면 연속 버퍼 중심 관리에서는 메모리 낭비와 admission 실패가 생기기 쉽다. PagedAttention은 block 단위로 KV cache를 관리해 이런 admission 유연성을 높여 준다. 그래서 scheduler가 slot을 잘 채우고 싶어도 cache가 막히지 않게 만드는 기반으로 함께 언급된다 [S1][S2].
- 심화 설명형 1 해설 포인트: static batching의 `S_idle`은 `(8-8)+(8-8)+(8-7)+(8-5)+(8-5)+(8-4)=8`이다. continuous batching의 `S_idle`은 `(8-8)+(8-8)+(8-8)+(8-8)+(8-7)+(8-8)=1`이다. 매우 단순화한 가정에서 step 시간 `\Delta t`가 비슷하다면, 평균 활성 시퀀스 수가 높은 continuous batching 쪽이 더 높은 TPS를 만든다. 새 요청 3개가 step 3~4 사이에 도착했을 때 static batching은 빈 slot이 있어도 기존 batch 경계를 기다리게 할 수 있다. 반면 continuous batching은 step 경계에서 곧바로 admission해 queueing latency를 줄인다. 그래서 두 방식의 차이는 "연산 자체가 더 빨라졌는가"보다 "빈 자원을 얼마나 빨리 재사용하는가"에서 발생한다 [S1][S2].
- 심화 설명형 2 해설 포인트: 첫째, GPU utilization 숫자만 보지 말고 step별 활성 시퀀스 수와 slot refill 지연부터 분리해서 본다. utilization이 아주 높지 않아도 refill이 늦으면 TTFT와 streaming이 동시에 나빠질 수 있다. 둘째, 상담 챗봇과 코드 리뷰 에이전트는 공통 prefix나 긴 세션을 만들 수 있으므로 prefix reuse hit와 shared prefix 유지 비용을 함께 본다 [S3]. 셋째, 새 요청이 늦게 들어오는 이유가 compute 부족인지, KV cache block 확보 실패인지, 즉 KV cache admission 문제인지 구분해야 한다 [S1][S2]. 넷째, 코드 리뷰 4-way sampling은 일부 후보만 오래 남아 slot을 잡을 수 있으므로 공정성 정책을 본다. 다섯째, 점심 시간 문서 요약이 prefill-heavy burst를 만들면 decode-heavy 채팅과 같은 풀에서 충돌하므로 prefill/decode 우선순위와 placement 분리를 검토한다 [S4]. 여섯째, 필요하면 disaggregated serving처럼 자원 풀을 나누어 prefill과 decode의 서로 다른 파형을 별도로 다루는 방향을 고려한다 [S4].
