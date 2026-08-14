# MoE Serving and Expert Parallelism 퀴즈

## 객관식 1
평균 utilization은 비슷하지만 p99가 급등했다. 가장 먼저 확인할 증거는 무엇인가?

1. Layer별 expert histogram, hot expert queue와 overflow
2. Vocabulary 크기
3. System prompt의 문장 부호 수
4. Attention head 이름

## 객관식 2
EP와 dense TP의 차이로 가장 적절한 것은 무엇인가?

1. EP에서는 모든 rank가 항상 같은 token 수를 처리한다.
2. Dense TP가 expert id에 따라 token을 보낸다.
3. EP는 routing에 따라 expert rank로 token을 보내므로 rank별 작업량이 입력마다 달라질 수 있다.
4. 두 방식의 통신은 완전히 같다.

## 객관식 3
EPLB 적용 시 함께 측정해야 할 비용은 무엇인가 [S4]?

1. 균형 개선만 보고 weight 이동은 무시한다.
2. 관측 window, mapping update, weight 이동 또는 replica memory와 tail 개선을 함께 본다.
3. Router 재학습 시간만 측정한다.
4. Tokenizer 생성 시간만 측정한다.

## 짧은 서술형 1
`pack -> dispatch all-to-all -> expert compute -> return all-to-all -> combine`에서 각 단계가 수행하는 일을 설명하라.

## 짧은 서술형 2
Low-batch interactive serving에서 expert FLOPs가 적어도 latency가 나빠질 수 있는 이유를 local batch, collective 고정비, top-k 관점에서 설명하라.

## 심화 설명형 1
Code assistant의 p99가 특정 언어 요청에서만 급등했다. Expert 7 선택률이 평소의 4배이고 remote bytes가 증가했으며 EPLB 직후 weight 이동 traffic도 관측됐다. Router skew, queue, placement와 rebalancing 비용을 분리해 진단하라.

## 심화 설명형 2
야간 대량 분류와 낮 시간 챗봇이 같은 MoE cluster를 공유한다. Concurrency, expert-local batch, p99/goodput, network bytes, balancedness와 dense 기준선을 포함한 workload별 실험을 설계하라.

## 정답 및 해설
- 객관식 1 정답: 1. 가장 늦은 expert를 기다리므로 routing skew와 queue가 평균 utilization보다 tail을 직접 설명한다 [합성] [S1][S2][S3].
- 객관식 2 정답: 3. Sharded expert 실행은 선택 expert로 token을 보내므로 rank별 입력량이 동적으로 변한다 [S2].
- 객관식 3 정답: 2. EPLB의 이득은 mapping 변경과 weight 이동, replica 비용을 포함해 판단해야 한다 [S4].
- 짧은 서술형 1 예시: Pack은 token을 목적지별 buffer로 모으고 dispatch는 expert rank로 보낸다. Expert 결과를 원래 rank로 돌려보낸 뒤 unpack/combine이 token 순서와 gate weight에 맞춰 출력을 복원한다 [합성] [S2].
- 짧은 서술형 2 예시: 동시 token이 적으면 expert GEMM이 작고 collective 고정 지연을 나눌 token도 부족하다. Top-k가 크면 복제와 반환 경로도 늘어난다 [합성] [S2][S3].
- 심화 설명형 1 포인트: 언어별 histogram으로 model-side skew를 확인하고 expert 7의 queue/kernel 시간을 분리한다. EPLB 직후와 안정 구간을 나눠 weight 이동이 일시적 비용인지 지속 placement 실패인지 판정한다 [S4].
- 심화 설명형 2 포인트: 야간 작업은 큰 batch로 비용을 amortize할 가능성이 높지만 챗봇은 순간 skew가 p99에 직접 나타난다. Workload별 concurrency를 고정해 goodput, tail, queue와 bytes를 측정하고 dense 기준선과 비교한다.
