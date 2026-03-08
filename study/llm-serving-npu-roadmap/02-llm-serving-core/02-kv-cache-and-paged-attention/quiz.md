# KV Cache and PagedAttention 퀴즈

## 객관식 1
다음 중 KV cache가 decode 단계에서 필요한 가장 직접적인 이유로 가장 적절한 것은 무엇인가?

1. attention을 convolution으로 바꾸기 위해
2. 과거 토큰의 K, V를 매 step 다시 계산하지 않기 위해
3. 모델 파라미터 수를 자동으로 줄이기 위해
4. 모든 요청의 출력 길이를 같게 만들기 위해

## 객관식 2
가변 길이 요청이 섞인 서빙 풀에서 "총 빈 메모리는 남아 있는데 새 긴 요청이 바로 못 들어간다"는 현상이 반복된다. 가장 먼저 떠올려야 할 문제는 무엇인가?

1. tokenizer 오류
2. fragmentation 또는 비효율적인 연속 메모리 예약
3. prefix reuse hit율 과다
4. 샘플링 temperature 부족

## 객관식 3
다음 중 cache reuse와 cache placement를 올바르게 구분한 설명은 무엇인가?

1. reuse는 KV cache를 압축하고, placement는 attention을 제거한다
2. reuse는 반복 prefix를 다시 쓰는 문제이고, placement는 cache를 어느 자원에 둘지 정하는 문제다
3. reuse와 placement는 같은 뜻이며 구현 이름만 다르다
4. reuse는 decode 전용이고 placement는 prefill 전용이라 서로 완전히 독립이다

## 짧은 서술형 1
다음 식이 운영 지표 관점에서 무엇을 말해 주는지 3~4문장으로 설명하라.

$$
M_{\mathrm{KV}} \approx 2 \times L \times H_{\mathrm{kv}} \times d_{\mathrm{head}} \times N_{\mathrm{tok}} \times b
$$

## 짧은 서술형 2
PagedAttention이 "메모리를 더 많이 넣는 방식"이 아니라 "배치 방식을 바꾸는 방식"이라고 말할 수 있는 이유를 3~4문장으로 설명하라.

## 심화 설명형 1
다음 상황에서 운영자가 어떤 순서로 문제를 좁혀 가야 하는지 설명하라.

- 서비스: 고객 지원 챗봇
- 공통 시스템 프롬프트: 1,200토큰
- 사용자 질문: 평균 150토큰
- 응답: 평균 80토큰
- 관측: 첫 토큰은 점점 빨라졌지만, 동시 접속이 늘면 GPU 메모리 한계에 자주 걸린다

`prefix reuse`, `live tokens`, `memory footprint`, KV cache의 역할을 모두 포함해서 6문장 이상으로 답하라.

## 심화 설명형 2
다음 상황에서 PagedAttention 도입 효과와 cache placement 이슈를 함께 설명하라.

- 워크로드 A: 짧은 질의응답, 출력 100토큰 내외
- 워크로드 B: 코드 생성, 출력 5,000토큰 이상이 자주 발생
- 두 워크로드가 같은 시간대에 섞인다
- prefill은 공용 노드에서 처리하고 decode는 별도 노드로 보내는 구조를 검토 중이다

`fragmentation`, `block table`, `disaggregated serving`, `cache transfer`를 모두 포함해서 6문장 이상으로 답하라.

## 정답 및 해설
- 객관식 1 정답: 2. KV cache의 핵심 역할은 과거 토큰의 K, V를 저장해 두고 decode에서 재사용하는 것이다. 이것이 없으면 긴 생성에서 같은 과거를 반복 계산하게 되어 비용이 급격히 커진다.
- 객관식 2 정답: 2. 이 현상은 총 용량 부족이 아니라 메모리 배치 실패일 수 있다. 가변 길이 시퀀스를 큰 연속 구간으로 예약하면 중간 빈 공간이 남아도 새 긴 요청을 바로 넣지 못하는 fragmentation이 생긴다.
- 객관식 3 정답: 2. cache reuse는 반복 prefix를 다시 계산하지 않게 하는 전략이고, cache placement는 KV cache를 어느 노드나 장치에 둘지 정하는 전략이다. 최신 serving 엔진은 이 둘을 별도 최적화 항목으로 다룬다.
- 짧은 서술형 1 예시: 이 식은 KV cache 메모리가 모델 구조와 현재 살아 있는 총 토큰 수의 곱으로 커진다는 뜻이다. 운영 중 가장 크게 흔들리는 항은 보통 `N_tok`다. 동시 요청 수가 늘거나 긴 시퀀스가 오래 남아 있으면 memory footprint가 빠르게 커진다. 따라서 throughput 문제를 볼 때는 파라미터 크기만이 아니라 live token 관리도 함께 봐야 한다.
- 짧은 서술형 2 예시: PagedAttention은 시퀀스를 block 단위의 논리 목록으로 관리하고, 각 block을 물리 메모리의 임의 위치에 배치한다. 그래서 시퀀스는 연속처럼 보이지만 실제 메모리는 조각난 공간을 활용할 수 있다. 핵심은 용량을 더 넣는 것이 아니라 논리 주소와 물리 주소를 분리하는 데 있다. 이 접근이 가변 길이 요청에서 fragmentation을 낮추는 이유다.
- 심화 설명형 1 해설 포인트: 이 상황에서 prefix reuse가 잘 작동하면 공통 시스템 프롬프트 재계산이 줄어 첫 토큰이 빨라질 수 있다. 하지만 응답이 계속 쌓이고 세션 수가 늘면 live tokens가 증가해 KV cache footprint는 여전히 커진다. 운영자는 reuse hit가 높다는 사실만 보고 안심하면 안 된다. 동시에 살아 있는 토큰 총량과 장수 세션 비율을 봐서 memory pressure의 원인이 무엇인지 구분해야 한다. KV cache는 decode를 빠르게 만들지만, 그 상태를 오래 들고 있으면 메모리 예산을 잡아먹는다. 따라서 이 사례의 핵심은 "reuse 성공"과 "footprint 안정화"를 별개의 목표로 관리하는 것이다.
- 심화 설명형 2 해설 포인트: 길이가 크게 다른 두 워크로드가 섞이면 연속 메모리 예약 방식에서 fragmentation이 커지기 쉽다. PagedAttention은 block table로 논리 시퀀스와 물리 block 배치를 분리해 이런 환경에서 메모리 활용을 개선한다. 그러나 prefill과 decode를 분리한 disaggregated serving을 도입하면 질문이 하나 더 생긴다. prefill 쪽에서 만든 KV cache를 decode 쪽으로 어떻게 전달할지, cache transfer 지연과 placement 정책을 어떻게 설계할지 봐야 한다. 즉, PagedAttention은 메모리 배치 문제를 완화하고, disaggregated serving은 자원 분리 문제를 푸는 접근이다. 둘은 함께 쓸 수 있지만, 하나가 다른 하나를 자동으로 대신하지는 않는다.
