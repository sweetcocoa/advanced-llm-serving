# LLM Inference Metrics 퀴즈

## 객관식 1
다음 중 `request throughput`의 정의로 가장 알맞은 것은 무엇인가?

1. 벽시계 시간 동안 완료한 요청 수
2. 벽시계 시간 동안 읽은 입력 토큰 수
3. 첫 토큰 이후 토큰 하나를 생성하는 평균 시간
4. 장비가 바빴던 시간의 비율

## 객관식 2
실시간 코딩 도우미에서 batch를 더 크게 모았더니 `output-token throughput`은 좋아졌지만 사용자 불만이 늘었다. 가장 그럴듯한 해석은 무엇인가?

1. batch 확대 때문에 queue 대기가 길어져 `TTFT`가 악화되었다.
2. batch 확대는 언제나 `TPOT`과 `TTFT`를 동시에 개선한다.
3. `request throughput`과 `output-token throughput`은 사실 같은 값이다.
4. `utilization`이 높아졌으므로 사용자 체감도 반드시 좋아졌어야 한다.

## 객관식 3
장문 계약서 JSON 추출 API에서 단일 `TPS` 숫자만으로 상태를 설명하기 어려운 가장 좋은 이유는 무엇인가?

1. JSON 응답은 토큰이 아니므로 throughput 계산이 불가능하다.
2. long context는 `prefill` 쪽을, 형식 제약은 `decode` 쪽을 각각 흔들 수 있어 문제 위치가 다르기 때문이다.
3. `TTFT`는 학습 단계에서만 쓰는 지표이기 때문이다.
4. structured output이 있으면 `utilization`을 측정할 수 없기 때문이다.

## 짧은 서술형 1
`TTFT`를 구성하는 요소를 쓰고, 왜 `queue -> prefill -> schedule -> first decode` 순서로 분해하는 것이 장애 분석에 유용한지 3~4문장으로 설명하라.

답변 조건:

- `queue`, `prefill`, `schedule`을 모두 포함할 것
- "첫 반응"이라는 표현을 포함할 것

## 짧은 서술형 2
`request throughput`과 `output-token throughput`을 혼동하면 어떤 오판이 생길 수 있는지 설명하라.

답변 조건:

- `짧은 요청`과 `긴 출력`이라는 표현을 모두 포함할 것
- `TTFT` 또는 `TPOT` 중 하나를 함께 언급할 것

## 심화 설명형 1
실시간 코딩 도우미를 운영한다고 하자. 짧은 버그 수정 요청과 긴 리팩터링 요청이 함께 들어온다. `TTFT`, `TPOT`, `request throughput`, `output-token throughput`, `utilization`의 우선순위를 어떻게 둘지 설명하라.

답변 조건:

- 우선순위 1, 2를 명시할 것
- batch를 키웠을 때의 tradeoff를 포함할 것
- `queue`, `첫 반응`, `스트리밍`을 모두 포함할 것

## 심화 설명형 2
장문 계약서 JSON 추출 API를 설계한다고 하자. 왜 `output-token throughput`만 보고 좋은 설정이라고 판단하면 위험한지 설명하라.

답변 조건:

- `prefill`과 `decode`를 구분할 것
- `request throughput`을 어디에 쓰는지 설명할 것
- 형식 유효성 실패율, 재생성 비율 같은 보조 지표를 하나 이상 제안할 것

## 정답 및 해설
- 객관식 1 정답: 1. `request throughput`은 벽시계 시간 동안 완료한 요청 수다. 출력 토큰 총량을 보는 것은 `output-token throughput`이고, 첫 토큰 이후 평균 시간을 보는 것은 `TPOT`이다.
- 객관식 2 정답: 1. batch를 더 크게 모으면 총 생성 토큰 수나 장비 점유율은 좋아질 수 있지만, 짧은 요청은 queue에서 더 오래 기다려 `TTFT`가 악화될 수 있다.
- 객관식 3 정답: 2. long context는 보통 `prefill` 부담을 키우고, structured output은 `decode` 안정성이나 재생성 비용을 따로 보게 만든다. 그래서 단일 `TPS` 숫자는 완료 요청 수를 늘린 것인지, 출력 토큰 총량을 늘린 것인지, 혹은 첫 반응 지연을 숨긴 것인지 구분하지 못한다.
- 짧은 서술형 1 해설 포인트: `TTFT = queue + prefill + schedule + first decode`처럼 적을 수 있다. 이 분해가 중요한 이유는 첫 반응 지연의 원인이 대기열인지, 긴 입력 처리인지, 스케줄링인지, 첫 decode 준비인지 순서대로 분리하게 만들기 때문이다.
- 짧은 서술형 2 해설 포인트: `request throughput`은 완료 요청 수, `output-token throughput`은 생성 토큰 총량이다. 둘을 혼동하면 짧은 요청을 빨리 많이 끝내는 서비스와 긴 출력을 오래 생성하는 서비스를 같은 총량으로 오해하게 된다. 그 결과 `TTFT`나 `TPOT` 악화를 놓친 채 잘못된 batch 설정을 유지할 수 있다.
- 심화 설명형 1 채점 포인트: 실시간 코딩 도우미라면 보통 `TTFT`와 `TPOT`이 우선이다. 첫 반응과 스트리밍 리듬이 사용자 체감의 핵심이기 때문이다. 그다음에 `request throughput`을 보고, `output-token throughput`과 `utilization`은 보조 지표로 둔다. batch를 키우면 queue가 길어져 짧은 요청의 `TTFT`가 나빠질 수 있다는 tradeoff가 반드시 들어가야 한다.
- 심화 설명형 2 채점 포인트: 답변은 긴 입력이 `prefill`을 키운다는 점과, JSON 형식 제약이 `decode` 구간의 안정성 문제를 따로 드러낸다는 점을 구분해야 한다. `request throughput`은 같은 시간에 몇 건의 문서를 끝냈는지 보는 운영 총량 지표로 설명해야 한다. 형식 유효성 실패율, 재생성 비율, schema-valid 응답 비율 같은 보조 지표를 제안하면 좋다.
