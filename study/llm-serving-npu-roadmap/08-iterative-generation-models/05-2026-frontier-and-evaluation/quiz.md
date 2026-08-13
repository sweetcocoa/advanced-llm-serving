# 2026 Frontier와 평가 방법 퀴즈

## 객관식 1
sampler speedup을 가장 공정하게 비교하는 조건은 무엇인가?

1. 가장 큰 batch에서 나온 tokens/s만 비교한다.
2. 같은 모델 checkpoint와 prompt/output set에서 quality tolerance를 고정하고 latency와 network evaluation을 함께 비교한다.
3. 논문 abstract의 최대 speedup만 비교한다.
4. perplexity가 같으면 hardware 조건은 생략한다.

## 객관식 2
adaptive recurrent depth에서 평균 loop 수 감소가 실제 serving speedup을 보장하지 않는 이유는 무엇인가?

1. looped model은 GPU에서 실행할 수 없기 때문이다.
2. token별 depth 차이가 dense batch를 깨뜨려 batching efficiency와 utilization을 낮출 수 있기 때문이다.
3. 모든 token은 항상 같은 loop 수를 쓰기 때문이다.
4. hidden state를 저장할 수 없기 때문이다.

## 짧은 서술형
diffusion sampler를 평가할 때 최종 정확도 외에 기록할 trajectory 지표 네 가지를 적어라.

## 심화 설명형
AR, diffusion, recurrent-depth 모델 세 개를 online chat workload에서 비교하는 benchmark를 설계하라. 고정 조건, latency 지표, quality 조건, compute accounting, streaming/commit 지표를 포함하라.

## 정답 및 해설
- 객관식 1 정답: 2. sampler는 quality-speed tradeoff를 바꾸므로 품질을 고정하지 않은 TPS 비교는 의미가 없다.
- 객관식 2 정답: 2. theoretical compute reduction이 실제 wall-clock 이득이 되려면 depth-aware scheduler가 batch 효율을 유지해야 한다.
- 짧은 서술형 예시: step별 commit 위치, 새로 확정한 token 수, rollback 수, 최초 quality target 도달 step, 실제 종료 step, answer-before-reasoning 비율 등이 있다.
- 심화 설명형 해설: 동일 hardware와 arrival trace, prompt/output set, quality threshold를 고정하고 TTFT, stable-final latency, p95/p99, SLO goodput을 측정한다. network evaluation과 active token-pass, loop/denoise histogram, commit/rollback event를 함께 기록해야 한다.
