# 반복 생성 모델의 평가와 도입 판단 퀴즈

## 객관식 1
Sampler speedup을 가장 공정하게 비교하는 조건은 무엇인가?

1. 각 논문이 보고한 최대 tokens/s를 사용한다.
2. 같은 checkpoint와 prompt/output set에서 품질 허용치를 고정하고 계산량과 latency를 함께 비교한다.
3. Perplexity가 같으면 hardware 조건은 생략한다.
4. 가장 큰 offline batch 결과만 비교한다.

## 객관식 2
Quality-constrained goodput이 세는 요청은 무엇인가?

1. 응답을 시작한 모든 요청
2. 평균 latency보다 빠른 요청
3. 품질 기준과 latency SLO를 모두 만족한 요청
4. 가장 적은 loop를 사용한 요청

## 짧은 서술형 1
Diffusion sampler의 trajectory log에 포함할 event를 네 가지 이상 적어라.

## 짧은 서술형 2
평균 recurrent loop 수가 줄어도 실제 serving speedup이 생기지 않을 수 있는 이유를 설명하라.

## 심화 설명형
AR, diffusion, recurrent-depth 모델을 online chat workload에서 비교하는 실험 계약서를 작성하라. 고정 조건, 품질 기준, compute accounting, trajectory telemetry, latency와 채택 판정을 포함하라.

## 정답 및 해설
- 객관식 1 정답: 2. Sampler는 quality-speed trade-off를 바꾸므로 품질이 다른 TPS는 직접 비교할 수 없다.
- 객관식 2 정답: 3. Goodput은 단순 완료량이 아니라 서비스가 약속한 품질과 지연을 모두 충족한 처리량이다.
- 짧은 서술형 1 예시: step별 commit 위치, confidence, rollback 또는 remask 위치, 남은 mutable token 수, 최초 품질 도달 step, 실제 종료 step, 최초 노출과 최종 안정화 시각을 기록한다.
- 짧은 서술형 2 예시: token별 depth 차이가 dense batch를 깨뜨리면 빈 slot과 scheduling overhead가 늘어 batching efficiency와 device utilization이 낮아질 수 있다.
- 심화 설명형 해설: 동일 hardware, model revision, tokenizer, arrival trace, prompt/output set을 사용한다. 품질 threshold를 먼저 고정하고 network evaluation, active token-pass, denoise/loop histogram을 기록한다. TTFT, inter-commit gap, stable-final latency, p95/p99와 quality-constrained goodput을 측정한 뒤 evidence level에 따라 adopt, pilot, hold를 판정한다.
