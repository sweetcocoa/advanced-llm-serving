# Looped Transformer와 Recurrent Depth 퀴즈

## 객관식 1
Looped Transformer의 recurrent depth를 가장 정확히 설명한 것은 무엇인가?

1. 이전 document segment의 hidden state를 다음 segment attention에 전달한다.
2. 동일한 Transformer block 또는 layer stack을 한 sequence의 hidden state에 깊이 방향으로 반복 적용한다.
3. 이전 출력 token을 prompt 끝에 붙여 같은 모델을 다시 호출한다.
4. 서로 다른 Transformer layer를 반복 없이 한 번씩 실행한다.

## 객관식 2
Deep Equilibrium Model(DEQ)과 일반적인 finite-depth looped Transformer의 차이로 가장 적절한 것은 무엇인가?

1. DEQ는 attention을 사용할 수 없다.
2. Looped Transformer는 parameter sharing을 할 수 없다.
3. DEQ는 equilibrium state를 root finding으로 구하는 것을 중심 계약으로 삼고, 일반 looped Transformer는 정해지거나 선택된 횟수의 trajectory와 중간 readout을 사용한다.
4. 두 모델은 이름만 다르고 학습과 추론이 완전히 같다.

## 객관식 3
Recurrent depth를 늘렸을 때 성능이 항상 좋아진다고 결론 낼 수 없는 가장 직접적인 이유는 무엇인가?

1. shared block은 두 번 이상 실행할 수 없기 때문이다.
2. 추가 loop가 train에서 보지 못한 상태 분포로 drift하거나 overthinking을 일으킬 수 있기 때문이다.
3. recurrent-depth model은 language modeling loss를 사용할 수 없기 때문이다.
4. loop 횟수는 FLOPs와 무관하기 때문이다.

## 짧은 서술형 1
ACT와 Universal Transformer가 현대 adaptive-depth looped LLM에 남긴 핵심 아이디어를 각각 한 문장으로 설명하라 [S1][S2].

## 짧은 서술형 2
Transformer-XL의 segment recurrence와 Huginn의 depth recurrence에서 각각 어떤 상태가 어느 축으로 전달되는지 비교하라 [S3][S7].

## 짧은 서술형 3
Huginn에 대한 latent-CoT probe [S8]와 LOTUS [S13]의 결과가 모순이 아닌 이유를 설명하라.

## 심화 설명형 1
같은 모델을 fixed depth $R=4$, sequence-level adaptive depth, token-level adaptive depth로 serving한다고 하자. 세 방식의 batch regularity, 이론적 FLOPs, 종료 판단, KV 관리 차이를 설명하라. 모델 논문의 FLOPs 감소가 실제 latency 감소를 보장하지 않는 이유도 포함하라.

## 심화 설명형 2
훈련에서는 최대 6회 반복한 모델이 12-hop compositional task를 풀도록 inference에서 12회 반복된다. LoopFormer의 trajectory consistency [S11], `Loop, Think, & Generalize`의 overthinking [S12], SCSE의 anchor-preserving state evolution [S14]이 이 상황의 서로 다른 어떤 실패 가능성을 다루는지 설명하라.

## 정답 및 해설
- 객관식 1 정답: 2. 이번 계보의 recurrence는 sequence나 출력 token 시간이 아니라 hidden representation의 계산 깊이 방향에 있다.
- 객관식 2 정답: 3. 두 계열은 shared transition과 dynamical-system 관점을 공유하지만, DEQ는 fixed-point solve를 모델 정의의 중심에 둔다.
- 객관식 3 정답: 2. inference-depth extrapolation은 학습 궤적 밖의 상태를 만들 수 있고, 과도한 반복이 예측을 악화시키는 overthinking도 관찰됐다 [S11][S12].
- 짧은 서술형 1 예시: ACT는 입력 난이도에 따라 differentiable halting으로 내부 계산 횟수를 고르는 문제를 제시했다. Universal Transformer는 self-attention block을 깊이 방향으로 반복하고 이 halting을 token 위치별로 결합했다.
- 짧은 서술형 2 예시: Transformer-XL은 이전 segment activation을 다음 segment의 attention memory로 전달한다. Huginn은 같은 sequence/token representation을 shared recurrent block에 다시 넣어 출력 전의 계산 깊이를 늘린다.
- 짧은 서술형 3 예시: Huginn 분석은 별도 CoT 정렬 없이 recurrent hidden state에서 사람이 읽을 수 있는 단계별 CoT가 자연 발생했다는 증거가 제한적이라고 본다. LOTUS는 gold CoT token으로 latent 위치를 직접 감독하므로, supervision을 통해 CoT-aligned latent를 만든 결과다.
- 심화 설명형 1 해설: fixed depth는 규칙적인 dense batch를 유지하지만 모든 요청에 같은 계산을 쓴다. sequence-level adaptive depth는 완료된 request slot을 제거하거나 재채워야 한다. token-level depth는 active token만 계산할 가능성이 있지만 ragged attention, gather/scatter, token별 KV visibility가 복잡해진다. FLOPs가 줄어도 작은 불규칙 kernel, synchronization, memory movement, router overhead 때문에 wall-clock latency가 줄지 않을 수 있다.
- 심화 설명형 2 해설: LoopFormer는 서로 다른 loop budget의 trajectory가 유용한 방향으로 이어지도록 학습해 train-test budget mismatch를 줄이려 한다. `Loop, Think, & Generalize`는 recurrence가 compositional depth extrapolation을 돕더라도 지나치면 overthinking으로 악화될 수 있음을 보여 준다. SCSE는 input-conditioned anchor와 deviation을 분리해 anchor에서 shared transition이 불필요한 forcing drift를 만들지 않도록 구조를 제약한다. 세 연구는 각각 trajectory 정렬, 최적 종료 깊이, 상태 전이의 기준점 안정성을 다룬다.
