# 반복 생성 모델의 2026 Frontier 퀴즈

## 객관식 1
Diffusion objective와 commitment policy의 관계를 가장 정확히 설명한 것은 무엇인가?

1. 좋은 objective는 commitment order를 자동으로 결정한다.
2. Commitment는 denoiser posterior를 실제 확정 순서와 종료 결정으로 바꾸는 별도 정책이다.
3. Commitment는 KV cache 구현만을 뜻한다.
4. Bidirectional attention에서는 commitment가 필요 없다.

## 객관식 2
Adaptive recurrent depth를 구성하는 세 요소로 가장 적절한 것은 무엇인가?

1. Trajectory, halting gate, intermediate readout
2. Tokenizer, optimizer, data loader
3. Prefill, KV transfer, autoscaling
4. Tensor parallelism, pipeline parallelism, expert parallelism

## 짧은 서술형 1
Deletion-insertion diffusion과 consistency training이 기존 masked diffusion의 서로 다른 어느 부분을 바꾸는지 설명하라.

## 짧은 서술형 2
Looped model에서 반복 횟수를 늘렸을 때 overthinking이 발생할 수 있는 이유를 state trajectory와 readout 관점에서 설명하라.

## 심화 설명형
LoopMDM에서 denoising step과 recurrent depth가 모두 반복 계산인데도 서로 교환 가능한 단위가 아닌 이유를 설명하라. 각 반복이 바꾸는 상태와 학습 의미를 포함하라.

## 정답 및 해설
- 객관식 1 정답: 2. Objective는 transition을 학습하지만 어느 위치를 확정하고 언제 끝낼지는 sampler의 commitment policy가 결정한다.
- 객관식 2 정답: 1. 유효한 adaptive depth에는 여러 깊이에서 의미 있는 state를 만드는 trajectory, 종료를 고르는 gate, 중간 state를 출력으로 읽는 readout이 모두 필요하다.
- 짧은 서술형 1 예시: deletion-insertion은 고정 길이 mask canvas 대신 길이가 변하는 상태 전이를 도입한다. Consistency training은 긴 reverse trajectory를 줄여도 서로 다른 noise level의 prediction이 일관되도록 objective를 바꾼다.
- 짧은 서술형 2 예시: 공유 transition이 유용한 state를 지난 뒤에도 residual update를 계속하면 state가 drift할 수 있다. 또는 좋은 중간 state가 존재해도 readout이나 gate가 적절한 depth를 선택하지 못할 수 있다.
- 심화 설명형 해설: denoising step은 noisy token state와 posterior 조건을 바꾸는 generative transition이다. Recurrent depth는 같은 noisy state 아래에서 hidden computation을 깊게 한다. 따라서 같은 횟수나 FLOP을 배분해도 서로 다른 transition kernel과 trajectory를 만든다.
