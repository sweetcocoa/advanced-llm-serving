# 수렴으로 보는 반복 정제 퀴즈

## 객관식 1
Diffusion LLM과 recurrent-depth Transformer의 관계를 가장 정확히 설명한 것은 무엇인가?

1. 같은 block을 반복하면 학습 objective와 관계없이 diffusion model이다.
2. 두 모델은 token을 왼쪽에서 오른쪽으로 한 번씩만 확정한다.
3. 둘 다 공유 update operator로 상태를 반복 정제할 수 있지만 상태 공간, 학습 목표, 확정 방식은 다르다.
4. recurrent-depth Transformer는 반드시 `[MASK]` corruption으로 학습한다.

## 객관식 2
2510.14961의 diffusion-forcing sampler가 recurrent-depth decoding을 빠르게 하는 핵심 원리는 무엇인가 [S4]?

1. recurrent block의 parameter를 양자화해 FLOP을 제거한다.
2. 한 token의 모든 recurrence가 끝나기를 기다리지 않고, 여러 token 위치의 latent state를 대각선 wavefront로 함께 정제한다.
3. causal attention을 bidirectional attention으로 교체한다.
4. coda와 prelude를 제거하고 token을 직접 출력한다.

## 객관식 3
`T=32, R=1`과 `T=8, R=4`가 같은 결과를 보장하지 않는 가장 중요한 이유는 무엇인가?

1. recurrent depth는 parameter 수를 매번 두 배로 늘리기 때문이다.
2. denoising step은 token/noise state와 transition condition을 바꾸지만 recurrent depth는 같은 denoising state에서 hidden computation을 더 수행하기 때문이다.
3. denoising step은 GPU에서 실행할 수 없기 때문이다.
4. recurrent depth는 항상 stochastic하고 denoising은 항상 deterministic하기 때문이다.

## 짧은 서술형 1
다음 공통 상태 전이식에서 $z_k$, $k$, $c$가 masked diffusion LM과 recurrent-depth Transformer에서 각각 무엇을 뜻하는지 설명하라.

$$
z_{k+1}=F_\theta(z_k,k,c)
$$

## 짧은 서술형 2
LoopMDM과 R-MDM의 공통점 하나와 차이점 두 개를 쓰고, 각 논문의 결과를 해석할 때 증거 범위를 왜 구분해야 하는지 설명하라 [S5][S6].

## 심화 설명형 1
recurrent-depth sampler가 latent relative delta $\delta_r<\epsilon$인 position을 freeze한다. 이 규칙이 latency를 줄이는 방식과 다음 두 실패 가능성을 설명하라.

- 잘못된 상태에 안정되는 경우
- 앞 token draft가 늦게 바뀌어 뒤 position의 condition이 달라지는 경우

## 심화 설명형 2
다음 request 두 개를 함께 처리하는 iterative serving scheduler를 설계하라.

- Request A: mask ratio가 높고, 최근 inner loop에서 uncertainty가 크게 감소했다.
- Request B: mask ratio는 낮지만 latent state delta가 threshold 위에 있고 latency deadline이 가깝다.

각 request에 outer denoising, inner recurrence, freeze 중 무엇을 우선할지 제안하고, batch fragmentation과 품질 위험을 함께 설명하라. 정답은 하나가 아니지만 판단 기준이 명시되어야 한다.

## 정답 및 해설
- 객관식 1 정답: 3. 공통점은 iterative state refinement이고, 동일성의 경계는 state, objective, attention, transition, termination에서 드러난다.
- 객관식 2 정답: 2. 각 forward pass가 새 token draft를 열면서 이전 위치의 latent도 계속 정제한다. 정보는 causal하게 흐르지만 여러 위치의 recurrence가 겹친다.
- 객관식 3 정답: 2. 두 설정의 block-equivalent FLOP이 비슷해도 outer step과 inner loop는 서로 다른 transition을 수행하므로 품질과 latency가 같다고 볼 수 없다.
- 짧은 서술형 1 예시: masked diffusion에서 $z_k$는 noisy/masked token state 또는 posterior, $k$는 denoising time, $c$는 clean prompt와 conditioning이다. recurrent-depth model에서 $z_k$는 hidden state, $k$는 recurrence index, $c$는 input embedding과 causal context다.
- 짧은 서술형 2 예시: 둘 다 diffusion step 안에서 shared Transformer computation을 반복해 parameter 증가 없이 effective depth를 키운다. LoopMDM은 early-middle layer를 선택적으로 loop하고 language modeling 및 reasoning 결과를 중심으로 분석한다. R-MDM은 denoiser block recursion과 step embedding을 사용하며 Sudoku, Countdown, Text8에서 parameter/step trade-off를 분석한다. 따라서 R-MDM의 structured-task 결과를 범용 instruction LLM의 성능 보장으로 확대하면 안 된다.
- 심화 설명형 1 해설: 안정된 position을 일찍 freeze하면 active wavefront와 반복 계산량을 줄일 수 있다. 그러나 작은 delta는 correctness를 뜻하지 않아 잘못된 fixed point를 확정할 수 있다. 또한 앞 token이 바뀌면 뒤 position의 conditioning이 바뀌므로 이전 delta가 무효가 되고, 너무 이른 freeze는 cascading error를 만든다. 최대 wavefront, 최소 recurrence, logit stability, rollback 가능성 같은 보조 조건이 필요하다.
- 심화 설명형 2 해설 예시: A는 같은 mask state에서 inner recurrence의 한계 이득이 아직 크므로 inner loop를 한 번 더 주는 선택이 합리적이다. B는 deadline 때문에 무제한 recurrence를 줄 수 없지만 delta가 높으므로 즉시 freeze하면 품질 위험이 크다. 짧은 추가 recurrence 뒤 강제 commit하거나 낮은 품질 tier의 종료 규칙을 적용할 수 있다. A와 B의 action이 달라 한 batch로 묶기 어렵다면 action별 queue를 두되, 작은 batch의 utilization 손실과 deadline miss를 함께 측정해야 한다.
