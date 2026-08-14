# Diffusion LLM의 계보와 생성 원리 퀴즈

## 객관식 1
masked diffusion LM과 일반적인 BERT-style MLM을 가장 정확히 구분한 것은 무엇인가?

1. masked diffusion만 Transformer를 사용한다.
2. masked diffusion은 noise level, reverse trajectory, 반복 sampler를 generative model의 일부로 정의한다.
3. BERT는 `[MASK]` token을 사용하지 않는다.
4. masked diffusion은 항상 continuous embedding에서 동작한다.

## 객관식 2
parallel decoding이 곧바로 latency 개선을 보장하지 않는 이유는 무엇인가?

1. diffusion LM은 GPU를 사용할 수 없기 때문이다.
2. 여러 token을 확정해도 denoising step마다 긴 block을 다시 처리하고, 보수적인 confidence policy가 많은 step을 요구할 수 있기 때문이다.
3. diffusion LM은 한 번에 한 token만 예측하기 때문이다.
4. 모든 diffusion LM이 CPU에서만 실행되기 때문이다.

## 객관식 3
AR-to-block-diffusion conversion 계보를 가장 정확히 설명한 것은 무엇인가?

1. AR checkpoint의 causal mask를 제거하기만 하면 추가 학습 없이 완성된다.
2. AR pretraining의 지식과 효율을 활용하되, block 내부 denoising을 위해 attention, corruption objective, sampler를 적응한다.
3. block 사이와 block 안을 모두 한 번의 독립적인 token 예측으로 끝낸다.
4. 모든 방법이 Fast-dLLM v2의 hierarchical cache를 동일하게 사용한다.

## 짧은 서술형
D3PM, SEDD, MDLM이 discrete text diffusion의 어떤 문제를 각각 다뤘는지 한 문장씩 설명하라.

## 비교 서술형
Fast-dLLM v2, SDAR, LLaDA2.0, DiffusionGemma가 기존 AR 모델의 자산과 block diffusion을 결합하는 방식을 비교하라. 공통 원리와 서로 같다고 볼 수 없는 구현 차이를 하나씩 포함하라 [S18][S19][S20][S21].

## 심화 설명형
수학 풀이를 masked diffusion LM으로 생성할 때 최종 답 token의 confidence가 가장 먼저 높아졌다. 즉시 확정하는 sampler와 frontier-gated sampler가 생성 품질과 병렬성에 어떤 차이를 만들지 설명하라 [S15].

## 정답 및 해설
- 객관식 1 정답: 2. MLM loss의 외형만이 아니라 forward corruption과 반복 reverse sampling이 generative diffusion을 구성한다.
- 객관식 2 정답: 2. 실제 비용은 network evaluation 횟수와 각 pass의 token 폭을 함께 봐야 한다.
- 객관식 3 정답: 2. 전환 계열은 AR의 학습 자산을 재사용하지만 bidirectional block denoising에 맞는 추가 adaptation과 생성 계약이 필요하다.
- 짧은 서술형 예시: D3PM은 discrete transition과 absorbing mask를 일반화했고, SEDD는 probability ratio 기반 discrete score objective를 제안했으며, MDLM은 masked diffusion objective를 weighted MLM 형태로 단순화했다.
- 비교 서술형 예시: 네 계열은 block 사이의 AR 진행과 block 내부의 병렬 denoising을 결합한다. Fast-dLLM v2는 약 1B-token adaptation과 complementary mask·계층 cache를, SDAR는 짧은 paradigm conversion과 dense/MoE scaling을, LLaDA2.0은 3단계 WSD adaptation과 16B/100B MoE를, DiffusionGemma는 cached prompt encoder와 256-token multi-canvas decoder를 제시한다. 따라서 공통 계보에 놓을 수 있지만 동일한 training recipe나 runtime으로 취급할 수 없다.
- 심화 설명형 해설: 즉시 확정하면 답이 아직 생성되지 않은 reasoning을 역으로 제약해 answer-only collapse가 날 수 있다. frontier gate는 확정 위치를 현재 reasoning 경계 근처로 제한해 순서 일관성을 높이지만, 동시에 확정 가능한 token 수를 줄여 병렬성 일부를 포기한다.
