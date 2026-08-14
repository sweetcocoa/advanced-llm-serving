# Transformer와 Autoregressive Language Model 퀴즈

## 객관식 1
Transformer와 autoregressive language model의 관계를 가장 정확하게 설명한 것은 무엇인가?

1. Transformer를 사용하면 항상 autoregressive model이다.
2. Autoregressive factorization은 Transformer에서만 구현할 수 있다.
3. Transformer는 표현을 계산하는 구조이고, autoregressive factorization은 sequence probability를 조건부 확률로 분해하는 방식이다.
4. 둘은 이름만 다르고 수학적으로 같은 개념이다.

## 객관식 2
Autoregressive LM 학습에서 여러 token 위치의 loss를 한 번에 계산할 수 있는 이유는 무엇인가?

1. 학습에서는 causal mask를 제거하기 때문이다.
2. 정답 sequence가 이미 주어져 있고, causal mask가 각 위치의 참조 범위를 제한하므로 masked matrix computation을 병렬 수행할 수 있기 때문이다.
3. 모든 위치가 미래 token까지 자유롭게 참조하기 때문이다.
4. 학습에서는 attention을 사용하지 않기 때문이다.

## 객관식 3
다음 중 model contract와 serving 구현 선택을 올바르게 구분한 것은 무엇인가?

1. Tokenizer vocabulary는 serving 구현 선택이고, batching 순서는 모델 계약이다.
2. Causal visibility는 모델 계약이고, 요청을 어떤 batch로 묶을지는 serving 구현 선택이다.
3. Temperature는 Transformer block의 고정된 모델 계약이다.
4. KV를 반드시 특정 메모리 형식에 저장해야 한다는 것이 autoregressive factorization의 정의다.

## 짧은 서술형 1
`shifted input`, `causal mask`, `shifted target`이 함께 필요한 이유를 네 문장 이내로 설명하라.

## 짧은 서술형 2
Self-attention과 position-wise MLP가 hidden representation을 갱신할 때 각각 맡는 역할을 비교하라.

## 적용 문제
같은 checkpoint를 두 inference implementation에 올렸더니 첫 step의 logits부터 다르다. 다음 항목을 어떤 순서로 비교할지 쓰고, 각 항목이 model contract인 이유를 설명하라.

- raw input과 normalization 결과
- token IDs와 special tokens
- position IDs
- causal/padding mask
- block별 hidden state
- final vocabulary logits

## 심화 설명형
팀원이 “파라미터가 두 배인 모델이므로 같은 data와 같은 training compute에서도 반드시 더 좋은 language model이다”라고 주장한다. Transformer capacity와 scaling 연구 [S4][S5]를 근거로 이 주장의 빈틈을 설명하라.

## 정답 및 해설
- 객관식 1 정답: 3. 구조와 확률분해는 독립된 설계 축이며 decoder-only GPT 계열에서 결합된다.
- 객관식 2 정답: 2. 학습 target이 알려져 있으므로 허용된 prefix만 보게 하는 mask 아래 여러 query 위치를 행렬로 함께 계산할 수 있다.
- 객관식 3 정답: 2. 미래를 보지 않는 조건은 학습된 확률분포의 의미를 정하지만, 서로 독립인 요청을 묶는 방식은 그 의미를 보존하는 실행 계획이다.
- 짧은 서술형 1 예시: 입력과 target을 한 token만큼 어긋나게 두어 현재 입력 위치의 logits가 다음 token을 예측하게 한다. Causal mask는 그 hidden state가 더 뒤의 입력 token을 참조하지 못하게 한다. 셋 중 하나가 틀리면 현재 또는 미래 정답이 입력에 노출될 수 있다.
- 짧은 서술형 2 예시: Self-attention은 한 위치가 허용된 다른 위치의 정보를 가중합하도록 해 token 사이 관계를 구성한다. MLP는 각 위치에서 공유된 비선형 변환을 적용해 혼합된 표현을 특징 공간에서 변환한다.
- 적용 문제 해설: 입력 전처리부터 logits 방향으로 비교한다. 앞 단계가 다르면 뒤 단계의 차이는 당연한 결과이므로, `raw/normalization -> token/special ID -> position -> mask -> block hidden state -> logits` 순서가 최초 불일치 지점을 찾기 좋다.
- 심화 설명형 해설: 더 많은 width와 depth는 더 큰 함수 공간을 제공하지만, 그 capacity가 실제로 학습됐다는 보장은 아니다. Loss는 model size뿐 아니라 data와 optimization compute의 영향을 받는다 [S4]. 고정 compute에서는 parameter 수와 training token 수를 함께 배분해야 하며, 지나치게 큰 모델을 적은 token으로 학습하면 compute-optimal하지 않을 수 있다 [S5].
