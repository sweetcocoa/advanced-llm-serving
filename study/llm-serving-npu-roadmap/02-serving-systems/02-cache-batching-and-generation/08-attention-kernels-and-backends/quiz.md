# 복습 퀴즈

FlashAttention으로 바꾼 후 피크 메모리는 줄었지만 dense prefill에서 읽는 토큰 쌍의 수는 그대로였다. 두 결과를 설명하라. FlexAttention으로 sliding window를 시험한다면 무엇을 더 관찰할까?

## 정답 및 해설

FlashAttention은 타일마다 누산하여 전체 점수 행렬을 HBM에 쓰는 일을 피한다. dense 마스크가 같으면 토큰 쌍의 수는 같다. Sliding window의 결과와 BlockMask가 실제로 건너뛴 블록을 각각 확인한다. [S1][S7]
