# 복습 퀴즈

BF16, query head 32개, KV head 8개, head dimension 128이면 층당 토큰 하나의 논리적 KV는 몇 바이트인가? MLA에는 어떤 크기 항목을 확인해야 하는가?

## 정답 및 해설

GQA는 K/V 두 텐서에 8 × 128 × 2바이트씩 저장하므로 4096바이트다. MLA는 공동 KV latent와 decoupled RoPE key 차원을 합해 계산한다. [S3][S4]
