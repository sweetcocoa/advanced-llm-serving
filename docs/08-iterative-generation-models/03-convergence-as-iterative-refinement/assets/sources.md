| 대상 | 근거 자료 | 원본 URL | 출처 유형 | 접근 날짜 | 사용 이유 | 저작권/이용 메모 |
| --- | --- | --- | --- | --- | --- | --- |
| 시각 자료 1: denoising step x recurrent depth 계산 격자 | Looped Diffusion Language Models; Recursive Scaling in Masked Diffusion Models | [LoopMDM](https://arxiv.org/abs/2605.26106), [R-MDM](https://arxiv.org/abs/2606.18022) | 원 논문을 바탕으로 재구성한 Mermaid | 2026-08-12 | outer denoising과 inner recurrence를 독립된 계산축으로 설명 | 논문 그림을 복제하지 않고 개념 관계를 직접 재작성 |
| 시각 자료 2: recurrent-depth의 대각선 diffusion-forcing wavefront | Efficient Parallel Samplers for Recurrent-Depth Models and Their Connection to Diffusion Language Models | [arXiv:2510.14961](https://arxiv.org/abs/2510.14961) | 원 논문을 바탕으로 재구성한 Mermaid | 2026-08-12 | token position과 recurrence가 겹쳐 실행되는 sampler 순서를 설명 | 논문 Figure 1을 복제하지 않고 수업용 sequence diagram으로 재작성 |
| 시각 자료 3: iterative serving scheduler 상태 | 2510.14961; Block Diffusion | [arXiv:2510.14961](https://arxiv.org/abs/2510.14961), [arXiv:2503.09573](https://arxiv.org/abs/2503.09573) | 원 논문의 runtime 요구를 합성한 Mermaid | 2026-08-12 | mutable state, freeze, KV cache, output commit의 관계를 설명 | 수업용 합성 해석이며 특정 논문의 시스템 구조를 그대로 옮기지 않음 |

외부 이미지 파일은 포함하지 않았다. 세 도식은 원 논문의 개념을 본문 목적에 맞게 직접 재구성했으며, 각 도식의 근거와 합성 여부를 위 표에 표시했다.
