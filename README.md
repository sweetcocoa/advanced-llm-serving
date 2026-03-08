# advanced-llm-serving

`LLM Serving + NPU` 학습 로드맵을 GitHub Pages에서 바로 읽을 수 있는 정적 학습 사이트로 구성한 저장소다.

## 구조
- `study/llm-serving-npu-roadmap`: 원본 학습 콘텐츠 소스
- `site/static`: GitHub Pages용 정적 자산
- `scripts/build_github_pages.py`: `study`를 읽어 `docs/` 사이트를 생성하는 빌드 스크립트
- `docs`: GitHub Pages 배포 산출물

## 로컬 빌드
```bash
python3 scripts/build_github_pages.py
```

빌드가 끝나면 `docs/index.html`이 생성된다.

## GitHub Pages 배포
1. 저장소를 GitHub에 푸시한다.
2. GitHub 저장소 설정에서 `Pages`를 연다.
3. Source를 `Deploy from a branch`로 선택한다.
4. Branch는 `main`, 폴더는 `/docs`를 선택한다.

이후 `main` 브랜치에 `docs/` 변경이 반영되면 GitHub Pages로 정적 학습 페이지를 서빙할 수 있다.

