#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import html
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COURSE_ROOT = REPO_ROOT / "study" / "llm-serving-npu-roadmap"
DOCS_ROOT = REPO_ROOT / "docs"
STATIC_ROOT = REPO_ROOT / "site" / "static"
NUMBERED_DIR = re.compile(r"^\d{2}-")
PUBLIC_MODULE_DIR = re.compile(r"^(0[1-9]|[1-9]\d)-")


@dataclass
class Heading:
    level: int
    title: str
    anchor: str


@dataclass
class Chapter:
    module_slug: str
    module_title: str
    slug: str
    title: str
    summary: str
    reading_time: str
    updated_at: str
    source_dir: Path
    output_dir: Path
    source_path: str
    quiz_path: str | None
    toc: list[Heading] = field(default_factory=list)
    content_html: str = ""
    quiz_html: str = ""
    prev_title: str | None = None
    prev_output_dir: Path | None = None
    next_title: str | None = None
    next_output_dir: Path | None = None


@dataclass
class Module:
    slug: str
    title: str
    summary: str
    readme_body: str
    source_dir: Path
    output_dir: Path
    chapters: list[Chapter] = field(default_factory=list)


def slugify(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"<[^>]+>", "", lowered)
    lowered = re.sub(r"[`*_~\[\](){}.!?,:;\"'\\/]+", "", lowered)
    lowered = re.sub(r"\s+", "-", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered)
    return lowered.strip("-") or "section"


def rel_href(from_dir: Path, target_dir: Path) -> str:
    relative = os.path.relpath(target_dir, start=from_dir).replace(os.sep, "/")
    if relative == ".":
        return "./"
    return relative.rstrip("/") + "/"


def asset_prefix(page_dir: Path) -> str:
    return os.path.relpath(DOCS_ROOT / "assets", start=page_dir).replace(os.sep, "/")


def root_prefix(page_dir: Path) -> str:
    relative = os.path.relpath(DOCS_ROOT, start=page_dir).replace(os.sep, "/")
    if relative == ".":
        return "./"
    return relative.rstrip("/") + "/"


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    if not text.startswith("---\n"):
        return {}, text
    lines = text.splitlines()
    end_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end_index = index
            break
    if end_index is None:
        return {}, text

    metadata: dict[str, object] = {}
    for raw_line in lines[1:end_index]:
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        if ":" not in raw_line:
            continue
        key, raw_value = raw_line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        if raw_value.startswith("[") or raw_value.startswith("{"):
            try:
                metadata[key] = ast.literal_eval(raw_value)
            except Exception:
                metadata[key] = raw_value
            continue
        if raw_value.startswith('"') or raw_value.startswith("'"):
            try:
                metadata[key] = ast.literal_eval(raw_value)
            except Exception:
                metadata[key] = raw_value.strip("\"'")
            continue
        metadata[key] = raw_value
    body = "\n".join(lines[end_index + 1 :]).lstrip()
    return metadata, body


def strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[#>*_`~|-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def first_paragraph(markdown_text: str) -> str:
    blocks = re.split(r"\n\s*\n", markdown_text.strip())
    for block in blocks:
        cleaned = strip_markdown(block)
        if cleaned:
            return cleaned
    return ""


def section_summary(markdown_text: str, heading: str) -> str:
    lines = markdown_text.splitlines()
    collecting = False
    buffer: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^##\s+", stripped):
            if collecting:
                break
            collecting = stripped == f"## {heading}"
            continue
        if collecting:
            buffer.append(line)
    return first_paragraph("\n".join(buffer))


def truncate_text(text: str, limit: int = 220) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def rewrite_link(url: str) -> str:
    if not url or url.startswith(("http://", "https://", "mailto:", "#")):
        return url
    normalized = url.replace("\\", "/")
    if normalized.endswith("/README.md"):
        return normalized[: -len("README.md")]
    if normalized == "README.md":
        return "./"
    if normalized.endswith(".md"):
        return normalized[: -3] + "/"
    return normalized


def split_table_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def is_table_divider(line: str) -> bool:
    stripped = line.strip().strip("|").replace(":", "").replace("-", "")
    return stripped == ""


class MarkdownRenderer:
    def __init__(self) -> None:
        self.headings: list[Heading] = []
        self._used_ids: dict[str, int] = {}

    def render(self, markdown_text: str) -> tuple[str, list[Heading]]:
        lines = markdown_text.splitlines()
        chunks: list[str] = []
        index = 0

        while index < len(lines):
            line = lines[index]
            stripped = line.strip()

            if not stripped:
                index += 1
                continue

            if stripped.startswith("```"):
                fence = stripped[:3]
                language = stripped[3:].strip()
                index += 1
                code_lines: list[str] = []
                while index < len(lines) and not lines[index].strip().startswith(fence):
                    code_lines.append(lines[index])
                    index += 1
                index += 1
                code = "\n".join(code_lines)
                if language == "mermaid":
                    chunks.append(f'<pre class="mermaid">{html.escape(code)}</pre>')
                else:
                    class_name = f' class="language-{html.escape(language)}"' if language else ""
                    chunks.append(
                        f"<pre><code{class_name}>{html.escape(code)}</code></pre>"
                    )
                continue

            if stripped == "$$":
                index += 1
                math_lines: list[str] = []
                while index < len(lines) and lines[index].strip() != "$$":
                    math_lines.append(lines[index])
                    index += 1
                index += 1
                math_content = "\n".join(math_lines)
                chunks.append(
                    '<div class="math-block">$$\n'
                    + html.escape(math_content)
                    + "\n$$</div>"
                )
                continue

            heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                anchor = self._unique_anchor(title)
                if 2 <= level <= 3:
                    self.headings.append(Heading(level=level, title=strip_markdown(title), anchor=anchor))
                chunks.append(
                    f"<h{level} id=\"{anchor}\">{self._inline(title)}</h{level}>"
                )
                index += 1
                continue

            if re.match(r"^[-*_]{3,}$", stripped):
                chunks.append("<hr>")
                index += 1
                continue

            if stripped.startswith(">"):
                quote_lines: list[str] = []
                while index < len(lines) and lines[index].strip().startswith(">"):
                    quote_lines.append(lines[index].strip()[1:].lstrip())
                    index += 1
                quote_html, _ = MarkdownRenderer().render("\n".join(quote_lines))
                chunks.append(f"<blockquote>{quote_html}</blockquote>")
                continue

            if (
                "|" in stripped
                and index + 1 < len(lines)
                and is_table_divider(lines[index + 1])
            ):
                rows: list[list[str]] = [split_table_row(lines[index])]
                index += 2
                while index < len(lines) and "|" in lines[index]:
                    rows.append(split_table_row(lines[index]))
                    index += 1
                header_cells = "".join(f"<th>{self._inline(cell)}</th>" for cell in rows[0])
                body_rows = []
                for row in rows[1:]:
                    cells = "".join(f"<td>{self._inline(cell)}</td>" for cell in row)
                    body_rows.append(f"<tr>{cells}</tr>")
                chunks.append(
                    "<div class=\"table-wrap\"><table><thead><tr>"
                    + header_cells
                    + "</tr></thead><tbody>"
                    + "".join(body_rows)
                    + "</tbody></table></div>"
                )
                continue

            list_match = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
            if list_match:
                kind = "ol" if list_match.group(2).endswith(".") else "ul"
                items: list[str] = []
                while index < len(lines):
                    current = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[index])
                    if not current:
                        break
                    item_text = current.group(3).strip()
                    checkbox = re.match(r"^\[( |x|X)\]\s+(.*)$", item_text)
                    if checkbox:
                        checked = checkbox.group(1).lower() == "x"
                        item_body = self._inline(checkbox.group(2))
                        marker = "checked" if checked else "unchecked"
                        items.append(
                            f'<li class="task-item {marker}"><span class="task-box" aria-hidden="true"></span>{item_body}</li>'
                        )
                    else:
                        items.append(f"<li>{self._inline(item_text)}</li>")
                    index += 1
                chunks.append(f"<{kind}>" + "".join(items) + f"</{kind}>")
                continue

            paragraph_lines = [stripped]
            index += 1
            while index < len(lines):
                candidate = lines[index].strip()
                if not candidate:
                    break
                if candidate.startswith(("```", "$$", ">", "#")):
                    break
                if re.match(r"^[-*_]{3,}$", candidate):
                    break
                if re.match(r"^(\s*)([-*]|\d+\.)\s+", lines[index]):
                    break
                if "|" in candidate and index + 1 < len(lines) and is_table_divider(lines[index + 1]):
                    break
                paragraph_lines.append(candidate)
                index += 1
            paragraph = " ".join(paragraph_lines)
            chunks.append(f"<p>{self._inline(paragraph)}</p>")

        return "\n".join(chunks), self.headings

    def _unique_anchor(self, title: str) -> str:
        base = slugify(strip_markdown(title))
        count = self._used_ids.get(base, 0)
        self._used_ids[base] = count + 1
        if count == 0:
            return base
        return f"{base}-{count + 1}"

    def _inline(self, text: str) -> str:
        placeholders: dict[str, str] = {}

        def reserve(value: str) -> str:
            token = f"@@TOKEN{len(placeholders)}@@"
            placeholders[token] = value
            return token

        def code_replace(match: re.Match[str]) -> str:
            code = html.escape(match.group(1))
            return reserve(f"<code>{code}</code>")

        escaped = re.sub(r"`([^`]+)`", code_replace, html.escape(text))

        def image_replace(match: re.Match[str]) -> str:
            alt = html.escape(match.group(1))
            src = html.escape(rewrite_link(match.group(2)))
            return f'<img src="{src}" alt="{alt}">'

        escaped = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", image_replace, escaped)

        def link_replace(match: re.Match[str]) -> str:
            label = match.group(1)
            href = html.escape(rewrite_link(match.group(2)))
            external = " target=\"_blank\" rel=\"noreferrer\"" if href.startswith(("http://", "https://")) else ""
            return f'<a href="{href}"{external}>{label}</a>'

        escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link_replace, escaped)
        escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
        escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)

        for token, value in placeholders.items():
            escaped = escaped.replace(token, value)
        return escaped


def render_quiz(markdown_text: str) -> str:
    answer_heading = "\n## 정답 및 해설"
    if answer_heading not in markdown_text:
        html_body, _ = MarkdownRenderer().render(markdown_text)
        return html_body
    body, answer = markdown_text.split(answer_heading, 1)
    body_html, _ = MarkdownRenderer().render(body.strip())
    answer_html, _ = MarkdownRenderer().render(("## 정답 및 해설" + answer).strip())
    return (
        body_html
        + "<details class=\"answer-key\"><summary>정답 및 해설 보기</summary>"
        + answer_html
        + "</details>"
    )


def nav_markup(modules: list[Module], current_slug: str | None, page_dir: Path) -> str:
    items = []
    for module in modules:
        href = rel_href(page_dir, module.output_dir)
        active = " is-active" if current_slug == module.slug else ""
        items.append(
            f'<a class="top-nav-link{active}" href="{href}">{html.escape(module.title)}</a>'
        )
    return "".join(items)


def toc_markup(headings: list[Heading]) -> str:
    if not headings:
        return "<p class=\"toc-empty\">이 페이지는 별도 섹션 목차가 없습니다.</p>"
    items = []
    for heading in headings:
        indent = " sub" if heading.level == 3 else ""
        items.append(
            f'<a class="toc-link{indent}" href="#{heading.anchor}">{html.escape(heading.title)}</a>'
        )
    return "".join(items)


def chapter_cards_markup(chapters: list[Chapter], page_dir: Path) -> str:
    cards = []
    for chapter in chapters:
        href = rel_href(page_dir, chapter.output_dir)
        display_summary = truncate_text(chapter.summary, limit=220)
        cards.append(
            "<a class=\"chapter-card\" "
            f'data-card-search="{html.escape((chapter.title + " " + chapter.summary + " " + chapter.module_title).lower())}" '
            f'data-progress-id="{html.escape(chapter.source_path)}" href="{href}">'
            f"<span class=\"chapter-module\">{html.escape(chapter.module_title)}</span>"
            f"<h3>{html.escape(chapter.title)}</h3>"
            f"<p>{html.escape(display_summary)}</p>"
            "<div class=\"chapter-card-meta\">"
            f"<span>{html.escape(chapter.reading_time or '읽기 시간 미정')}</span>"
            f"<span>{html.escape(chapter.updated_at or '날짜 미정')}</span>"
            "<span class=\"chapter-progress-state\">미완료</span>"
            "</div></a>"
        )
    return "".join(cards)


def module_cards_markup(modules: list[Module], page_dir: Path) -> str:
    cards = []
    for module in modules:
        href = rel_href(page_dir, module.output_dir)
        cards.append(
            "<a class=\"module-card\" "
            f'data-card-search="{html.escape((module.title + " " + module.summary).lower())}" '
            f'href="{href}">'
            f"<span class=\"module-index\">{html.escape(module.slug.split('-', 1)[0])}</span>"
            f"<h3>{html.escape(module.title)}</h3>"
            f"<p>{html.escape(module.summary)}</p>"
            "<div class=\"module-card-meta\">"
            f"<span>{len(module.chapters)} chapters</span>"
            "</div></a>"
        )
    return "".join(cards)


def page_shell(
    *,
    page_title: str,
    description: str,
    body_class: str,
    modules: list[Module],
    page_dir: Path,
    current_module_slug: str | None,
    page_kind: str,
    page_id: str,
    hero: str,
    main_content: str,
    sidebar: str = "",
) -> str:
    assets = asset_prefix(page_dir)
    root = root_prefix(page_dir)
    navigation = nav_markup(modules, current_module_slug, page_dir)
    sidebar_markup = f"<aside class=\"page-sidebar\">{sidebar}</aside>" if sidebar else ""
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(page_title)}</title>
  <meta name="description" content="{html.escape(description)}">
  <meta name="theme-color" content="#10182a">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="{assets}/style.css">
</head>
<body class="{body_class}" data-page-kind="{page_kind}" data-page-id="{html.escape(page_id)}" data-root-prefix="{root}">
  <div class="reading-progress" data-reading-progress></div>
  <header class="site-header">
    <a class="brand" href="{root}">
      <span class="brand-kicker">GitHub Pages Study</span>
      <strong>LLM Serving + NPU</strong>
    </a>
    <nav class="top-nav">{navigation}</nav>
    <div class="search-shell">
      <input type="search" placeholder="챕터, 모듈, 키워드 검색" aria-label="검색" data-global-search>
      <div class="search-results" data-search-results hidden></div>
    </div>
  </header>
  <main class="page-frame">
    {hero}
    <div class="page-layout">
      <section class="page-main content-shell">
        {main_content}
      </section>
      {sidebar_markup}
    </div>
  </main>
  <script type="module" src="{assets}/app.js"></script>
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true, theme: 'neutral', securityLevel: 'loose' }});
  </script>
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']] }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
</body>
</html>
"""

def copy_static_assets() -> None:
    target = DOCS_ROOT / "assets"
    shutil.copytree(STATIC_ROOT, target, dirs_exist_ok=True)


def copy_chapter_assets(chapter: Chapter) -> None:
    source_assets = chapter.source_dir / "assets"
    if source_assets.exists():
        shutil.copytree(source_assets, chapter.output_dir / "assets", dirs_exist_ok=True)


def ensure_docs_root() -> None:
    if DOCS_ROOT.exists():
        shutil.rmtree(DOCS_ROOT)
    DOCS_ROOT.mkdir(parents=True)


def load_modules() -> tuple[str, str, list[Module]]:
    course_readme = (COURSE_ROOT / "README.md").read_text(encoding="utf-8")
    _, course_body = parse_frontmatter(course_readme)
    course_title_match = re.search(r"^#\s+(.+)$", course_body, flags=re.M)
    course_title = course_title_match.group(1).strip() if course_title_match else "LLM Serving + NPU"
    course_summary = first_paragraph(re.sub(r"^#\s+.+$", "", course_body, count=1, flags=re.M))

    modules: list[Module] = []
    for module_dir in sorted(
        path
        for path in COURSE_ROOT.iterdir()
        if path.is_dir() and (path / "README.md").exists() and PUBLIC_MODULE_DIR.match(path.name)
    ):
        readme_text = (module_dir / "README.md").read_text(encoding="utf-8")
        _, body = parse_frontmatter(readme_text)
        title_match = re.search(r"^#\s+(.+)$", body, flags=re.M)
        title = title_match.group(1).strip() if title_match else module_dir.name
        summary = first_paragraph(re.sub(r"^#\s+.+$", "", body, count=1, flags=re.M))
        module = Module(
            slug=module_dir.name,
            title=title,
            summary=summary,
            readme_body=body,
            source_dir=module_dir,
            output_dir=DOCS_ROOT / module_dir.name,
        )

        chapter_dirs = sorted(
            path
            for path in module_dir.iterdir()
            if path.is_dir() and (path / "README.md").exists() and NUMBERED_DIR.match(path.name)
        )
        for chapter_dir in chapter_dirs:
            readme_text = (chapter_dir / "README.md").read_text(encoding="utf-8")
            metadata, body = parse_frontmatter(readme_text)
            chapter_title = str(metadata.get("title") or chapter_dir.name.replace("-", " "))
            chapter_summary = section_summary(body, "수업 개요") or first_paragraph(
                re.sub(r"^#\s+.+$", "", body, count=1, flags=re.M)
            )
            quiz_path = chapter_dir / "quiz.md"
            chapter = Chapter(
                module_slug=module.slug,
                module_title=module.title,
                slug=chapter_dir.name,
                title=chapter_title,
                summary=chapter_summary,
                reading_time=str(metadata.get("estimated_reading_time", "")),
                updated_at=str(metadata.get("updated_at", "")),
                source_dir=chapter_dir,
                output_dir=module.output_dir / chapter_dir.name,
                source_path=str(chapter_dir.relative_to(REPO_ROOT)).replace(os.sep, "/"),
                quiz_path=str(quiz_path.relative_to(REPO_ROOT)).replace(os.sep, "/") if quiz_path.exists() else None,
            )
            module.chapters.append(chapter)
        modules.append(module)
    return course_title, course_summary, modules


def render_home_page(course_title: str, course_summary: str, modules: list[Module]) -> str:
    all_chapters = [chapter for module in modules for chapter in module.chapters]
    latest_update = max((chapter.updated_at for chapter in all_chapters if chapter.updated_at), default="-")
    hero = f"""
    <section class="hero hero-home">
      <div class="hero-copy">
        <p class="eyebrow">Static Learning Roadmap</p>
        <h1>{html.escape(course_title)}</h1>
        <p class="hero-summary">{html.escape(course_summary)}</p>
      </div>
      <div class="hero-stats">
        <div class="stat-card"><strong>{len(modules)}</strong><span>modules</span></div>
        <div class="stat-card"><strong>{len(all_chapters)}</strong><span>chapters</span></div>
        <div class="stat-card"><strong>{html.escape(latest_update)}</strong><span>latest update</span></div>
      </div>
    </section>
    """
    main = f"""
    <section class="content-panel surface">
      <div class="section-heading">
        <div>
          <p class="eyebrow">Roadmap</p>
          <h2>모듈별 학습 흐름</h2>
        </div>
        <input type="search" placeholder="이 페이지에서 카드 필터" aria-label="카드 필터" data-card-filter>
      </div>
      <div class="module-grid" data-card-container>
        {module_cards_markup(modules, DOCS_ROOT)}
      </div>
    </section>
    <section class="content-panel surface">
      <div class="section-heading">
        <div>
          <p class="eyebrow">All Chapters</p>
          <h2>바로 시작할 수 있는 챕터</h2>
        </div>
      </div>
      <div class="chapter-grid" data-card-container>
        {chapter_cards_markup(all_chapters, DOCS_ROOT)}
      </div>
    </section>
    """
    sidebar = """
    <div class="sidebar-panel surface">
      <p class="eyebrow">How To Use</p>
      <h3>읽는 흐름</h3>
      <ol>
        <li>모듈 개요를 먼저 읽는다.</li>
        <li>각 챕터의 본문과 퀴즈를 본다.</li>
        <li>완료 버튼으로 개인 진행률을 저장한다.</li>
      </ol>
    </div>
    """
    return page_shell(
        page_title=course_title,
        description=course_summary,
        body_class="page-home",
        modules=modules,
        page_dir=DOCS_ROOT,
        current_module_slug=None,
        page_kind="home",
        page_id="home",
        hero=hero,
        main_content=main,
        sidebar=sidebar,
    )


def render_module_page(module: Module, modules: list[Module]) -> str:
    renderer = MarkdownRenderer()
    body_html, _ = renderer.render(module.readme_body)
    hero = f"""
    <section class="hero">
      <div class="hero-copy">
        <p class="eyebrow">Module</p>
        <h1>{html.escape(module.title)}</h1>
        <p class="hero-summary">{html.escape(module.summary)}</p>
      </div>
      <div class="hero-stats">
        <div class="stat-card"><strong>{len(module.chapters)}</strong><span>chapters</span></div>
      </div>
    </section>
    """
    main = f"""
    <article class="content-panel surface prose">
      {body_html}
    </article>
    <section class="content-panel surface">
      <div class="section-heading">
        <div>
          <p class="eyebrow">Chapters</p>
          <h2>{html.escape(module.title)} 안의 학습 단위</h2>
        </div>
        <input type="search" placeholder="챕터 필터" aria-label="챕터 필터" data-card-filter>
      </div>
      <div class="chapter-grid" data-card-container>
        {chapter_cards_markup(module.chapters, module.output_dir)}
      </div>
    </section>
    """
    sidebar = """
    <div class="sidebar-panel surface">
      <p class="eyebrow">Progress</p>
      <h3>진행 방식</h3>
      <p>챕터 페이지의 완료 버튼은 브라우저 로컬 저장소에 기록됩니다.</p>
    </div>
    """
    return page_shell(
        page_title=f"{module.title} | LLM Serving + NPU",
        description=module.summary,
        body_class="page-module",
        modules=modules,
        page_dir=module.output_dir,
        current_module_slug=module.slug,
        page_kind="module",
        page_id=module.slug,
        hero=hero,
        main_content=main,
        sidebar=sidebar,
    )


def render_chapter_page(chapter: Chapter, modules: list[Module]) -> str:
    readme_text = (chapter.source_dir / "README.md").read_text(encoding="utf-8")
    _, body = parse_frontmatter(readme_text)
    renderer = MarkdownRenderer()
    content_html, headings = renderer.render(body)
    chapter.content_html = content_html
    chapter.toc = headings

    quiz_html = ""
    if chapter.quiz_path:
        quiz_text = (REPO_ROOT / chapter.quiz_path).read_text(encoding="utf-8")
        quiz_html = render_quiz(quiz_text)
        chapter.quiz_html = quiz_html

    prev_link = (
        f'<a class="pager-link" href="{rel_href(chapter.output_dir, chapter.prev_output_dir)}"><span>이전</span><strong>{html.escape(chapter.prev_title or "")}</strong></a>'
        if chapter.prev_output_dir and chapter.prev_title
        else "<span class=\"pager-link is-empty\"></span>"
    )
    next_link = (
        f'<a class="pager-link" href="{rel_href(chapter.output_dir, chapter.next_output_dir)}"><span>다음</span><strong>{html.escape(chapter.next_title or "")}</strong></a>'
        if chapter.next_output_dir and chapter.next_title
        else "<span class=\"pager-link is-empty\"></span>"
    )

    hero = f"""
    <section class="hero">
      <div class="hero-copy">
        <p class="eyebrow">{html.escape(chapter.module_title)}</p>
        <h1>{html.escape(chapter.title)}</h1>
        <p class="hero-summary">{html.escape(chapter.summary)}</p>
      </div>
      <div class="hero-stats">
        <div class="stat-card"><strong>{html.escape(chapter.reading_time or '시간 미정')}</strong><span>reading time</span></div>
        <div class="stat-card"><strong>{html.escape(chapter.updated_at or '-')}</strong><span>updated</span></div>
        <div class="stat-card"><button class="progress-toggle" type="button" data-progress-toggle data-progress-id="{html.escape(chapter.source_path)}">완료로 표시</button></div>
      </div>
    </section>
    """
    main = f"""
    <article class="content-panel surface prose" data-reading-target>
      {chapter.content_html}
    </article>
    <section class="content-panel surface quiz-panel">
      <div class="section-heading">
        <div>
          <p class="eyebrow">Quiz</p>
          <h2>복습 문제</h2>
        </div>
      </div>
      <div class="prose">
        {quiz_html or '<p>이 챕터에는 별도 퀴즈가 없습니다.</p>'}
      </div>
    </section>
    <nav class="pager">
      {prev_link}
      {next_link}
    </nav>
    """
    sidebar = f"""
    <div class="sidebar-panel surface">
      <p class="eyebrow">On This Page</p>
      <div class="toc" data-toc>
        {toc_markup(chapter.toc)}
      </div>
    </div>
    """
    return page_shell(
        page_title=f"{chapter.title} | {chapter.module_title}",
        description=chapter.summary,
        body_class="page-chapter",
        modules=modules,
        page_dir=chapter.output_dir,
        current_module_slug=chapter.module_slug,
        page_kind="chapter",
        page_id=chapter.source_path,
        hero=hero,
        main_content=main,
        sidebar=sidebar,
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_site() -> None:
    ensure_docs_root()
    copy_static_assets()
    write_text(DOCS_ROOT / ".nojekyll", "")

    course_title, course_summary, modules = load_modules()

    flat_chapters = [chapter for module in modules for chapter in module.chapters]
    for index, chapter in enumerate(flat_chapters):
        if index > 0:
            chapter.prev_title = flat_chapters[index - 1].title
            chapter.prev_output_dir = flat_chapters[index - 1].output_dir
        if index + 1 < len(flat_chapters):
            chapter.next_title = flat_chapters[index + 1].title
            chapter.next_output_dir = flat_chapters[index + 1].output_dir

    write_text(DOCS_ROOT / "index.html", render_home_page(course_title, course_summary, modules))

    for module in modules:
        write_text(module.output_dir / "index.html", render_module_page(module, modules))
        for chapter in module.chapters:
            copy_chapter_assets(chapter)
            write_text(chapter.output_dir / "index.html", render_chapter_page(chapter, modules))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the GitHub Pages study site into docs/.")
    parser.parse_args()
    build_site()


if __name__ == "__main__":
    main()
