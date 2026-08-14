const PROGRESS_STORAGE_KEY = "advanced-llm-serving:progress:v1";

function normalizePath(input) {
  try {
    const url = new URL(input, window.location.origin);
    let path = url.pathname.replace(/\/index\.html$/, "/");
    if (!path.endsWith("/")) {
      path += "/";
    }
    return path;
  } catch {
    return input;
  }
}

function progressKey(input) {
  if (!input) {
    return normalizePath(window.location.pathname);
  }
  if (input.startsWith("/") || input.startsWith("http://") || input.startsWith("https://")) {
    return normalizePath(input);
  }
  return input;
}

function readProgress() {
  try {
    const raw = window.localStorage.getItem(PROGRESS_STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function writeProgress(state) {
  try {
    window.localStorage.setItem(PROGRESS_STORAGE_KEY, JSON.stringify(state));
  } catch {
    // Ignore storage failures in private browsing or restricted environments.
  }
}

function isCurrentPageChapter() {
  const hints = [
    document.body.dataset.pageKind,
    document.documentElement.dataset.pageKind,
    document.querySelector("[data-page-kind]")?.getAttribute("data-page-kind"),
  ].filter(Boolean);

  if (hints.some((value) => value === "chapter")) {
    return true;
  }

  return Boolean(
    document.querySelector(".chapter-layout, [data-page-layout], [data-chapter-slug]") ||
      document.querySelector("details.answer-key, details[data-answer-key]")
  );
}

function ensureReadingProgressBar() {
  let root = document.querySelector(".reading-progress, [data-reading-progress]");
  if (!root) {
    root = document.createElement("div");
    root.className = "reading-progress";
    root.setAttribute("data-reading-progress", "true");
    root.innerHTML = '<span class="reading-progress__bar"></span>';
    document.body.prepend(root);
  }

  const bar =
    root.querySelector(".reading-progress__bar") ||
    root.appendChild(document.createElement("span"));
  bar.classList.add("reading-progress__bar");

  const target =
    document.querySelector("[data-reading-target]") ||
    document.querySelector("main article") ||
    document.querySelector("article") ||
    document.querySelector("main");

  const render = () => {
    if (!target) {
      bar.style.width = "0%";
      return;
    }

    const rect = target.getBoundingClientRect();
    const scrollTop = window.scrollY + rect.top;
    const viewport = window.innerHeight;
    const total = Math.max(target.scrollHeight - viewport * 0.65, 1);
    const current = Math.min(Math.max(window.scrollY - scrollTop + viewport * 0.2, 0), total);
    const percent = Math.max(0, Math.min(100, (current / total) * 100));
    bar.style.width = `${percent}%`;
  };

  render();
  window.addEventListener("scroll", render, { passive: true });
  window.addEventListener("resize", render);
}

function findProgressAnchor() {
  return (
    document.querySelector("[data-progress-actions]") ||
    document.querySelector(".hero .button-row, .page-hero .button-row, [data-hero] .button-row") ||
    document.querySelector(".hero, .page-hero, [data-hero]") ||
    document.querySelector("main article") ||
    document.querySelector("main")
  );
}

function ensureProgressToggle() {
  const pagePath = progressKey(document.body.dataset.pageId);
  const state = readProgress();
  const pageComplete = Boolean(state[pagePath]);

  let buttons = Array.from(
    document.querySelectorAll("[data-progress-toggle], .progress-toggle, [data-mark-complete]")
  );

  if (!buttons.length && isCurrentPageChapter()) {
    const anchor = findProgressAnchor();
    if (anchor) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "progress-toggle";
      button.setAttribute("data-progress-toggle", pagePath);
      button.setAttribute("data-progress-path", pagePath);
      const row =
        anchor.matches(".button-row, [data-button-row], [data-progress-actions]") ? anchor : null;

      if (row) {
        row.appendChild(button);
      } else {
        const wrap = document.createElement("div");
        wrap.className = "button-row";
        wrap.setAttribute("data-progress-actions", "true");
        wrap.appendChild(button);
        anchor.appendChild(wrap);
      }

      buttons = [button];
    }
  }

  const syncButton = (button, complete) => {
    button.classList.toggle("is-complete", complete);
    button.setAttribute("aria-pressed", String(complete));
    button.textContent = complete ? "학습 완료 표시됨" : "학습 완료로 표시";
  };

  for (const button of buttons) {
    const targetPath = progressKey(
      button.getAttribute("data-progress-id") ||
        button.getAttribute("data-progress-path") ||
        window.location.pathname
    );
    syncButton(button, Boolean(state[targetPath]));

    button.addEventListener("click", () => {
      const nextState = readProgress();
      if (nextState[targetPath]) {
        delete nextState[targetPath];
      } else {
        nextState[targetPath] = {
          completedAt: new Date().toISOString(),
          title:
            document.querySelector("h1")?.textContent?.trim() ||
            document.title.replace(/\s+\|\s+.+$/, ""),
        };
      }
      writeProgress(nextState);
      const complete = Boolean(nextState[targetPath]);
      buttons
        .filter((candidate) => {
          const candidatePath = progressKey(
            candidate.getAttribute("data-progress-id") ||
              candidate.getAttribute("data-progress-path") ||
              window.location.pathname
          );
          return candidatePath === targetPath;
        })
        .forEach((candidate) => syncButton(candidate, complete));
      decorateCompletedCards();
    });
  }
}

function decorateCompletedCards() {
  const state = readProgress();
  const cards = document.querySelectorAll(
    "[data-search-card], .chapter-card, .module-card, .lesson-card, .card, article, li"
  );

  for (const card of cards) {
    const link =
      card.matches?.('a[href]:not([href^="#"])')
        ? card
        : card.querySelector('a[href]:not([href^="#"])');
    if (!link) {
      continue;
    }

    const targetPath = progressKey(card.getAttribute("data-progress-id") || link.href);
    const complete = Boolean(state[targetPath]);
    card.classList.toggle("is-complete", complete);
    if (complete) {
      card.setAttribute("data-complete", "true");
    } else {
      card.removeAttribute("data-complete");
    }

    let badge = card.querySelector("[data-completion-badge]");
    if (complete && !badge) {
      badge = document.createElement("span");
      badge.className = "completion-badge";
      badge.setAttribute("data-completion-badge", "true");
      badge.textContent = "완료";
      const title = card.querySelector("h2, h3, h4, strong");
      if (title) {
        title.insertAdjacentElement("afterend", badge);
      } else {
        card.prepend(badge);
      }
    }
    if (!complete && badge) {
      badge.remove();
    }
  }
}

function initCardFilters() {
  for (const input of document.querySelectorAll("[data-card-filter]")) {
    const panel = input.closest("section, article, main") || document;
    const scope = panel.querySelector("[data-card-container]");
    if (!scope) {
      continue;
    }

    input.addEventListener("input", () => {
      const query = input.value.trim().toLowerCase();
      for (const card of scope.querySelectorAll("[data-search-card]")) {
        const text = (card.dataset.searchText || card.textContent || "").toLowerCase();
        card.hidden = Boolean(query) && !text.includes(query);
      }
    });
  }
}

function initGlobalSearch() {
  const input = document.querySelector("[data-global-search]");
  const results = document.querySelector("[data-search-results]");
  if (!input || !results) {
    return;
  }

  const root = document.body.dataset.rootPrefix || "./";
  let indexPromise;
  const loadIndex = () => {
    indexPromise ||= fetch(new URL(`${root}search-index.json`, window.location.href))
      .then((response) => (response.ok ? response.json() : []))
      .catch(() => []);
    return indexPromise;
  };

  const close = () => {
    results.replaceChildren();
    results.hidden = true;
    results.classList.remove("is-visible");
  };

  const render = (items) => {
    results.replaceChildren();
    for (const item of items.slice(0, 8)) {
      const link = document.createElement("a");
      link.className = "search-result-link";
      link.href = new URL(`${root}${item.href}`, window.location.href);

      const context = document.createElement("span");
      context.textContent = [item.track, item.section].filter(Boolean).join(" / ");
      const title = document.createElement("strong");
      title.textContent = item.title;
      link.append(context, title);
      results.append(link);
    }
    results.hidden = items.length === 0;
    results.classList.toggle("is-visible", items.length > 0);
  };

  input.addEventListener("input", async () => {
    const query = input.value.trim().toLowerCase();
    if (query.length < 2) {
      close();
      return;
    }
    const index = await loadIndex();
    render(index.filter((item) => item.searchText.includes(query)));
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      close();
    }
  });
}

function initTocSpy() {
  const toc =
    document.querySelector("[data-toc]") ||
    document.querySelector(".toc, .chapter-toc") ||
    document.querySelector('nav[aria-label*="목차"], nav[aria-label*="Table"]');
  if (!toc) {
    return;
  }

  const links = Array.from(toc.querySelectorAll('a[href^="#"]'));
  if (!links.length) {
    return;
  }

  const headings = links
    .map((link) => document.getElementById(link.getAttribute("href").slice(1)))
    .filter(Boolean);
  if (!headings.length) {
    return;
  }

  const setActive = (id) => {
    for (const link of links) {
      const active = link.getAttribute("href") === `#${id}`;
      link.classList.toggle("is-active", active);
      if (active) {
        link.setAttribute("aria-current", "true");
      } else {
        link.removeAttribute("aria-current");
      }
    }
  };

  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (visible[0]?.target?.id) {
        setActive(visible[0].target.id);
      }
    },
    {
      rootMargin: "-20% 0px -65% 0px",
      threshold: [0.1, 0.5, 1],
    }
  );

  headings.forEach((heading) => observer.observe(heading));
  setActive(headings[0].id);
}

function enhanceAnswerKeyDetails() {
  const detailsList = document.querySelectorAll("details.answer-key, details[data-answer-key]");
  for (const details of detailsList) {
    if (!details.hasAttribute("open")) {
      details.open = false;
    }

    const summary = details.querySelector("summary");
    if (summary && !summary.dataset.enhanced) {
      summary.dataset.enhanced = "true";
      if (!summary.textContent.trim()) {
        summary.textContent = "정답 및 해설 보기";
      }
    }
  }
}

function highlightCurrentNavLink() {
  const current = normalizePath(window.location.pathname);
  const navLinks = document.querySelectorAll("header a[href], nav a[href]");
  for (const link of navLinks) {
    const href = link.getAttribute("href");
    if (!href || href.startsWith("#")) {
      continue;
    }
    const target = normalizePath(link.href);
    if (target === current) {
      link.classList.add("is-active");
      link.setAttribute("aria-current", "page");
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  ensureReadingProgressBar();
  ensureProgressToggle();
  decorateCompletedCards();
  initCardFilters();
  initGlobalSearch();
  initTocSpy();
  enhanceAnswerKeyDetails();
  highlightCurrentNavLink();
});
