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
  const pagePath = normalizePath(window.location.pathname);
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
    const targetPath = normalizePath(
      button.getAttribute("data-progress-path") ||
        button.getAttribute("data-progress-toggle") ||
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
          const candidatePath = normalizePath(
            candidate.getAttribute("data-progress-path") ||
              candidate.getAttribute("data-progress-toggle") ||
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

    const targetPath = normalizePath(link.href);
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

function searchScopes() {
  const scopes = Array.from(
    document.querySelectorAll(
      "[data-search-scope], .card-grid, .chapter-grid, .module-grid, .lesson-grid, .overview-grid"
    )
  );

  if (scopes.length) {
    return scopes;
  }

  const fallbackCards = document.querySelectorAll("[data-search-card], .chapter-card, .module-card, .lesson-card");
  return fallbackCards.length ? [fallbackCards[0].parentElement] : [];
}

function initSearchFilters() {
  const inputs = Array.from(document.querySelectorAll("[data-search-input], input[type='search']"));
  if (!inputs.length) {
    return;
  }

  const scopes = searchScopes().filter(Boolean);
  if (!scopes.length) {
    return;
  }

  const filterScope = (scope, query) => {
    const cards = Array.from(
      scope.querySelectorAll("[data-search-card], .chapter-card, .module-card, .lesson-card, .card, article")
    ).filter((card) => !card.closest(".toc, [data-toc], nav"));

    if (!cards.length) {
      return { visible: 0, total: 0 };
    }

    let visible = 0;
    for (const card of cards) {
      const text = (card.dataset.searchText || card.textContent || "").replace(/\s+/g, " ").toLowerCase();
      const matched = !query || text.includes(query);
      card.hidden = !matched;
      if (matched) {
        visible += 1;
      }
    }

    const empty =
      scope.querySelector("[data-search-empty]") ||
      scope.parentElement?.querySelector("[data-search-empty]");
    if (empty) {
      empty.classList.toggle("is-visible", query.length > 0 && visible === 0);
    }

    return { visible, total: cards.length };
  };

  const update = (value) => {
    const query = value.trim().toLowerCase();
    const summary = scopes.reduce(
      (acc, scope) => {
        const result = filterScope(scope, query);
        return {
          visible: acc.visible + result.visible,
          total: acc.total + result.total,
        };
      },
      { visible: 0, total: 0 }
    );

    document
      .querySelectorAll("[data-search-count]")
      .forEach((node) => (node.textContent = `${summary.visible}/${summary.total}`));
  };

  for (const input of inputs) {
    input.addEventListener("input", (event) => {
      update(event.currentTarget.value);
    });
  }

  update(inputs[0].value || "");
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
  initSearchFilters();
  initTocSpy();
  enhanceAnswerKeyDetails();
  highlightCurrentNavLink();
});
