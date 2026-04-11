"""HTML rendering helpers for the OrbitOps demo app."""

from __future__ import annotations

import json
from html import escape

try:
    from .data import (
        BRAND_NAME,
        FAQS,
        FEATURES,
        PLANS,
        TAGLINE,
        TRUST_BADGES,
        dashboard_payload,
    )
except ImportError:  # pragma: no cover - convenience when running from the demo dir
    from data import (  # type: ignore[no-redef]
        BRAND_NAME,
        FAQS,
        FEATURES,
        PLANS,
        TAGLINE,
        TRUST_BADGES,
        dashboard_payload,
    )


def _layout(*, title: str, body_class: str, body: str, bootstrap: dict | None = None) -> str:
    script = ""
    if bootstrap is not None:
        script = (
            "<script>"
            f"window.ORBIT_BOOTSTRAP = {json.dumps(bootstrap)};"
            "</script>"
        )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{escape(title)}</title>
    <link rel="stylesheet" href="/static/styles.css">
  </head>
  <body class="{escape(body_class)}">
    {body}
    {script}
    <script src="/static/app.js" defer></script>
  </body>
</html>
"""


def _nav(active: str) -> str:
    marketing_class = "site-nav__link is-active" if active == "home" else "site-nav__link"
    dashboard_class = "site-nav__link is-active" if active == "dashboard" else "site-nav__link"
    return f"""
<header class="site-shell">
  <nav class="site-nav">
    <a class="site-nav__brand" href="/">{escape(BRAND_NAME)}</a>
    <div class="site-nav__links">
      <a class="{marketing_class}" href="/">Product</a>
      <a class="{dashboard_class}" href="/dashboard">Dashboard</a>
    </div>
  </nav>
</header>
"""


def render_home_page() -> str:
    features = "".join(
        f"""
        <article class="feature-card">
          <p class="eyebrow">Capability</p>
          <h3>{escape(feature["title"])}</h3>
          <p>{escape(feature["copy"])}</p>
        </article>
        """
        for feature in FEATURES
    )
    trust = "".join(f"<span>{escape(name)}</span>" for name in TRUST_BADGES)
    pricing = "".join(
        f"""
        <article class="pricing-card{' pricing-card--featured' if plan['name'] == 'Scale' else ''}">
          <p class="pricing-card__badge">{escape(plan['badge'])}</p>
          <h3>{escape(plan['name'])}</h3>
          <p class="pricing-card__price">{escape(plan['price'])}<span>{escape(plan['suffix'])}</span></p>
          <p class="pricing-card__description">{escape(plan['description'])}</p>
          <ul class="pricing-card__list">
            {''.join(f"<li>{escape(item)}</li>" for item in plan['items'])}
          </ul>
        </article>
        """
        for plan in PLANS
    )
    faqs = "".join(
        f"""
        <details class="faq-item">
          <summary>{escape(item["question"])}</summary>
          <p>{escape(item["answer"])}</p>
        </details>
        """
        for item in FAQS
    )
    body = f"""
{_nav("home")}
<main>
  <section class="hero site-shell">
    <div class="hero__copy">
      <p class="eyebrow">Launch control for product teams</p>
      <h1>{escape(TAGLINE)}</h1>
      <p class="hero__lede">
        A fictional launch-coordination SaaS. Built so muscle-memory can learn
        real frontend and product-surface work.
      </p>
      <div class="hero__actions">
        <a class="button button--primary" href="/dashboard">Open live dashboard</a>
        <a class="button button--secondary" href="DOGFOOD.md">Dogfood prompts</a>
      </div>
    </div>
    <aside class="hero__panel">
      <p class="eyebrow">This week</p>
      <div class="hero-stat">
        <strong>31</strong>
        <span>launch rooms kept on track</span>
      </div>
      <div class="hero-stat">
        <strong>42m</strong>
        <span>median rescue time after an alert</span>
      </div>
      <div class="hero-stat">
        <strong>118%</strong>
        <span>net revenue retention across active workspaces</span>
      </div>
    </aside>
  </section>

  <section class="trust-strip">
    <div class="site-shell trust-strip__inner">
      <p>Operators at fast-moving teams would trust OrbitOps.</p>
      <div class="trust-strip__badges">{trust}</div>
    </div>
  </section>

  <section class="site-shell section-grid">
    <div class="section-heading">
      <p class="eyebrow">Why this demo exists</p>
      <h2>A realistic surface for repeated engineering work</h2>
      <p>
        Change copy, tune metrics, adjust layout, and verify behavior across a
        marketing page plus an interactive dashboard without dragging in a full framework.
      </p>
    </div>
    <div class="feature-grid">{features}</div>
  </section>

  <section class="site-shell pricing-section">
    <div class="section-heading">
      <p class="eyebrow">Pricing</p>
      <h2>Believable enough to feel like a real product site</h2>
      <p>The details are fake, but the workflows are intentionally familiar.</p>
    </div>
    <div class="pricing-grid">{pricing}</div>
  </section>

  <section class="site-shell faq-section">
    <div class="section-heading">
      <p class="eyebrow">FAQ</p>
      <h2>How to use this while dogfooding muscle-memory</h2>
    </div>
    <div class="faq-list">{faqs}</div>
  </section>
</main>
"""
    return _layout(title=f"{BRAND_NAME} Demo", body_class="theme-marketing", body=body)


def _render_segment_markup(payload: dict) -> str:
    segment_name = payload["default_segment"]
    segment = payload["segments"][segment_name]
    bullets = "".join(f"<li>{escape(item)}</li>" for item in segment["bullets"])
    return f"""
    <article class="health-card" data-segment-board data-active-segment="{escape(segment_name)}">
      <p class="eyebrow">Segment focus</p>
      <h3>{escape(segment['label'])}</h3>
      <p class="health-card__headline">{escape(segment['headline'])}</p>
      <p class="health-card__summary">{escape(segment['summary'])}</p>
      <div class="health-card__score">
        <span>Health score</span>
        <strong>{segment['health']}%</strong>
      </div>
      <ul class="health-card__list">{bullets}</ul>
    </article>
    """


def render_dashboard_page() -> str:
    payload = dashboard_payload()
    metrics = "".join(
        f"""
        <article class="metric-card">
          <p>{escape(item['label'])}</p>
          <strong>{escape(item['value'])}</strong>
          <span>{escape(item['delta'])}</span>
          <small>{escape(item['note'])}</small>
        </article>
        """
        for item in payload["metrics"]
    )
    activity = "".join(
        f"""
        <li>
          <strong>{escape(item['team'])}</strong>
          <span>{escape(item['note'])}</span>
          <small>{escape(item['time'])}</small>
        </li>
        """
        for item in payload["activity"]
    )
    checklist = "".join(f"<li>{escape(item)}</li>" for item in payload["checklist"])
    segment_buttons = "".join(
        f"""
        <button class="segment-switcher__button{' is-active' if key == payload['default_segment'] else ''}"
                data-segment="{escape(key)}"
                type="button">
          {escape(value['label'])}
        </button>
        """
        for key, value in payload["segments"].items()
    )
    body = f"""
{_nav("dashboard")}
<main class="dashboard site-shell">
  <section class="dashboard-hero">
    <div>
      <p class="eyebrow">Live rollout board</p>
      <h1>One glance for revenue risk, launch velocity, and rescue work.</h1>
      <p class="dashboard-hero__lede">
        Use this screen to dogfood recurring tasks: update metrics, tune defaults,
        change the active segment, and verify the experience with the local checks.
      </p>
    </div>
    <aside class="status-card">
      <p>Average health</p>
      <strong>{payload['average_health']}%</strong>
      <span>Updated {escape(payload['updated_at'])}</span>
    </aside>
  </section>

  <section class="metric-grid">
    {metrics}
  </section>

  <section class="dashboard-grid">
    <div class="dashboard-stack">
      <div class="segment-switcher">
        <p class="eyebrow">Focus view</p>
        <div class="segment-switcher__buttons">
          {segment_buttons}
        </div>
      </div>
      {_render_segment_markup(payload)}
    </div>
    <aside class="dashboard-sidebar">
      <section class="side-card">
        <p class="eyebrow">Recent activity</p>
        <ul class="activity-list">{activity}</ul>
      </section>
      <section class="side-card">
        <p class="eyebrow">Release checklist</p>
        <ol class="checklist">{checklist}</ol>
      </section>
    </aside>
  </section>
</main>
"""
    return _layout(
        title=f"{BRAND_NAME} Dashboard",
        body_class="theme-dashboard",
        body=body,
        bootstrap=payload,
    )
