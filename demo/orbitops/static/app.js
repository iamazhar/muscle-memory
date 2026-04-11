const board = document.querySelector("[data-segment-board]");
const buttons = Array.from(document.querySelectorAll("[data-segment]"));

function renderSegment(segmentName, segments) {
  if (!board || !segments[segmentName]) {
    return;
  }

  const segment = segments[segmentName];
  board.dataset.activeSegment = segmentName;
  board.innerHTML = `
    <p class="eyebrow">Segment focus</p>
    <h3>${segment.label}</h3>
    <p class="health-card__headline">${segment.headline}</p>
    <p class="health-card__summary">${segment.summary}</p>
    <div class="health-card__score">
      <span>Health score</span>
      <strong>${segment.health}%</strong>
    </div>
    <ul class="health-card__list">
      ${segment.bullets.map((item) => `<li>${item}</li>`).join("")}
    </ul>
  `;
}

async function hydrateDashboard() {
  if (!board || buttons.length === 0) {
    return;
  }

  const response = await fetch("/api/dashboard.json");
  if (!response.ok) {
    return;
  }

  const payload = await response.json();
  const segments = payload.segments || {};
  let active = payload.default_segment || buttons[0].dataset.segment;
  renderSegment(active, segments);

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      active = button.dataset.segment;
      buttons.forEach((item) => item.classList.toggle("is-active", item === button));
      renderSegment(active, segments);
    });
  });
}

hydrateDashboard().catch(() => {
  // Keep the server-rendered fallback if hydration fails.
});
