# Project Roadmap

## Completed ✅

| Item | What was built |
|------|---------------|
| Phase 1 | XGBoost classifier, 44 features, walk-forward validation, backtest engine |
| Phase 2 | Multi-ticker scanner, 200-day SMA regime filter, model caching (7-day staleness) |
| Phase 3 | Earnings avoidance filter (±2 days), 2% stop-loss, self-simulated paper trader |
| Phase 3 | SQLite position tracking, Telegram alerts, GitHub Actions daily cron (9:15am ET) |

---

## During Paper Trading (60–90 days) — Build the Client Layer

Run the paper trader untouched. While it runs, build these in order:

---

### 1. Streamlit Dashboard — 1–2 weeks
**Goal:** Make everything visible. This is what you demo to a client.

**Pages:**
- **Today** — today's signals, regime status, earnings flags, confidence scores
- **Positions** — open paper trades with unrealised P&L and days held
- **History** — closed trades table, win/loss, avg return per trade
- **Equity Curve** — cumulative P&L chart vs SPY buy-and-hold

**Data source:** reads directly from `paper_trading.db` — no backend needed.

**Files to create:**
- `dashboard/app.py` — Streamlit entry point
- `dashboard/pages/positions.py`
- `dashboard/pages/history.py`
- `dashboard/pages/equity.py`

**Run locally:** `streamlit run dashboard/app.py`

**Deploy free:** Streamlit Community Cloud (connect GitHub repo, done).

---

### 2. Weekly Email/PDF Report — 2–3 days
**Goal:** Auto-generated Sunday summary you can forward to a client.

**Contents:**
- Week's P&L (realised + unrealised)
- Best and worst trade of the week
- Current open positions
- Win rate and trade count since inception

**Implementation:**
- Add `reporter.py` — queries `paper_trading.db`, generates HTML report
- Add to GitHub Actions: new workflow `weekly_report.yml` runs every Sunday 8am ET
- Send via Telegram (simplest) or email via `smtplib`

---

### 3. Research: Backtest More Strategies (ongoing, no deadline)
**Goal:** Find if there's a better edge before going live.

**Experiments to run (each takes ~1 hour):**
- Different hold periods: 3 days vs 5 days vs 10 days
- Different thresholds: per-ticker optimised vs global 0.52
- Sector ETFs: XLK, XLF, XLE, XLV — do they have better signal quality?
- Small caps: higher volatility, potentially stronger momentum signals

**Rule:** Never touch the live paper trader based on research results. Research is separate.

---

### 4. Model Improvement Research (ongoing, no deadline)
**Goal:** Test if additional features improve accuracy before the live account.

**Ideas in priority order:**
1. **Per-ticker threshold** — instead of global 0.52, find the optimal threshold per ticker from walk-forward results. Low effort, potentially meaningful.
2. **LightGBM comparison** — drop-in replacement for XGBoost, often faster and slightly better on tabular data. 1 day of work.
3. **Sentiment overlay** — add a VIX spike filter (if VIX > 30, suppress all signals). Simple macro protection.
4. **News sentiment** — pull headlines from yfinance, score with VADER. Adds a feature, not a replacement.

---

## After 60–90 Days — Go Live

### 5. Portfolio-Level Position Sizing — 2 days
Right now each ticker is sized independently. Real improvement:
- Total portfolio exposure capped at 100%
- Max 20% per ticker
- Allocate proportionally by confidence score across all signals

**Only build this when going live — it changes the backtest logic.**

---

### 6. Live Account ($1k CAD) — after paper proves out
Only after paper trading shows **positive expectancy** (avg trade return > 0 after costs).
- Open a **Questrade** account (Canadian, no commission on ETFs)
- Or **IBKR** (supports API, paper + live on same account)
- Start with $1k, max 3 open positions, same signals as paper trader

Realistic expectation: 15–30% annualised = $150–$300/year on $1k.
Small, but it proves the system and gives you a track record to show clients.

---

### 7. Client Dashboard — when you have results to show
- Password-protected Streamlit app per client
- Shows their allocated capital, open positions, cumulative P&L
- Weekly PDF report emailed automatically
- Multi-client risk profiles (conservative = 0.60 threshold, aggressive = 0.52)

---

## Cut For Now

| Item | Why cut |
|------|---------|
| React frontend | Streamlit is sufficient until you have paying clients |
| Model ensemble | Adds complexity; per-ticker threshold has higher ROI first |
| Short selling | Needs margin account; add after long-only is profitable |
| Options overlay | Requires options approval and more capital |
| Pairs trading | Different strategy entirely — distraction right now |
| Hyperparameter tuning | Nice to have, not urgent |
| Intraday signals | Requires tick data feed and faster infrastructure |
| C execution engine | Zero benefit for daily trading regardless of language |

---

## Timeline

| Phase | What | When |
|-------|------|------|
| ✅ Done | Phase 1–3 fully built | Week 1–3 |
| Now | Streamlit dashboard | Week 3–5 |
| Now | Weekly report | Week 5–6 |
| Ongoing | Research / backtest experiments | During paper trading |
| Month 3 | Paper trading results review | Day 60–90 |
| Month 3 | Portfolio sizing + live $1k | After paper proves out |
| Month 4+ | Client dashboard + onboarding | When results are real |
