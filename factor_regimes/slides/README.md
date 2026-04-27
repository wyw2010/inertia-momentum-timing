# Inertia v2 Presentation Slides (Beamer)

20-slide deck pitching Inertia v2 (FF5 factor regime detection applied to the
MPSIF Sub-Fund). Built in Beamer, matching the MPSIF annual report aesthetic.

## Compile on Overleaf

1. Create a new blank Overleaf project.
2. Drag `presentation.tex` and the `figures/` folder into the project.
3. Set compiler to **pdfLaTeX**.
4. Click **Recompile**.

## Compile locally

```bash
cd factor_regimes/slides
pdflatex presentation.tex
pdflatex presentation.tex   # second pass for cross-references
```

## Slide layout (~30 minutes)

| # | Slide | Time | Content |
|---:|---|---|---|
| 1 | Title | 1m | Fund name, principals, headline tagline |
| 2 | Question | 1m | What we're trying to answer |
| 3 | Motivation | 2m | Decade-by-decade FF5 Sharpe (factors are time-varying) |
| 4 | Methods | 2m | Three regime-detection approaches at a glance |
| 5 | Honest finding 1 | 2m | No single method beats static FF5 |
| 6 | Honest finding 2 | 2m | Edge correlations are 0.76 to 0.88 (high) |
| 7 | Pivot | 2m | Defensive hybrid construction |
| 8 | Scoreboard | 2m | Sharpe + drawdown bar charts |
| 9 | Headline cumret | 2m | 1990-2026 cumulative return |
| 10 | Statistical test | 2m | Paired Sharpe-difference table |
| 11 | Crisis zoom | 1m | GFC + COVID protection |
| 12 | Robustness | 1m | Subsample Sharpe by decade |
| 13 | MPSIF application | 2m | Synthetic MPSIF + overlay |
| 14 | MPSIF exposures | 2m | Factor beta bar chart, before/after |
| 15 | MPSIF cumret | 1m | With/without overlay |
| 16 | MPSIF drawdown | 1m | Drawdown comparison |
| 17 | Risks | 2m | Risks table |
| 18 | Why it works | 2m | Economic interpretation |
| 19 | Conclusion | 1m | Summary |
| 20 | Q&A | 1m | Closing |

## Files

- `presentation.tex` - Beamer source (20 slides, 16:9)
- `figures/` - 13 PNG figures referenced in the slides (subset of `factor_regimes/figures/`)

## Style

- 16:9 aspect, 11pt base
- Latin Modern serif (matches the MPSIF annual report)
- Primary purple `#5D4E8C`, cornflower blue `#5B7BC9` (matching report color scheme)
- Sage green `#3FA47D` for positives, warm red `#C84B4B` for negatives
- Frame titles in purple bold with thin purple rule below
- Footer: just the slide counter at the bottom right
- No em-dashes or en-dashes anywhere in body text

## To customize

- Edit principal names on slides 1 and 20
- Adjust colors in the `\definecolor` block at the top of the .tex
- Add `\pause` directives within frames if you want progressive reveals during the live talk
