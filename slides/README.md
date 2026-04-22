# Inertia Presentation (Beamer)

Ten-slide preview deck for the Inertia pitch. Matches the report aesthetic (Computer
Modern serif, purple primary, dark blue secondary, white background, minimal chrome).

## Files

- `presentation.tex` - the Beamer source
- `figures/` - the five PNG figures referenced in the preview slides

## Compile on Overleaf

1. Create a new blank Overleaf project.
2. Drag `presentation.tex` and the `figures/` folder into the project.
3. Set compiler to **pdfLaTeX** and TeX Live to the current default.
4. Click **Recompile**.

## Compile locally

```bash
cd slides/
pdflatex presentation.tex
pdflatex presentation.tex   # second pass for page-count references
```

## What's in this preview (10 of 20 slides)

1. Title
2. Hero chart: vol-matched cumulative return
3. Performance at a glance (3-row table)
4. Momentum is reliable but fragile (side-by-side charts)
5. Daniel and Moskowitz (2015) benchmark (equations)
6. Approach A: Ridge regression (text + compact table)
7. Approach B: Crash shield (chart + rule)
8. Approach C: HMM (chart)
9. The ensemble (equation + bullets)
10. Empirical decorrelation (correlation table)

The remaining ten slides (crisis zoom, paired Sharpe test, alpha regression, risk-return
scatter, subsample robustness, risks, and close) can be added after reviewing the visual
style of this preview.

## Aspect ratio and dimensions

Compiled at 16:9 aspect (standard modern projector), 11pt base size. Suitable for both
a projector and a shared PDF preview. Change `\documentclass[aspectratio=169, 11pt]{beamer}`
to `[aspectratio=43]` if your venue uses the older 4:3 aspect.
