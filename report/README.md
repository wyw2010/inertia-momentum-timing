# Inertia Prospectus (LaTeX source)

One-page executive summary plus nine-page prospectus for the Inertia Momentum Timing
fund, as required by the UG54 final project.

## Files

- `main.tex` - the source document. Compiles to approximately 10 pages at 12pt with 0.9in margins.
- `figures/` - the nine PNG figures referenced in the document, exported from the project notebooks at 1200 DPI.

## Compile on Overleaf

1. Go to [overleaf.com](https://www.overleaf.com) and create a new blank project.
2. Delete the default `main.tex` in the project.
3. Drag the contents of this `report/` folder (the `main.tex` file and the `figures/` subfolder) into the Overleaf file panel, preserving the directory structure.
4. Set compiler to **pdfLaTeX** (default) and TeX Live version to the current default.
5. Click **Recompile**.

## Compile locally

```bash
cd report/
pdflatex main.tex
pdflatex main.tex   # second pass for TOC and cross-references
```

## Style

- Font: 12pt Times-like serif (`mathptmx`) in the body, with Inertia brand colors (blue `#4A7BF7`, dark `#1E2A3A`, purple `#7B5EA7`) used for headings and accents so the report matches the charts.
- No em-dashes or en-dashes in any sentence text.
- Abbreviations introduced once and used by acronym thereafter (UMD, DM, OOS, HMM, GBM, FF5).

## Principals placeholders

Edit the `\textbf{Principals:}` line near the top of the executive summary to add your names before submission.

## Before submitting

- [ ] Fill in principal names.
- [ ] Confirm page count is within 10 (adjust `\clearpage` placements or table widths if a section overflows).
- [ ] Regenerate figures at final resolution if any tables or numbers have been updated since last execution.
- [ ] Upload the single compiled PDF plus the Jupyter notebook link to Brightspace.
