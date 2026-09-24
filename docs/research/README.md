# Research notes

Raw reports produced during planning by research and design-review agents, saved for traceability of the
code comments and the backlog. They are reproduced as delivered,
lightly formatted, with no edits to their claims.

How they were produced: each agent worked from a scoped brief with the instruction to verify against
primary sources rather than recall. Versions, wheel tags and Python requirements came from PyPI's JSON
API; licenses and activity from the GitHub API and raw LICENSE files; clinical numbers from the papers
(PubMed Central full text where open, abstracts otherwise). Where an agent could not verify a claim it
says so inline; treat those as unverified.

Caveats: figures quoted from papers passed through an agent's summary, not a human reading. Before any
number is used in a clinical claim or a capture guide, check it against the cited source. Dates are
when the report was produced; the landscape (model releases, licenses, versions) moves.

| file | scope | produced |
|---|---|---|
| `2026-09-01-pose-estimation-landscape.md` | 2D/3D pose models, markerless mocap toolkits, licenses, Apple Silicon / CUDA / Python-version facts | 2026-09-01 |
| `2026-09-01-biomechanics-and-clinical-validation.md` | biomechanics software, gait-event algorithms, single-camera validation evidence, norms, datasets, regulatory framing | 2026-09-01 |
| `2026-09-01-existing-projects-survey.md` | open-source and commercial PT/rehab/gait CV projects, reuse vs build, failure modes | 2026-09-01 |
| `2026-09-01-scaffold-design-review.md` | design review of the initial package scaffold and Slice 0/1 plan | 2026-09-01 |
| `2026-09-02-desktop-ui-stack.md` | PySide6 vs alternatives, video decode options, packaging, licensing for a desktop viewer | 2026-09-02 |
| `2026-09-02-viewer-design-review.md` | design review of the desktop viewer against the real code | 2026-09-02 |
