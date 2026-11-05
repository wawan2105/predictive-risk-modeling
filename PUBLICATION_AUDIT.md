# Publication audit — 2026-11-06

Decision: **APPROVED FOR PUBLIC**

## Findings and action

Ready after attribution and display-label repairs; UCI explicitly lists CC BY 4.0. Five integrity tests pass, snapshot checksums and prediction/metric agreement verified. Full analysis reproduces from a clean copy and independent environment.

## Verification

- Existing statistical/data-integrity suite: 5 tests passed.
- Git-history credential scan: 36 historical blobs checked; zero hits for the checked token/private-key patterns and the known local database password; no tracked `.env` file.
- Notebook files were checked for saved error outputs; no error outputs were found. This does not prove all methodological assumptions.
- README/source links were checked locally. Analysis limitations and provenance are retained.
- Scans are targeted checks, not a guarantee that every possible secret or identifier has been detected.

## Clean-copy reproduction

Analysis and tests passed in an independently installed Python 3.11 virtual environment, from copies containing only Git-trackable project files. Generated models, caches and local credential files were absent. Primary metrics matched the existing results. Direct dependencies were pinned; the fresh environment resolved transitive dependencies independently.

All four notebooks were re-executed after label repairs. No model specification, split, selected threshold or substantive metric was changed.

## Source evidence reviewed

- [Source documentation](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- [Source documentation](https://creativecommons.org/licenses/by/4.0/)

The approval/defer decision is limited to this portfolio release and the evidence reviewed. It is not a certification of downstream deployment suitability or an assertion that deferred datasets prohibit redistribution.
