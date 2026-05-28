# External validation dataset — T5a selection gate

**Status**: gate open (2026-05-27). Decision deadline: **2026-06-10**.

T5b (actual external validation run) is BLOCKED until this gate passes.
If no candidate clears all three checks by the deadline, this directory
becomes a `WHY_NOT_FOUND.md` report and T5b is closed without execution —
that is itself a valid honest outcome.

## Candidate datasets

| Dataset | Year | Organism | Label type | License | Notes |
|---|---|---|---|---|---|
| **Stokes et al. 2020** (Cell) | 2020 | *E. coli* MG1655 | binary (OD600 inhibition) | open via Supplementary | **organism mismatch**: E. coli ≠ PA → cross-strain extrapolation, weak |
| **Liu et al. 2023** (Nat. Chem. Biol.) "abaucin" | 2023 | *A. baumannii* | binary | open | also organism mismatch but Gram-negative; carbapenem-resistant focus |
| **Wong et al. 2023** (Nature) | 2023 | *S. aureus* | binary | open | **Gram-positive — reject for PA validation** |
| **PA-specific ChEMBL pull (non-COADD assays)** | varies | P. aeruginosa | varies | open via ChEMBL | best organism match; require curation (heterogeneous assay protocols) |
| **CO-ADD recent batches (post-tutorial cutoff)** | 2024+ | P. aeruginosa | binary | academic via ChEMBL | direct temporal split — strongest match if available |

## Gate checklist (ALL three must PASS)

### G1 — Organism / strain consistency
- [ ] Direct *P. aeruginosa* measurement, OR
- [ ] Gram-negative cross-strain measurement with **explicit justification document** linking the cross-strain MIC distribution to PA-relevant target biology

### G2 — Label-definition consistency
- [ ] hit-call threshold comparable to CO-ADD PA rule (`max_inhib ≥ 50 OR MIC ≤ 16 µg/mL OR ≤ 40_000 nM`), OR
- [ ] re-labelling procedure documented to harmonize (e.g. re-binarize external MICs at the same thresholds)

### G3 — License & redistribution
- [ ] Confirmed academic-use OK
- [ ] Redistribution status documented (in-repo data file OK? or download-only?)
- [ ] Citation provided in this README

## Decision matrix template

When the gate is evaluated, fill this table and set GO/NO-GO:

```
Candidate: ____________________
G1 (organism)  : PASS / FAIL  — note: __________
G2 (labels)    : PASS / FAIL  — note: __________
G3 (license)   : PASS / FAIL  — note: __________
Overall        : GO / NO-GO
Decided by     : __________ on 2026-__-__
```

## If NO-GO

Rename this file `WHY_NOT_FOUND.md` and add:
- Which candidates were attempted, in what order
- Specific reason each failed which gate
- One paragraph on whether a different evaluation strategy (e.g. temporal split within CO-ADD itself) is more honest than forcing a mismatched external set

The negative result is the deliverable. Do not ship T5b on a candidate that
fails any of G1/G2/G3 — that would silently inject a task-mismatch metric
into the tutorial.
