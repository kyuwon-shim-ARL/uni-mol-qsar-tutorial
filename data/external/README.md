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

---

# External sets for other targets (cancer-protein extension)

The tutorial extension to single-target cancer protein QSAR (see
`docs/EXPLAINABILITY_SCOPE.md`) added its own external-set fetcher.

## `braf_bindingdb.csv` — BRAF (UniProt P15056) from BindingDB

435 molecules, 41.4% active at IC50 ≤ 10 nM. Reproduce:

```bash
python scripts/download_bindingdb_target.py \
    --uniprot P15056 \
    --name braf_bindingdb \
    --subtract data/raw/braf_per_molecule.csv
```

87.9% of BindingDB BRAF molecules overlap with the ChEMBL training set
(both curate from the same primary literature); `--subtract` removes the
overlap by canonical SMILES match.

The gate-checklist framework above (G1/G2/G3) does not apply to this set
in its original form (cross-strain organism check is not relevant for a
single-protein target). For BRAF the analogous gates are:

- **G1' — Target consistency**: same UniProt accession (P15056) ✓
- **G2' — Label consistency**: same pchembl ≥ 8 / IC50 ≤ 10 nM cutoff ✓
  (BindingDB's affinity field is converted to pchembl-equivalent before
  thresholding)
- **G3' — License**: BindingDB academic-use OK; redistribution requires
  attribution (citation in any downstream publication)

## Known external-fetch risks

1. **BindingDB REST response key typo** — live API returns
   `"getLindsByUniprotsResponse"` (missing "ig"). Script handles both.
2. **Non-canonical SMILES** — BindingDB SMILES are canonicalized
   in-script via RDKit before deduplication.
3. **Affinity-type heterogeneity** — BindingDB mixes Ki/Kd/IC50/EC50.
   Script takes min nM across types per molecule.

## What `--subtract` does NOT remove

Near-analogues (Tanimoto > 0.9 to training set) are kept. The external
set is therefore "compounds not in training but possibly close to it" —
the realistic regime, not a synthetic adversarial split.
