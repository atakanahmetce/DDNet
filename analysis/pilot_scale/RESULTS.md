# Pilot 1: does the phase-1 picture hold at scale without hub-biased drug selection?

*Run 2026-09-25 on a shared 4-core box. The DDNet repo was only read. Everything here is under `phase2/pilot1/`. Total compute was about 40 min wall time; for part of it another job on the box was using about 1.5 cores.*

## What was run

| Script | What it does | Output |
|---|---|---|
| `p1_prep.py` | Builds unordered DrugBank pairs; ECFP4 (radius 2, 1024 bits, RDKit 2026.03), 4 RDKit descriptors and target sets from `drugs.json`; draws the drug samples. | `out/prep.npz`, `out/samples.json`, `out/unordered_pairs_global.npy`, `logs/prep.log` |
| `p1_eval.py SAMPLE` | Warm, S1 and S2 evaluation of every method below. | `out/eval_<SAMPLE>.json` (per-fold AUROC, AP, stratified AUROC, Spearman ρ) |
| `p1_beyond_pop.py SAMPLE` | Diagnostic: do ECFP4 or proxies add anything on top of the out-of-sample DrugBank degree? | `out/beyond_pop_<SAMPLE>.json` |
| `p1_proxy_corr.py` | Drug-level check: how well do the proxies track DrugBank degree? | `out/proxy_corr.json` |
| `p1_tables.py` | Builds the tables. | `out/compact.md`, `out/tables.md` (both appended below) |

**Data.**
- 2,768,191 DrugBank rows collapse to 1,384,419 unordered pairs over 4,418 drugs.
- 3,619 of these drugs have SMILES in `drugs.json`. RDKit could not parse one, which leaves a **pool of 3,618**.
- Global degree: median 578 over the 4,418 drugs. The top quartile starts at degree 969. The pool median is higher, at 720, because the drugs that have SMILES are better connected.

**Samples** (seeded `RandomState`):

| Sample | Seed | How drawn | Prevalence |
|---|---|---|---|
| **U1000** | 0 | Uniform, 1,000 drugs from the pool | 0.165 |
| **U300** | 2 | Uniform, 300 drugs (same size as H300, so size and hub bias can be separated) | 0.181 |
| **H300** | 1 | Hub-biased: 300 drugs drawn uniformly from the 1,088 pool drugs in the top global-degree quartile | **0.704** |

- H300's median degree is at the 88th percentile, against 54–59 for the uniform samples. Phase-1 D1 was at the 82nd–85th percentile with 58% density.
- `U300b` and `U300c` (seeds 3 and 4) were drawn but **not evaluated**.

**Protocol.**
- Each unordered pair {a,b} with a ≠ b is one example. Positive means DrugBank lists the pair; negative means any other pair inside the sample.
- **warm**: `KFold(5, shuffle, seed 0)` over pairs.
- **S1/S2**: `KFold(5, shuffle, seed 0)` over drugs. Train on pairs with both drugs in the training drugs. S1 is pairs with exactly one held-out drug; S2 is pairs with both held out.
- Every label-derived quantity uses **training pairs only**.
- Metrics are AUROC and average precision (AP) from scores, reported as mean ± sd (ddof = 1) over 5 folds.
- U1000 node folds 0–3 ran in one process. Node fold 4 ran in a second process with identical seeds (`out/eval_U1000_nodefold4.json`), and the two are merged in `p1_tables.py`.

**Methods** (prefix = family in the tables):

- **B: degree prior.** Per-drug positive rate among training pairs. Score = product (or sum). An unseen drug gets the training mean.
- **B: identity one-hot + LR.** Warm only; this makes it an additive per-drug popularity model.
- **S: ECFP4 + HGB.**
  - Features: `[fp_a+fp_b, Tanimoto]`, plus target count (`+ tc`) or target count and descriptors (`+ proxies`).
  - HGB settings: 150 iterations, learning rate 0.1, 31 leaves, no early stopping, `random_state` 0.
  - **Deviation 1:** `fp_a*fp_b` is dropped. For trees it is exactly `1[fp_a+fp_b = 2]`, so it is redundant. Dropping it halves the cost.
  - **Deviation 2:** 150 iterations instead of 200.
  - Check: on 4 H300 warm folds, 2,049 features × 200 iterations gave ECFP AUROC 0.860 against 0.852 here. The ECFP-free proxy model moved by the same amount (0.807 against 0.797), so the gap comes from the iteration count, not the dropped block.
  - **Deviation 3:** every HGB in a fold uses the same seeded subsample of **at most 50,000 training rows**. This affects U1000 only (warm 399,600 and node 319,600 train rows); the 300-drug samples use all rows.
- **N: Vilar-style propagation** (Tanimoto, training positives only).
  - `max`: max(max over c ∈ N(a) of T(b,c), max over c ∈ N(b) of T(a,c)).
  - `mean`: the same, averaged over the union of partners.
  - It is 0.5 by construction in S2, since neither drug has known partners.
- **N: SimKNN.** Two-sided k = 10 Tanimoto-neighbour smoothing of the training label matrix, `K = W·Y·Wᵀ / W·M·Wᵀ`, as in phase 1. It works in S2.
- **P: drug-side popularity proxies** (available for a new drug), each fed to HGB as the lossless symmetric pair encoding [min, max]:
  - number of target accessions (tc);
  - MW, Crippen logP, rotatable bonds and ring count, each alone and all 4 together;
  - tc + 4 descriptors ("proxies").
- **T: shared targets.** Number of shared accessions and Jaccard of the target sets, raw and via HGB, alone and added to the proxies.
- **U: raw scores.** Tanimoto, (tc_a+1)(tc_b+1), shared-target count and Jaccard.
- **[diag] external DrugBank degree.** Each drug's number of DrugBank partners *outside the sample*. It uses no within-sample label, but a genuinely new drug would not have it. It measures how much of the task is popularity; it is **not a method**.
- **Degree-stratified AUROC (sAUROC).** Positive–negative comparisons are made only within the same stratum. A stratum is an unordered pair of external-degree quintiles, giving 15 strata. It is coarse: the external-degree product itself still reaches about 0.63 within strata in the uniform samples.

## Answers

### 1) With uniform sampling, how large is the warm → S1 → S2 drop?

It is large, and the best cold-start models end far below warm. U1000 AUROC (AP, prevalence 0.165):

| U1000 | warm | S1 | S2 |
|---|---|---|---|
| ECFP4 (HGB) | 0.873 (0.603) | 0.774 (0.399) | 0.632 (0.258) |
| ECFP4 + proxies (HGB) | 0.882 (0.627) | 0.798 (0.439) | 0.674 (0.301) |
| SimKNN k=10 | 0.850 (0.587) | 0.791 (0.471) | **0.687** (0.337) |
| Degree prior (train labels) | **0.901** (0.653) | 0.781 (0.364) | 0.500 |
| Identity one-hot LR | 0.898 (0.653) | – | – |
| [diag] external DrugBank degree product | 0.900 (0.646) | 0.899 (0.645) | **0.900** (0.652) |

- **Warm is solved by popularity.** No learned model beats the train-label degree prior (0.901) or additive identity LR (0.898).
  - ECFP HGB, trained on 50k of the 400k rows, reaches 0.873.
  - The phase-1 finding "warm = memorised popularity/identity" holds at scale and without hub selection.
- **Warm → S2 drop.**
  - Best method per split: about −0.21 AUROC (0.90 to 0.69).
  - ECFP HGB alone: −0.24 (0.873 to 0.632).
  - This is the same S2 level as phase-1 D1/MP, where ECFP4 RF reached 0.63–0.66.
  - S2 fold sd is 0.02–0.04 for U1000, against about 0.05 for U300.
- **Where the gap comes from.** The out-of-sample DrugBank degree scores **0.90 AUROC in S2 (AP 0.65, 3.9× prevalence)**, flat across all three splits.
  - The whole cold-start gap is therefore the model's inability to estimate how "popular" (interaction-rich) a new drug is. If that one number per drug were known, S2 would score like warm.
- **U300 behaves like U1000 but noisier.**
  - U300 warm: degree prior 0.902; ECFP 0.917 (with all 36k training rows).
  - U300 S2: ECFP 0.599 ± 0.051; SimKNN 0.668 ± 0.038.

### 2) How much of S1/S2 AUROC do drug-side popularity proxies give, compared with ECFP structure?

Most of it. Share of ECFP-HGB's above-chance AUROC, (AUROC − 0.5)/(AUROC_ECFP − 0.5), from table T5:

| | U1000 S1 | U1000 S2 | U300 S1 | U300 S2 | H300 S1 | H300 S2 |
|---|---|---|---|---|---|---|
| proxies (tc + 4 descriptors, HGB), no fingerprint | 0.736 → **86%** | 0.616 → **88%** | 0.755 → 93% | 0.582 → 83% | 0.704 → 86% | 0.598 → 93% |
| target count alone (HGB) | 0.638 → 50% | 0.620 → **91%** | 0.644 → 53% | 0.601 → 102% | 0.589 → 37% | 0.558 → 55% |
| 4 descriptors alone (HGB) | 0.708 → 76% | 0.559 → 44% | 0.731 → 84% | 0.528 → 28% | 0.688 → 79% | 0.583 → 78% |

- **Target count needs no model and no training in S2.** In U1000 S2 the untrained (tc_a+1)(tc_b+1) product scores **0.638**, which is ≥ ECFP4-HGB at 0.632.
  - In `drugs.json`, 38% of pool drugs have 0 accessions. Their median DrugBank degree is 362, against 804 for drugs with ≥1 accession.
  - The drug-level Spearman correlation between target count and DrugBank degree is 0.37 (`out/proxy_corr.json`).
  - Target count is mostly a proxy for annotation depth or how well a drug is studied.
- **In S1 the seen drug's own training degree is enough.** The degree prior gives 0.781 against ECFP 0.774 (U1000) and 0.776 against 0.774 (U300).
- **Descriptors behave differently by split.**
  - Continuous descriptors are quasi-unique per drug, so in **warm** an HGB on MW alone acts as an identity code: U300 0.871, U1000 0.716, 4 descriptors 0.899 in U300.
  - In **S2** they carry little transferable signal (0.53–0.58).
  - Their high warm and S1 numbers should therefore not be read as "physicochemistry predicts DDI".
- **ECFP HGB also leans on popularity.** Its S1 scores correlate with the external-degree product at Spearman ρ = 0.57 (U1000), and in S2 at ρ = 0.24. Its within-stratum AUROC in S2 is only 0.546 (U1000), against 0.634 for the external-degree product itself.

### 3) Does the hub-biased sample reproduce the inflated picture?

**Yes, for the parts that were inflated in 2022, but not by raising AUROC.**

- **Prevalence and AP.** H300 prevalence is 0.704, against 0.165–0.181 for the uniform samples; phase-1 D1/D2 had 0.58 and MP 0.45.
  - AP looks excellent everywhere: ECFP AP 0.928 warm and 0.771 S2. Even the raw Tanimoto score gets AP 0.75.
  - The lift over prevalence is small: 1.32× warm and 1.08× S2. For U1000 it is 3.7× and 1.6×.
  - Hard-label or AP-type metrics are therefore inflated by the hub selection alone.
- **The popularity baselines collapse.** Among hubs, degree varies much less, so it stops discriminating:
  - degree prior warm 0.741 (U1000: 0.901);
  - identity LR 0.742;
  - external-degree product 0.726 (U1000: 0.900).
- **Structure therefore looks useful.** In H300 warm, ECFP HGB (0.853) and SimKNN (0.807) beat the degree prior by +0.11 and +0.07.
  - In the uniform samples the order reverses: ECFP 0.873 < degree 0.901 in U1000.
  - This reproduces the phase-1 D1 picture (DDNet RF 0.906 against degree 0.815) and explains it: hub selection hides the degree shortcut rather than removing it.
- **AUROC is not inflated.** H300 AUROCs are lower than the uniform ones (ECFP warm 0.853 against 0.873–0.917).
  - Under cold start all samples converge to about 0.60–0.69: ECFP S2 is H300 0.606, U300 0.599 and U1000 0.632.
  - The hub-biased S2 is *not* easier; it only looks better on AP because of prevalence.

### 4) Is there any signal that structure or network features add beyond the popularity proxies?

Paired per-fold differences, from table T4.

- **Yes, beyond the weak proxies available for new drugs.**
  - ECFP4 + proxies − proxies in U1000: **+0.061 AUROC in S1** and **+0.058 in S2**, positive in 5/5 folds for both. AP gains are +0.084 (S1) and +0.060 (S2).
  - Smaller and less consistent elsewhere: U300 +0.031 (5/5) / +0.022 (4/5); H300 +0.050 (5/5) / +0.030 (4/5).
  - SimKNN − proxies: U1000 **+0.055 S1** and **+0.071 S2** (5/5 each).
  - SimKNN (0.687) is the best leakage-free S2 method in U1000 and the only one whose within-stratum AUROC (0.623) approaches the external degree's residual (0.634).
- **The proxies also add to ECFP.**
  - ECFP4 + tc − ECFP4 in U1000: +0.016 in S1 and **+0.039 in S2**, 5/5 folds each. The fingerprint model does not fully recover the annotation-depth signal.
  - The full proxy set gives +0.024 (S1) and +0.042 (S2).
- **Shared-target features add about nothing.** Proxies + shared targets − proxies is +0.003 in S1 and +0.004 in S2 (U1000); raw Jaccard scores 0.51–0.53.
- **Beyond the true popularity signal the gain is small but consistent in the uniform samples, and larger among hubs.**
  - HGB on the out-of-sample DrugBank degree alone ([diag], min/max of log degree) scores **U1000 S2 0.901 ± 0.008**.
  - Adding ECFP4 gives 0.905 (**ΔAUROC +0.005 ± 0.002, ΔAP +0.019 ± 0.010, 5/5 folds**).
  - In S1: +0.009 AUROC and +0.033 AP (5/5).
  - U300 shows the same size of gain but less consistently: S2 +0.005 AUROC (3/5), +0.019 AP (5/5).
  - H300, where degree barely varies, gains more: S1 +0.065, S2 +0.038 AUROC (5/5).
  - Naively rank-averaging SimKNN with the degree product *lowers* S2 AUROC in the uniform samples (U1000 0.861, U300 0.842, against about 0.90), but raises it in H300 (0.744 against 0.720).
- **Within-degree-strata AUROC (T3) agrees.**
  - Warm label propagation carries real signal beyond popularity: Vilar-max sAUROC 0.740 and SimKNN 0.764 in U1000, against 0.636 for external degree.
  - Under S2, SimKNN keeps 0.623. ECFP-HGB keeps only 0.546; ECFP + proxies 0.568.
- **Net.** Structure carries a real but small residual signal on top of popularity: about +0.005–0.01 AUROC or +0.02–0.03 AP in the uniform samples.
  - It looks large (+0.04–0.07) only when the sample removes degree variance (hub selection) or when the popularity estimate is weak (proxies).

## Caveats and what was not run

- **One sample per design.** U300b and U300c were drawn but not evaluated, and there was no repeated U1000 draw. The U300 S2 sd (≈0.05) shows that a single 300-drug sample is noisy.
- **U1000 HGB models saw at most 50k of the 320–400k training rows.**
  - Warm ECFP is therefore probably underestimated.
  - The S1/S2 comparisons are fair, because every HGB in a fold shares the same subsample.
  - No hyperparameter tuning; k = 10 for SimKNN is untuned.
- **Not re-run here.** No DDNet node2vec or path features (phase 1 showed they equal ECFP under cold start), no FFNN/GNN, no hard-label metrics.
- **The external DrugBank degree is from the same DrugBank snapshot** that is committed in the repo (release date unknown). It is a diagnostic ceiling. Whether a new drug's DrugBank degree reflects real pharmacology (e.g. CYP metabolism) or curation depth was **not tested**.
- **"Accessions" in CROssBAR v1 `drugs.json`** are used as target counts. Whether they include enzymes, transporters or carriers was **not verified**.
- **Degree strata are coarse** (quintiles). Residual popularity inside a stratum remains: the external degree still has sAUROC ≈ 0.63.
- **Fold-wise paired differences only.** Folds share training data, so the "k/5 folds" counts are indicative, not formal tests. No DeLong or bootstrap tests.
- **The beyond-popularity diagnostic uses information a new drug does not have.** It tests whether structure adds anything on top of popularity; it is not a deployable model.

## What this means for the paper

- **The phase-1 picture holds at scale.** It was not an artefact of 179–289 hub drugs:
  - popularity or identity solves warm;
  - cold-start S2 falls to about 0.63–0.69 AUROC;
  - hub selection inflates AP and hides the degree shortcut.
- **The strongest, cheapest message for a moderate-effort evaluation paper** (direction A): in uniform cold-start DrugBank DDI,
  - one popularity number per drug (DrugBank degree to drugs outside the benchmark) gives AUROC 0.90;
  - proxies a new drug does have (target-annotation count, untrained) match the ECFP model in S2;
  - structure adds about +0.04–0.06 AUROC on top of those proxies, but only about +0.005–0.01 AUROC (+0.02–0.03 AP) on top of true popularity.
- **Novelty caveat.** Generic degree-bias findings are pre-empted (Zietz 2024, Aiyappa 2025, Bonner 2022). The DDI-specific, cold-start, annotation-depth angle and the "hub selection hides the shortcut" demonstration are the contribution.
- **Next pilots:**
  - repeat on DDI-Ben or ogbl-ddi splits and on a time split;
  - test pharmacology-grounded proxies (CYP/transporter annotations from CROssBAR) against curation-depth proxies;
  - add 2–3 repeated uniform draws.

---

## Headline tables (generated by p1_tables.py -> out/compact.md)

### AUROC (mean ± std over folds; prevalence: U1000 0.165, U300 0.181, H300 0.704)

| Method | U1000 warm | U1000 S1 | U1000 S2 | U300 warm | U300 S1 | U300 S2 | H300 warm | H300 S1 | H300 S2 |
|---|---|---|---|---|---|---|---|---|---|
| B: degree prior, train labels (product) | 0.901 ± 0.001 | 0.781 ± 0.004 | 0.500 ± 0.000 | 0.902 ± 0.004 | 0.776 ± 0.018 | 0.500 ± 0.000 | 0.741 ± 0.004 | 0.665 ± 0.014 | 0.500 ± 0.000 |
| B: identity one-hot LR | 0.898 ± 0.002 | – | – | 0.899 ± 0.004 | – | – | 0.742 ± 0.004 | – | – |
| [diag] external DrugBank degree product | 0.900 ± 0.001 | 0.899 ± 0.002 | 0.900 ± 0.008 | 0.902 ± 0.004 | 0.901 ± 0.008 | 0.897 ± 0.024 | 0.726 ± 0.008 | 0.727 ± 0.007 | 0.720 ± 0.016 |
| U: Tanimoto ECFP4 (raw) | 0.571 ± 0.002 | 0.572 ± 0.014 | 0.568 ± 0.035 | 0.574 ± 0.006 | 0.573 ± 0.008 | 0.584 ± 0.022 | 0.551 ± 0.005 | 0.549 ± 0.006 | 0.557 ± 0.043 |
| U: target-count product (raw) | 0.637 ± 0.001 | 0.637 ± 0.009 | 0.638 ± 0.026 | 0.639 ± 0.005 | 0.640 ± 0.011 | 0.641 ± 0.027 | 0.595 ± 0.007 | 0.597 ± 0.008 | 0.581 ± 0.032 |
| N: Vilar max-Tanimoto to partners | 0.838 ± 0.003 | 0.812 ± 0.005 | 0.500 ± 0.000 | 0.809 ± 0.003 | 0.781 ± 0.019 | 0.500 ± 0.000 | 0.695 ± 0.008 | 0.657 ± 0.016 | 0.500 ± 0.000 |
| N: Vilar mean-Tanimoto to partners | 0.636 ± 0.002 | 0.661 ± 0.023 | 0.500 ± 0.000 | 0.621 ± 0.007 | 0.657 ± 0.027 | 0.500 ± 0.000 | 0.588 ± 0.005 | 0.570 ± 0.013 | 0.500 ± 0.000 |
| N: SimKNN k=10 (2-sided) | 0.850 ± 0.001 | 0.791 ± 0.012 | 0.687 ± 0.035 | 0.844 ± 0.006 | 0.775 ± 0.017 | 0.668 ± 0.038 | 0.807 ± 0.008 | 0.748 ± 0.010 | 0.647 ± 0.023 |
| P: target count (HGB) | 0.652 ± 0.002 | 0.638 ± 0.011 | 0.620 ± 0.034 | 0.675 ± 0.003 | 0.644 ± 0.017 | 0.601 ± 0.040 | 0.605 ± 0.006 | 0.589 ± 0.009 | 0.558 ± 0.023 |
| P: 4 descriptors (HGB) | 0.814 ± 0.003 | 0.708 ± 0.010 | 0.559 ± 0.013 | 0.899 ± 0.004 | 0.731 ± 0.039 | 0.528 ± 0.065 | 0.792 ± 0.002 | 0.688 ± 0.008 | 0.583 ± 0.014 |
| P: proxies = tc+4 desc (HGB) | 0.821 ± 0.002 | 0.736 ± 0.014 | 0.616 ± 0.029 | 0.899 ± 0.003 | 0.755 ± 0.019 | 0.582 ± 0.047 | 0.799 ± 0.005 | 0.704 ± 0.007 | 0.598 ± 0.022 |
| T: proxies + shared targets (HGB) | 0.822 ± 0.002 | 0.739 ± 0.014 | 0.620 ± 0.028 | 0.899 ± 0.003 | 0.756 ± 0.015 | 0.582 ± 0.050 | 0.803 ± 0.004 | 0.711 ± 0.011 | 0.608 ± 0.027 |
| S: ECFP4 (HGB) | 0.873 ± 0.002 | 0.774 ± 0.008 | 0.632 ± 0.028 | 0.917 ± 0.004 | 0.774 ± 0.031 | 0.599 ± 0.051 | 0.853 ± 0.007 | 0.738 ± 0.012 | 0.606 ± 0.022 |
| S: ECFP4 + tc (HGB) | 0.875 ± 0.002 | 0.790 ± 0.008 | 0.671 ± 0.024 | 0.918 ± 0.002 | 0.785 ± 0.025 | 0.621 ± 0.046 | 0.854 ± 0.009 | 0.746 ± 0.017 | 0.617 ± 0.031 |
| S: ECFP4 + proxies (HGB) | 0.882 ± 0.002 | 0.798 ± 0.008 | 0.674 ± 0.025 | 0.930 ± 0.003 | 0.786 ± 0.029 | 0.604 ± 0.055 | 0.866 ± 0.005 | 0.754 ± 0.020 | 0.629 ± 0.032 |

### AP (mean ± std over folds; prevalence: U1000 0.165, U300 0.181, H300 0.704)

| Method | U1000 warm | U1000 S1 | U1000 S2 | U300 warm | U300 S1 | U300 S2 | H300 warm | H300 S1 | H300 S2 |
|---|---|---|---|---|---|---|---|---|---|
| B: degree prior, train labels (product) | 0.653 ± 0.005 | 0.364 ± 0.010 | 0.166 ± 0.007 | 0.683 ± 0.014 | 0.387 ± 0.059 | 0.188 ± 0.084 | 0.879 ± 0.005 | 0.825 ± 0.009 | 0.711 ± 0.015 |
| B: identity one-hot LR | 0.653 ± 0.005 | – | – | 0.686 ± 0.014 | – | – | 0.880 ± 0.004 | – | – |
| [diag] external DrugBank degree product | 0.646 ± 0.005 | 0.645 ± 0.008 | 0.652 ± 0.028 | 0.675 ± 0.015 | 0.667 ± 0.028 | 0.659 ± 0.099 | 0.865 ± 0.006 | 0.866 ± 0.006 | 0.863 ± 0.017 |
| U: Tanimoto ECFP4 (raw) | 0.213 ± 0.002 | 0.213 ± 0.007 | 0.212 ± 0.024 | 0.230 ± 0.005 | 0.227 ± 0.027 | 0.240 ± 0.087 | 0.751 ± 0.004 | 0.749 ± 0.004 | 0.760 ± 0.033 |
| U: target-count product (raw) | 0.251 ± 0.002 | 0.251 ± 0.008 | 0.258 ± 0.024 | 0.257 ± 0.006 | 0.258 ± 0.029 | 0.273 ± 0.083 | 0.779 ± 0.009 | 0.779 ± 0.005 | 0.774 ± 0.027 |
| N: Vilar max-Tanimoto to partners | 0.476 ± 0.007 | 0.446 ± 0.021 | 0.166 ± 0.007 | 0.469 ± 0.009 | 0.440 ± 0.030 | 0.188 ± 0.084 | 0.838 ± 0.008 | 0.815 ± 0.013 | 0.711 ± 0.015 |
| N: Vilar mean-Tanimoto to partners | 0.232 ± 0.002 | 0.244 ± 0.016 | 0.166 ± 0.007 | 0.241 ± 0.008 | 0.250 ± 0.016 | 0.188 ± 0.084 | 0.773 ± 0.006 | 0.755 ± 0.014 | 0.711 ± 0.015 |
| N: SimKNN k=10 (2-sided) | 0.587 ± 0.003 | 0.471 ± 0.036 | 0.337 ± 0.047 | 0.584 ± 0.010 | 0.444 ± 0.055 | 0.319 ± 0.131 | 0.904 ± 0.006 | 0.865 ± 0.006 | 0.805 ± 0.013 |
| P: target count (HGB) | 0.280 ± 0.003 | 0.256 ± 0.010 | 0.236 ± 0.025 | 0.326 ± 0.009 | 0.286 ± 0.039 | 0.252 ± 0.114 | 0.792 ± 0.008 | 0.778 ± 0.009 | 0.759 ± 0.022 |
| P: 4 descriptors (HGB) | 0.482 ± 0.005 | 0.315 ± 0.008 | 0.196 ± 0.009 | 0.680 ± 0.012 | 0.352 ± 0.082 | 0.211 ± 0.081 | 0.898 ± 0.003 | 0.832 ± 0.008 | 0.762 ± 0.013 |
| P: proxies = tc+4 desc (HGB) | 0.499 ± 0.005 | 0.354 ± 0.020 | 0.240 ± 0.026 | 0.687 ± 0.011 | 0.383 ± 0.071 | 0.240 ± 0.094 | 0.902 ± 0.005 | 0.845 ± 0.003 | 0.785 ± 0.016 |
| T: proxies + shared targets (HGB) | 0.507 ± 0.004 | 0.374 ± 0.017 | 0.262 ± 0.024 | 0.686 ± 0.012 | 0.393 ± 0.067 | 0.246 ± 0.086 | 0.905 ± 0.004 | 0.853 ± 0.007 | 0.797 ± 0.019 |
| S: ECFP4 (HGB) | 0.603 ± 0.008 | 0.399 ± 0.014 | 0.258 ± 0.021 | 0.736 ± 0.010 | 0.437 ± 0.071 | 0.268 ± 0.117 | 0.928 ± 0.005 | 0.857 ± 0.009 | 0.771 ± 0.025 |
| S: ECFP4 + tc (HGB) | 0.611 ± 0.005 | 0.428 ± 0.021 | 0.297 ± 0.030 | 0.737 ± 0.009 | 0.446 ± 0.073 | 0.284 ± 0.104 | 0.929 ± 0.005 | 0.863 ± 0.009 | 0.781 ± 0.024 |
| S: ECFP4 + proxies (HGB) | 0.627 ± 0.006 | 0.439 ± 0.020 | 0.301 ± 0.027 | 0.772 ± 0.012 | 0.440 ± 0.081 | 0.271 ± 0.103 | 0.936 ± 0.002 | 0.870 ± 0.011 | 0.794 ± 0.024 |

### Beyond the strongest popularity signal ([diag] out-of-sample DrugBank degree; p1_beyond_pop.py), AUROC / AP, mean ± std over node folds

| Method | U1000 S1 (folds) | U1000 S2 (folds) | U300 S1 (folds) | U300 S2 (folds) | H300 S1 (folds) | H300 S2 (folds) |
|---|---|---|---|---|---|---|
| ext-degree (HGB) | 0.903 ± 0.002 / 0.648 ± 0.007 (5) | 0.901 ± 0.008 / 0.645 ± 0.028 (5) | 0.910 ± 0.010 / 0.687 ± 0.034 (5) | 0.886 ± 0.023 / 0.625 ± 0.093 (5) | 0.742 ± 0.021 / 0.873 ± 0.011 (5) | 0.703 ± 0.019 / 0.854 ± 0.016 (5) |
| ext-degree + proxies (HGB) | 0.909 ± 0.002 / 0.672 ± 0.009 (5) | 0.903 ± 0.008 / 0.656 ± 0.031 (5) | 0.913 ± 0.013 / 0.704 ± 0.016 (5) | 0.890 ± 0.022 / 0.642 ± 0.077 (5) | 0.778 ± 0.018 / 0.892 ± 0.010 (5) | 0.727 ± 0.018 / 0.867 ± 0.011 (5) |
| ext-degree + ECFP4 (HGB) | 0.912 ± 0.003 / 0.681 ± 0.009 (5) | 0.905 ± 0.009 / 0.664 ± 0.028 (5) | 0.915 ± 0.009 / 0.706 ± 0.023 (5) | 0.890 ± 0.018 / 0.644 ± 0.093 (5) | 0.806 ± 0.012 / 0.906 ± 0.007 (5) | 0.741 ± 0.018 / 0.874 ± 0.010 (5) |
| ext-degree product (raw) | 0.899 ± 0.002 / 0.645 ± 0.008 (5) | 0.900 ± 0.008 / 0.652 ± 0.028 (5) | 0.901 ± 0.008 / 0.667 ± 0.028 (5) | 0.897 ± 0.024 / 0.659 ± 0.099 (5) | 0.727 ± 0.007 / 0.866 ± 0.006 (5) | 0.720 ± 0.016 / 0.863 ± 0.017 (5) |
| rank-avg(ext-degree product, SimKNN) | 0.900 ± 0.006 / 0.677 ± 0.016 (5) | 0.861 ± 0.018 / 0.593 ± 0.037 (5) | 0.883 ± 0.007 / 0.652 ± 0.020 (5) | 0.842 ± 0.024 / 0.571 ± 0.100 (5) | 0.799 ± 0.007 / 0.901 ± 0.006 (5) | 0.744 ± 0.020 / 0.872 ± 0.014 (5) |

### Paired per-fold gain over HGB on out-of-sample degree alone (mean ± sd; #folds > 0)

| sample | split | +ECFP4 ΔAUROC | +ECFP4 ΔAP | +proxies ΔAUROC | +proxies ΔAP |
|---|---|---|---|---|---|
| U1000 | S1 | +0.009 ± 0.001 (5/5) | +0.033 ± 0.005 (5/5) | +0.006 ± 0.001 (5/5) | +0.024 ± 0.003 (5/5) |
| U1000 | S2 | +0.005 ± 0.002 (5/5) | +0.019 ± 0.010 (5/5) | +0.003 ± 0.001 (5/5) | +0.011 ± 0.005 (5/5) |
| U300 | S1 | +0.005 ± 0.005 (4/5) | +0.019 ± 0.017 (4/5) | +0.004 ± 0.009 (4/5) | +0.016 ± 0.024 (4/5) |
| U300 | S2 | +0.005 ± 0.007 (3/5) | +0.019 ± 0.005 (5/5) | +0.004 ± 0.006 (3/5) | +0.016 ± 0.017 (4/5) |
| H300 | S1 | +0.065 ± 0.013 (5/5) | +0.033 ± 0.006 (5/5) | +0.036 ± 0.009 (5/5) | +0.019 ± 0.004 (5/5) |
| H300 | S2 | +0.038 ± 0.008 (5/5) | +0.020 ± 0.009 (5/5) | +0.024 ± 0.010 (5/5) | +0.013 ± 0.010 (4/5) |

## Full tables (generated by p1_tables.py -> out/tables.md)

### T1. Samples and split sizes (mean over folds)

| Sample | drugs | pairs | prevalence | median out-of-sample DrugBank degree | median #targets | % drugs with 0 targets | folds warm/node | warm test n (prev) | S1 n (prev) | S2 n (prev) | train rows (HGB rows) warm / node |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U1000 | 1000 | 499500 | 0.165 | 492 | 1 | 38 | 5/5 | 99900 (0.165) | 160000 (0.165) | 19900 (0.166) | 399600 (50000) / 319600 (50000) |
| U300 | 300 | 44850 | 0.181 | 669 | 1 | 39 | 5/5 | 8970 (0.181) | 14400 (0.179) | 1770 (0.188) | 35880 (35880) / 28680 (28680) |
| H300 | 300 | 44850 | 0.704 | 1159 | 2 | 18 | 5/5 | 8970 (0.704) | 14400 (0.703) | 1770 (0.711) | 35880 (35880) / 28680 (28680) |

### T2-AUROC. U1000: AUROC, mean ± std over folds

| Method | warm | S1 | S2 |
|---|---|---|---|
| B: degree prior, train labels (product) | 0.901 ± 0.001 | 0.781 ± 0.004 | 0.500 ± 0.000 |
| B: degree prior, train labels (sum) | 0.879 ± 0.001 | 0.781 ± 0.004 | 0.500 ± 0.000 |
| U: Tanimoto ECFP4 (raw) | 0.571 ± 0.002 | 0.572 ± 0.014 | 0.568 ± 0.035 |
| U: target-count product (raw) | 0.637 ± 0.001 | 0.637 ± 0.009 | 0.638 ± 0.026 |
| U: # shared targets (raw) | 0.517 ± 0.001 | 0.517 ± 0.002 | 0.517 ± 0.005 |
| U: Jaccard of target sets (raw) | 0.517 ± 0.001 | 0.517 ± 0.002 | 0.517 ± 0.005 |
| [diag] external DrugBank degree product | 0.900 ± 0.001 | 0.899 ± 0.002 | 0.900 ± 0.008 |
| N: Vilar max-Tanimoto to partners | 0.838 ± 0.003 | 0.812 ± 0.005 | 0.500 ± 0.000 |
| N: Vilar mean-Tanimoto to partners | 0.636 ± 0.002 | 0.661 ± 0.023 | 0.500 ± 0.000 |
| N: SimKNN k=10 (2-sided) | 0.850 ± 0.001 | 0.791 ± 0.012 | 0.687 ± 0.035 |
| B: identity one-hot LR | 0.898 ± 0.002 | – | – |
| P: target count (HGB) | 0.652 ± 0.002 | 0.638 ± 0.011 | 0.620 ± 0.034 |
| P: MW (HGB) | 0.716 ± 0.003 | 0.627 ± 0.010 | 0.515 ± 0.021 |
| P: logP (HGB) | 0.716 ± 0.004 | 0.647 ± 0.009 | 0.546 ± 0.018 |
| P: rot. bonds (HGB) | 0.575 ± 0.001 | 0.560 ± 0.006 | 0.536 ± 0.014 |
| P: ring count (HGB) | 0.566 ± 0.002 | 0.554 ± 0.010 | 0.537 ± 0.025 |
| P: 4 descriptors (HGB) | 0.814 ± 0.003 | 0.708 ± 0.010 | 0.559 ± 0.013 |
| P: proxies = tc+4 desc (HGB) | 0.821 ± 0.002 | 0.736 ± 0.014 | 0.616 ± 0.029 |
| T: shared targets (HGB) | 0.517 ± 0.001 | 0.517 ± 0.002 | 0.517 ± 0.005 |
| T: tc + shared targets (HGB) | 0.654 ± 0.002 | 0.640 ± 0.012 | 0.622 ± 0.035 |
| T: proxies + shared targets (HGB) | 0.822 ± 0.002 | 0.739 ± 0.014 | 0.620 ± 0.028 |
| S: ECFP4 (HGB) | 0.873 ± 0.002 | 0.774 ± 0.008 | 0.632 ± 0.028 |
| S: ECFP4 + tc (HGB) | 0.875 ± 0.002 | 0.790 ± 0.008 | 0.671 ± 0.024 |
| S: ECFP4 + proxies (HGB) | 0.882 ± 0.002 | 0.798 ± 0.008 | 0.674 ± 0.025 |

### T2-AUROC. U300: AUROC, mean ± std over folds

| Method | warm | S1 | S2 |
|---|---|---|---|
| B: degree prior, train labels (product) | 0.902 ± 0.004 | 0.776 ± 0.018 | 0.500 ± 0.000 |
| B: degree prior, train labels (sum) | 0.878 ± 0.005 | 0.776 ± 0.018 | 0.500 ± 0.000 |
| U: Tanimoto ECFP4 (raw) | 0.574 ± 0.006 | 0.573 ± 0.008 | 0.584 ± 0.022 |
| U: target-count product (raw) | 0.639 ± 0.005 | 0.640 ± 0.011 | 0.641 ± 0.027 |
| U: # shared targets (raw) | 0.512 ± 0.002 | 0.513 ± 0.002 | 0.511 ± 0.009 |
| U: Jaccard of target sets (raw) | 0.512 ± 0.002 | 0.513 ± 0.002 | 0.511 ± 0.009 |
| [diag] external DrugBank degree product | 0.902 ± 0.004 | 0.901 ± 0.008 | 0.897 ± 0.024 |
| N: Vilar max-Tanimoto to partners | 0.809 ± 0.003 | 0.781 ± 0.019 | 0.500 ± 0.000 |
| N: Vilar mean-Tanimoto to partners | 0.621 ± 0.007 | 0.657 ± 0.027 | 0.500 ± 0.000 |
| N: SimKNN k=10 (2-sided) | 0.844 ± 0.006 | 0.775 ± 0.017 | 0.668 ± 0.038 |
| B: identity one-hot LR | 0.899 ± 0.004 | – | – |
| P: target count (HGB) | 0.675 ± 0.003 | 0.644 ± 0.017 | 0.601 ± 0.040 |
| P: MW (HGB) | 0.871 ± 0.004 | 0.691 ± 0.050 | 0.487 ± 0.057 |
| P: logP (HGB) | 0.870 ± 0.006 | 0.685 ± 0.041 | 0.487 ± 0.067 |
| P: rot. bonds (HGB) | 0.618 ± 0.009 | 0.580 ± 0.013 | 0.531 ± 0.047 |
| P: ring count (HGB) | 0.602 ± 0.005 | 0.588 ± 0.023 | 0.559 ± 0.079 |
| P: 4 descriptors (HGB) | 0.899 ± 0.004 | 0.731 ± 0.039 | 0.528 ± 0.065 |
| P: proxies = tc+4 desc (HGB) | 0.899 ± 0.003 | 0.755 ± 0.019 | 0.582 ± 0.047 |
| T: shared targets (HGB) | 0.513 ± 0.002 | 0.513 ± 0.002 | 0.512 ± 0.008 |
| T: tc + shared targets (HGB) | 0.676 ± 0.003 | 0.646 ± 0.016 | 0.605 ± 0.041 |
| T: proxies + shared targets (HGB) | 0.899 ± 0.003 | 0.756 ± 0.015 | 0.582 ± 0.050 |
| S: ECFP4 (HGB) | 0.917 ± 0.004 | 0.774 ± 0.031 | 0.599 ± 0.051 |
| S: ECFP4 + tc (HGB) | 0.918 ± 0.002 | 0.785 ± 0.025 | 0.621 ± 0.046 |
| S: ECFP4 + proxies (HGB) | 0.930 ± 0.003 | 0.786 ± 0.029 | 0.604 ± 0.055 |

### T2-AUROC. H300: AUROC, mean ± std over folds

| Method | warm | S1 | S2 |
|---|---|---|---|
| B: degree prior, train labels (product) | 0.741 ± 0.004 | 0.665 ± 0.014 | 0.500 ± 0.000 |
| B: degree prior, train labels (sum) | 0.743 ± 0.004 | 0.665 ± 0.014 | 0.500 ± 0.000 |
| U: Tanimoto ECFP4 (raw) | 0.551 ± 0.005 | 0.549 ± 0.006 | 0.557 ± 0.043 |
| U: target-count product (raw) | 0.595 ± 0.007 | 0.597 ± 0.008 | 0.581 ± 0.032 |
| U: # shared targets (raw) | 0.530 ± 0.002 | 0.531 ± 0.002 | 0.528 ± 0.006 |
| U: Jaccard of target sets (raw) | 0.530 ± 0.002 | 0.531 ± 0.002 | 0.528 ± 0.006 |
| [diag] external DrugBank degree product | 0.726 ± 0.008 | 0.727 ± 0.007 | 0.720 ± 0.016 |
| N: Vilar max-Tanimoto to partners | 0.695 ± 0.008 | 0.657 ± 0.016 | 0.500 ± 0.000 |
| N: Vilar mean-Tanimoto to partners | 0.588 ± 0.005 | 0.570 ± 0.013 | 0.500 ± 0.000 |
| N: SimKNN k=10 (2-sided) | 0.807 ± 0.008 | 0.748 ± 0.010 | 0.647 ± 0.023 |
| B: identity one-hot LR | 0.742 ± 0.004 | – | – |
| P: target count (HGB) | 0.605 ± 0.006 | 0.589 ± 0.009 | 0.558 ± 0.023 |
| P: MW (HGB) | 0.741 ± 0.006 | 0.640 ± 0.014 | 0.527 ± 0.037 |
| P: logP (HGB) | 0.734 ± 0.006 | 0.612 ± 0.007 | 0.496 ± 0.035 |
| P: rot. bonds (HGB) | 0.564 ± 0.009 | 0.554 ± 0.010 | 0.538 ± 0.019 |
| P: ring count (HGB) | 0.608 ± 0.009 | 0.598 ± 0.009 | 0.587 ± 0.028 |
| P: 4 descriptors (HGB) | 0.792 ± 0.002 | 0.688 ± 0.008 | 0.583 ± 0.014 |
| P: proxies = tc+4 desc (HGB) | 0.799 ± 0.005 | 0.704 ± 0.007 | 0.598 ± 0.022 |
| T: shared targets (HGB) | 0.530 ± 0.002 | 0.531 ± 0.002 | 0.527 ± 0.005 |
| T: tc + shared targets (HGB) | 0.612 ± 0.005 | 0.596 ± 0.009 | 0.564 ± 0.024 |
| T: proxies + shared targets (HGB) | 0.803 ± 0.004 | 0.711 ± 0.011 | 0.608 ± 0.027 |
| S: ECFP4 (HGB) | 0.853 ± 0.007 | 0.738 ± 0.012 | 0.606 ± 0.022 |
| S: ECFP4 + tc (HGB) | 0.854 ± 0.009 | 0.746 ± 0.017 | 0.617 ± 0.031 |
| S: ECFP4 + proxies (HGB) | 0.866 ± 0.005 | 0.754 ± 0.020 | 0.629 ± 0.032 |

### T2-AP. U1000: Average precision (compare with prevalence in T1), mean ± std over folds

| Method | warm | S1 | S2 |
|---|---|---|---|
| B: degree prior, train labels (product) | 0.653 ± 0.005 | 0.364 ± 0.010 | 0.166 ± 0.007 |
| B: degree prior, train labels (sum) | 0.634 ± 0.005 | 0.364 ± 0.010 | 0.166 ± 0.007 |
| U: Tanimoto ECFP4 (raw) | 0.213 ± 0.002 | 0.213 ± 0.007 | 0.212 ± 0.024 |
| U: target-count product (raw) | 0.251 ± 0.002 | 0.251 ± 0.008 | 0.258 ± 0.024 |
| U: # shared targets (raw) | 0.187 ± 0.001 | 0.187 ± 0.002 | 0.189 ± 0.007 |
| U: Jaccard of target sets (raw) | 0.184 ± 0.001 | 0.183 ± 0.002 | 0.185 ± 0.007 |
| [diag] external DrugBank degree product | 0.646 ± 0.005 | 0.645 ± 0.008 | 0.652 ± 0.028 |
| N: Vilar max-Tanimoto to partners | 0.476 ± 0.007 | 0.446 ± 0.021 | 0.166 ± 0.007 |
| N: Vilar mean-Tanimoto to partners | 0.232 ± 0.002 | 0.244 ± 0.016 | 0.166 ± 0.007 |
| N: SimKNN k=10 (2-sided) | 0.587 ± 0.003 | 0.471 ± 0.036 | 0.337 ± 0.047 |
| B: identity one-hot LR | 0.653 ± 0.005 | – | – |
| P: target count (HGB) | 0.280 ± 0.003 | 0.256 ± 0.010 | 0.236 ± 0.025 |
| P: MW (HGB) | 0.324 ± 0.004 | 0.238 ± 0.010 | 0.173 ± 0.012 |
| P: logP (HGB) | 0.319 ± 0.007 | 0.252 ± 0.004 | 0.188 ± 0.015 |
| P: rot. bonds (HGB) | 0.197 ± 0.002 | 0.187 ± 0.004 | 0.175 ± 0.010 |
| P: ring count (HGB) | 0.192 ± 0.001 | 0.186 ± 0.006 | 0.178 ± 0.016 |
| P: 4 descriptors (HGB) | 0.482 ± 0.005 | 0.315 ± 0.008 | 0.196 ± 0.009 |
| P: proxies = tc+4 desc (HGB) | 0.499 ± 0.005 | 0.354 ± 0.020 | 0.240 ± 0.026 |
| T: shared targets (HGB) | 0.187 ± 0.001 | 0.187 ± 0.002 | 0.188 ± 0.008 |
| T: tc + shared targets (HGB) | 0.297 ± 0.004 | 0.275 ± 0.009 | 0.255 ± 0.026 |
| T: proxies + shared targets (HGB) | 0.507 ± 0.004 | 0.374 ± 0.017 | 0.262 ± 0.024 |
| S: ECFP4 (HGB) | 0.603 ± 0.008 | 0.399 ± 0.014 | 0.258 ± 0.021 |
| S: ECFP4 + tc (HGB) | 0.611 ± 0.005 | 0.428 ± 0.021 | 0.297 ± 0.030 |
| S: ECFP4 + proxies (HGB) | 0.627 ± 0.006 | 0.439 ± 0.020 | 0.301 ± 0.027 |

### T2-AP. U300: Average precision (compare with prevalence in T1), mean ± std over folds

| Method | warm | S1 | S2 |
|---|---|---|---|
| B: degree prior, train labels (product) | 0.683 ± 0.014 | 0.387 ± 0.059 | 0.188 ± 0.084 |
| B: degree prior, train labels (sum) | 0.662 ± 0.012 | 0.387 ± 0.059 | 0.188 ± 0.084 |
| U: Tanimoto ECFP4 (raw) | 0.230 ± 0.005 | 0.227 ± 0.027 | 0.240 ± 0.087 |
| U: target-count product (raw) | 0.257 ± 0.006 | 0.258 ± 0.029 | 0.273 ± 0.083 |
| U: # shared targets (raw) | 0.195 ± 0.006 | 0.194 ± 0.025 | 0.202 ± 0.083 |
| U: Jaccard of target sets (raw) | 0.196 ± 0.005 | 0.193 ± 0.025 | 0.204 ± 0.083 |
| [diag] external DrugBank degree product | 0.675 ± 0.015 | 0.667 ± 0.028 | 0.659 ± 0.099 |
| N: Vilar max-Tanimoto to partners | 0.469 ± 0.009 | 0.440 ± 0.030 | 0.188 ± 0.084 |
| N: Vilar mean-Tanimoto to partners | 0.241 ± 0.008 | 0.250 ± 0.016 | 0.188 ± 0.084 |
| N: SimKNN k=10 (2-sided) | 0.584 ± 0.010 | 0.444 ± 0.055 | 0.319 ± 0.131 |
| B: identity one-hot LR | 0.686 ± 0.014 | – | – |
| P: target count (HGB) | 0.326 ± 0.009 | 0.286 ± 0.039 | 0.252 ± 0.114 |
| P: MW (HGB) | 0.611 ± 0.006 | 0.320 ± 0.081 | 0.191 ± 0.101 |
| P: logP (HGB) | 0.608 ± 0.005 | 0.295 ± 0.071 | 0.185 ± 0.084 |
| P: rot. bonds (HGB) | 0.246 ± 0.008 | 0.218 ± 0.036 | 0.209 ± 0.090 |
| P: ring count (HGB) | 0.237 ± 0.004 | 0.226 ± 0.023 | 0.218 ± 0.056 |
| P: 4 descriptors (HGB) | 0.680 ± 0.012 | 0.352 ± 0.082 | 0.211 ± 0.081 |
| P: proxies = tc+4 desc (HGB) | 0.687 ± 0.011 | 0.383 ± 0.071 | 0.240 ± 0.094 |
| T: shared targets (HGB) | 0.196 ± 0.006 | 0.193 ± 0.026 | 0.203 ± 0.082 |
| T: tc + shared targets (HGB) | 0.336 ± 0.006 | 0.297 ± 0.039 | 0.257 ± 0.117 |
| T: proxies + shared targets (HGB) | 0.686 ± 0.012 | 0.393 ± 0.067 | 0.246 ± 0.086 |
| S: ECFP4 (HGB) | 0.736 ± 0.010 | 0.437 ± 0.071 | 0.268 ± 0.117 |
| S: ECFP4 + tc (HGB) | 0.737 ± 0.009 | 0.446 ± 0.073 | 0.284 ± 0.104 |
| S: ECFP4 + proxies (HGB) | 0.772 ± 0.012 | 0.440 ± 0.081 | 0.271 ± 0.103 |

### T2-AP. H300: Average precision (compare with prevalence in T1), mean ± std over folds

| Method | warm | S1 | S2 |
|---|---|---|---|
| B: degree prior, train labels (product) | 0.879 ± 0.005 | 0.825 ± 0.009 | 0.711 ± 0.015 |
| B: degree prior, train labels (sum) | 0.880 ± 0.005 | 0.825 ± 0.009 | 0.711 ± 0.015 |
| U: Tanimoto ECFP4 (raw) | 0.751 ± 0.004 | 0.749 ± 0.004 | 0.760 ± 0.033 |
| U: target-count product (raw) | 0.779 ± 0.009 | 0.779 ± 0.005 | 0.774 ± 0.027 |
| U: # shared targets (raw) | 0.721 ± 0.006 | 0.720 ± 0.005 | 0.727 ± 0.018 |
| U: Jaccard of target sets (raw) | 0.721 ± 0.006 | 0.720 ± 0.005 | 0.727 ± 0.018 |
| [diag] external DrugBank degree product | 0.865 ± 0.006 | 0.866 ± 0.006 | 0.863 ± 0.017 |
| N: Vilar max-Tanimoto to partners | 0.838 ± 0.008 | 0.815 ± 0.013 | 0.711 ± 0.015 |
| N: Vilar mean-Tanimoto to partners | 0.773 ± 0.006 | 0.755 ± 0.014 | 0.711 ± 0.015 |
| N: SimKNN k=10 (2-sided) | 0.904 ± 0.006 | 0.865 ± 0.006 | 0.805 ± 0.013 |
| B: identity one-hot LR | 0.880 ± 0.004 | – | – |
| P: target count (HGB) | 0.792 ± 0.008 | 0.778 ± 0.009 | 0.759 ± 0.022 |
| P: MW (HGB) | 0.870 ± 0.005 | 0.802 ± 0.007 | 0.729 ± 0.021 |
| P: logP (HGB) | 0.868 ± 0.004 | 0.783 ± 0.006 | 0.711 ± 0.033 |
| P: rot. bonds (HGB) | 0.746 ± 0.012 | 0.738 ± 0.005 | 0.737 ± 0.017 |
| P: ring count (HGB) | 0.769 ± 0.006 | 0.760 ± 0.004 | 0.762 ± 0.015 |
| P: 4 descriptors (HGB) | 0.898 ± 0.003 | 0.832 ± 0.008 | 0.762 ± 0.013 |
| P: proxies = tc+4 desc (HGB) | 0.902 ± 0.005 | 0.845 ± 0.003 | 0.785 ± 0.016 |
| T: shared targets (HGB) | 0.722 ± 0.006 | 0.721 ± 0.005 | 0.726 ± 0.017 |
| T: tc + shared targets (HGB) | 0.802 ± 0.006 | 0.790 ± 0.008 | 0.773 ± 0.024 |
| T: proxies + shared targets (HGB) | 0.905 ± 0.004 | 0.853 ± 0.007 | 0.797 ± 0.019 |
| S: ECFP4 (HGB) | 0.928 ± 0.005 | 0.857 ± 0.009 | 0.771 ± 0.025 |
| S: ECFP4 + tc (HGB) | 0.929 ± 0.005 | 0.863 ± 0.009 | 0.781 ± 0.024 |
| S: ECFP4 + proxies (HGB) | 0.936 ± 0.002 | 0.870 ± 0.011 | 0.794 ± 0.024 |

### T3. Degree-stratified AUROC (pos-neg pairs compared only within the same stratum of out-of-sample DrugBank degree quintiles; 15 unordered strata) - mean over folds

| Method | U1000 warm | U1000 S1 | U1000 S2 | U300 warm | U300 S1 | U300 S2 | H300 warm | H300 S1 | H300 S2 |
|---|---|---|---|---|---|---|---|---|---|
| B: degree prior, train labels (product) | 0.647 | 0.556 | 0.500 | 0.639 | 0.543 | 0.500 | 0.621 | 0.553 | 0.500 |
| B: degree prior, train labels (sum) | 0.629 | 0.556 | 0.500 | 0.618 | 0.543 | 0.500 | 0.623 | 0.553 | 0.500 |
| U: Tanimoto ECFP4 (raw) | 0.543 | 0.544 | 0.539 | 0.527 | 0.524 | 0.534 | 0.547 | 0.544 | 0.558 |
| U: target-count product (raw) | 0.525 | 0.525 | 0.526 | 0.532 | 0.534 | 0.534 | 0.537 | 0.539 | 0.520 |
| U: # shared targets (raw) | 0.512 | 0.512 | 0.513 | 0.508 | 0.508 | 0.508 | 0.526 | 0.526 | 0.522 |
| U: Jaccard of target sets (raw) | 0.512 | 0.512 | 0.513 | 0.508 | 0.508 | 0.508 | 0.526 | 0.526 | 0.522 |
| [diag] external DrugBank degree product | 0.636 | 0.636 | 0.634 | 0.638 | 0.637 | 0.631 | 0.568 | 0.568 | 0.568 |
| N: Vilar max-Tanimoto to partners | 0.740 | 0.694 | 0.500 | 0.693 | 0.638 | 0.500 | 0.676 | 0.643 | 0.500 |
| N: Vilar mean-Tanimoto to partners | 0.605 | 0.599 | 0.500 | 0.563 | 0.556 | 0.500 | 0.580 | 0.567 | 0.500 |
| N: SimKNN k=10 (2-sided) | 0.764 | 0.703 | 0.623 | 0.695 | 0.634 | 0.567 | 0.781 | 0.724 | 0.642 |
| B: identity one-hot LR | 0.644 | – | – | 0.636 | – | – | 0.623 | – | – |
| P: target count (HGB) | 0.531 | 0.528 | 0.519 | 0.529 | 0.521 | 0.501 | 0.549 | 0.540 | 0.511 |
| P: MW (HGB) | 0.545 | 0.521 | 0.504 | 0.622 | 0.548 | 0.505 | 0.651 | 0.581 | 0.530 |
| P: logP (HGB) | 0.551 | 0.528 | 0.509 | 0.638 | 0.540 | 0.497 | 0.632 | 0.551 | 0.503 |
| P: rot. bonds (HGB) | 0.507 | 0.506 | 0.502 | 0.527 | 0.523 | 0.535 | 0.513 | 0.510 | 0.502 |
| P: ring count (HGB) | 0.523 | 0.520 | 0.517 | 0.506 | 0.508 | 0.478 | 0.575 | 0.567 | 0.569 |
| P: 4 descriptors (HGB) | 0.603 | 0.553 | 0.514 | 0.692 | 0.569 | 0.505 | 0.726 | 0.629 | 0.568 |
| P: proxies = tc+4 desc (HGB) | 0.607 | 0.565 | 0.529 | 0.699 | 0.585 | 0.539 | 0.734 | 0.637 | 0.562 |
| T: shared targets (HGB) | 0.512 | 0.512 | 0.513 | 0.508 | 0.507 | 0.509 | 0.525 | 0.526 | 0.521 |
| T: tc + shared targets (HGB) | 0.535 | 0.532 | 0.524 | 0.531 | 0.523 | 0.504 | 0.557 | 0.549 | 0.522 |
| T: proxies + shared targets (HGB) | 0.611 | 0.570 | 0.533 | 0.701 | 0.586 | 0.534 | 0.741 | 0.648 | 0.577 |
| S: ECFP4 (HGB) | 0.640 | 0.580 | 0.546 | 0.745 | 0.590 | 0.534 | 0.810 | 0.681 | 0.592 |
| S: ECFP4 + tc (HGB) | 0.645 | 0.590 | 0.557 | 0.744 | 0.600 | 0.545 | 0.811 | 0.687 | 0.592 |
| S: ECFP4 + proxies (HGB) | 0.662 | 0.604 | 0.568 | 0.786 | 0.617 | 0.545 | 0.828 | 0.695 | 0.597 |

### T4. Paired per-fold differences (AUROC and AP; mean ± std; #folds with diff > 0 / #folds)

| A − B | sample | warm ΔAUROC | warm ΔAP | S1 ΔAUROC | S1 ΔAP | S2 ΔAUROC | S2 ΔAP |
|---|---|---|---|---|---|---|---|
| S: ECFP4 + proxies (HGB) − P: proxies = tc+4 desc (HGB) | U1000 | +0.061 ± 0.001 (5/5) | +0.128 ± 0.002 (5/5) | +0.061 ± 0.007 (5/5) | +0.084 ± 0.013 (5/5) | +0.058 ± 0.018 (5/5) | +0.060 ± 0.021 (5/5) |
| S: ECFP4 + proxies (HGB) − P: proxies = tc+4 desc (HGB) | U300 | +0.031 ± 0.001 (5/5) | +0.085 ± 0.005 (5/5) | +0.031 ± 0.016 (5/5) | +0.057 ± 0.039 (5/5) | +0.022 ± 0.035 (4/5) | +0.031 ± 0.027 (4/5) |
| S: ECFP4 + proxies (HGB) − P: proxies = tc+4 desc (HGB) | H300 | +0.068 ± 0.006 (5/5) | +0.034 ± 0.005 (5/5) | +0.050 ± 0.018 (5/5) | +0.025 ± 0.011 (5/5) | +0.030 ± 0.023 (4/5) | +0.009 ± 0.020 (4/5) |
| S: ECFP4 (HGB) − P: proxies = tc+4 desc (HGB) | U1000 | +0.052 ± 0.001 (5/5) | +0.104 ± 0.005 (5/5) | +0.038 ± 0.010 (5/5) | +0.045 ± 0.022 (5/5) | +0.016 ± 0.028 (3/5) | +0.018 ± 0.027 (3/5) |
| S: ECFP4 (HGB) − P: proxies = tc+4 desc (HGB) | U300 | +0.018 ± 0.002 (5/5) | +0.049 ± 0.005 (5/5) | +0.019 ± 0.020 (4/5) | +0.054 ± 0.043 (5/5) | +0.017 ± 0.043 (3/5) | +0.028 ± 0.035 (4/5) |
| S: ECFP4 (HGB) − P: proxies = tc+4 desc (HGB) | H300 | +0.054 ± 0.005 (5/5) | +0.026 ± 0.004 (5/5) | +0.034 ± 0.012 (5/5) | +0.012 ± 0.010 (4/5) | +0.008 ± 0.025 (3/5) | -0.014 ± 0.022 (2/5) |
| S: ECFP4 + tc (HGB) − S: ECFP4 (HGB) | U1000 | +0.002 ± 0.000 (5/5) | +0.008 ± 0.004 (5/5) | +0.016 ± 0.004 (5/5) | +0.029 ± 0.015 (5/5) | +0.039 ± 0.011 (5/5) | +0.039 ± 0.023 (5/5) |
| S: ECFP4 + tc (HGB) − S: ECFP4 (HGB) | U300 | +0.001 ± 0.002 (3/5) | +0.001 ± 0.006 (3/5) | +0.012 ± 0.011 (5/5) | +0.009 ± 0.026 (2/5) | +0.022 ± 0.027 (4/5) | +0.016 ± 0.025 (4/5) |
| S: ECFP4 + tc (HGB) − S: ECFP4 (HGB) | H300 | +0.001 ± 0.004 (3/5) | +0.001 ± 0.003 (4/5) | +0.008 ± 0.008 (4/5) | +0.006 ± 0.010 (4/5) | +0.011 ± 0.015 (4/5) | +0.010 ± 0.019 (4/5) |
| S: ECFP4 + proxies (HGB) − S: ECFP4 (HGB) | U1000 | +0.009 ± 0.001 (5/5) | +0.024 ± 0.004 (5/5) | +0.024 ± 0.003 (5/5) | +0.040 ± 0.015 (5/5) | +0.042 ± 0.011 (5/5) | +0.043 ± 0.020 (5/5) |
| S: ECFP4 + proxies (HGB) − S: ECFP4 (HGB) | U300 | +0.013 ± 0.002 (5/5) | +0.036 ± 0.005 (5/5) | +0.012 ± 0.006 (5/5) | +0.003 ± 0.024 (1/5) | +0.005 ± 0.022 (4/5) | +0.003 ± 0.032 (3/5) |
| S: ECFP4 + proxies (HGB) − S: ECFP4 (HGB) | H300 | +0.013 ± 0.005 (5/5) | +0.007 ± 0.004 (5/5) | +0.016 ± 0.012 (4/5) | +0.013 ± 0.011 (4/5) | +0.023 ± 0.026 (4/5) | +0.024 ± 0.029 (4/5) |
| T: proxies + shared targets (HGB) − P: proxies = tc+4 desc (HGB) | U1000 | +0.001 ± 0.002 (3/5) | +0.008 ± 0.005 (5/5) | +0.003 ± 0.002 (5/5) | +0.020 ± 0.005 (5/5) | +0.004 ± 0.007 (4/5) | +0.021 ± 0.011 (5/5) |
| T: proxies + shared targets (HGB) − P: proxies = tc+4 desc (HGB) | U300 | -0.000 ± 0.002 (2/5) | -0.001 ± 0.006 (3/5) | +0.001 ± 0.005 (3/5) | +0.010 ± 0.007 (5/5) | -0.000 ± 0.010 (2/5) | +0.006 ± 0.010 (4/5) |
| T: proxies + shared targets (HGB) − P: proxies = tc+4 desc (HGB) | H300 | +0.005 ± 0.003 (5/5) | +0.003 ± 0.002 (5/5) | +0.006 ± 0.005 (5/5) | +0.007 ± 0.004 (5/5) | +0.010 ± 0.009 (5/5) | +0.012 ± 0.009 (5/5) |
| N: SimKNN k=10 (2-sided) − P: proxies = tc+4 desc (HGB) | U1000 | +0.029 ± 0.002 (5/5) | +0.087 ± 0.004 (5/5) | +0.055 ± 0.020 (5/5) | +0.116 ± 0.038 (5/5) | +0.071 ± 0.044 (5/5) | +0.097 ± 0.047 (5/5) |
| N: SimKNN k=10 (2-sided) − P: proxies = tc+4 desc (HGB) | U300 | -0.056 ± 0.005 (0/5) | -0.103 ± 0.008 (0/5) | +0.020 ± 0.025 (4/5) | +0.061 ± 0.053 (4/5) | +0.085 ± 0.074 (4/5) | +0.078 ± 0.059 (5/5) |
| N: SimKNN k=10 (2-sided) − P: proxies = tc+4 desc (HGB) | H300 | +0.008 ± 0.011 (3/5) | +0.003 ± 0.007 (3/5) | +0.044 ± 0.013 (5/5) | +0.019 ± 0.005 (5/5) | +0.048 ± 0.035 (4/5) | +0.020 ± 0.021 (4/5) |
| N: SimKNN k=10 (2-sided) − B: degree prior, train labels (product) | U1000 | -0.050 ± 0.001 (0/5) | -0.067 ± 0.005 (0/5) | +0.010 ± 0.013 (4/5) | +0.107 ± 0.036 (5/5) | +0.187 ± 0.035 (5/5) | +0.171 ± 0.049 (5/5) |
| N: SimKNN k=10 (2-sided) − B: degree prior, train labels (product) | U300 | -0.058 ± 0.005 (0/5) | -0.099 ± 0.011 (0/5) | -0.001 ± 0.020 (3/5) | +0.056 ± 0.022 (5/5) | +0.168 ± 0.038 (5/5) | +0.131 ± 0.050 (5/5) |
| N: SimKNN k=10 (2-sided) − B: degree prior, train labels (product) | H300 | +0.066 ± 0.008 (5/5) | +0.026 ± 0.004 (5/5) | +0.083 ± 0.010 (5/5) | +0.040 ± 0.004 (5/5) | +0.147 ± 0.023 (5/5) | +0.095 ± 0.012 (5/5) |
| P: proxies = tc+4 desc (HGB) − B: degree prior, train labels (product) | U1000 | -0.080 ± 0.001 (0/5) | -0.154 ± 0.002 (0/5) | -0.045 ± 0.015 (0/5) | -0.009 ± 0.027 (2/5) | +0.116 ± 0.029 (5/5) | +0.074 ± 0.028 (5/5) |
| P: proxies = tc+4 desc (HGB) − B: degree prior, train labels (product) | U300 | -0.002 ± 0.002 (0/5) | +0.004 ± 0.004 (4/5) | -0.021 ± 0.015 (0/5) | -0.004 ± 0.036 (3/5) | +0.082 ± 0.047 (5/5) | +0.052 ± 0.032 (5/5) |
| P: proxies = tc+4 desc (HGB) − B: degree prior, train labels (product) | H300 | +0.058 ± 0.005 (5/5) | +0.023 ± 0.004 (5/5) | +0.039 ± 0.012 (5/5) | +0.021 ± 0.007 (5/5) | +0.098 ± 0.022 (5/5) | +0.075 ± 0.016 (5/5) |

### T5. Share of above-chance AUROC reached without structure: (AUROC_proxy − 0.5) / (AUROC_ECFP − 0.5)

| sample | split | ECFP4 (HGB) | proxies tc+4desc (HGB) | target count only (HGB) | 4 descriptors (HGB) | share proxies | share tc | share desc |
|---|---|---|---|---|---|---|---|---|
| U1000 | warm | 0.873 | 0.821 | 0.652 | 0.814 | 0.86 | 0.41 | 0.84 |
| U1000 | S1 | 0.774 | 0.736 | 0.638 | 0.708 | 0.86 | 0.50 | 0.76 |
| U1000 | S2 | 0.632 | 0.616 | 0.620 | 0.559 | 0.88 | 0.91 | 0.44 |
| U300 | warm | 0.917 | 0.899 | 0.675 | 0.899 | 0.96 | 0.42 | 0.96 |
| U300 | S1 | 0.774 | 0.755 | 0.644 | 0.731 | 0.93 | 0.53 | 0.84 |
| U300 | S2 | 0.599 | 0.582 | 0.601 | 0.528 | 0.83 | 1.02 | 0.28 |
| H300 | warm | 0.853 | 0.799 | 0.605 | 0.792 | 0.85 | 0.30 | 0.83 |
| H300 | S1 | 0.738 | 0.704 | 0.589 | 0.688 | 0.86 | 0.37 | 0.79 |
| H300 | S2 | 0.606 | 0.598 | 0.558 | 0.583 | 0.93 | 0.55 | 0.78 |

### T6. Spearman correlation of test scores with (i) the out-of-sample DrugBank degree product and (ii) the proxies-HGB score (mean over folds)

| Method | U1000 S1 ρ_ext / ρ_proxy | U1000 S2 ρ_ext / ρ_proxy | U300 S1 ρ_ext / ρ_proxy | U300 S2 ρ_ext / ρ_proxy | H300 S1 ρ_ext / ρ_proxy | H300 S2 ρ_ext / ρ_proxy |
|---|---|---|---|---|---|---|
| B: degree prior, train labels (product) | 0.65 / 0.57 | – | 0.65 / 0.69 | – | 0.62 / 0.55 | – |
| B: degree prior, train labels (sum) | 0.65 / 0.57 | – | 0.65 / 0.69 | – | 0.62 / 0.55 | – |
| U: Tanimoto ECFP4 (raw) | 0.09 / 0.19 | 0.09 / 0.19 | 0.11 / 0.16 | 0.10 / 0.18 | 0.03 / 0.10 | 0.02 / 0.08 |
| U: target-count product (raw) | 0.29 / 0.48 | 0.29 / 0.49 | 0.26 / 0.32 | 0.26 / 0.34 | 0.35 / 0.40 | 0.35 / 0.43 |
| U: # shared targets (raw) | 0.09 / 0.10 | 0.09 / 0.09 | 0.07 / 0.07 | 0.08 / 0.05 | 0.12 / 0.18 | 0.12 / 0.18 |
| U: Jaccard of target sets (raw) | 0.09 / 0.10 | 0.09 / 0.09 | 0.07 / 0.07 | 0.08 / 0.05 | 0.12 / 0.18 | 0.12 / 0.17 |
| [diag] external DrugBank degree product | 1.00 / 0.47 | 1.00 / 0.20 | 1.00 / 0.52 | 1.00 / 0.11 | 1.00 / 0.46 | 1.00 / 0.24 |
| N: Vilar max-Tanimoto to partners | 0.57 / 0.48 | – | 0.57 / 0.58 | – | 0.15 / 0.14 | – |
| N: Vilar mean-Tanimoto to partners | 0.32 / 0.28 | – | 0.37 / 0.33 | – | 0.05 / 0.10 | – |
| N: SimKNN k=10 (2-sided) | 0.39 / 0.43 | 0.24 / 0.25 | 0.47 / 0.52 | 0.28 / 0.23 | 0.26 / 0.44 | 0.10 / 0.27 |
| B: identity one-hot LR | – | – | – | – | – | – |
| P: target count (HGB) | 0.29 / 0.49 | 0.26 / 0.49 | 0.29 / 0.40 | 0.23 / 0.38 | 0.29 / 0.41 | 0.23 / 0.40 |
| P: MW (HGB) | 0.26 / 0.46 | 0.01 / 0.29 | 0.40 / 0.60 | -0.05 / 0.22 | 0.36 / 0.55 | 0.04 / 0.28 |
| P: logP (HGB) | 0.30 / 0.51 | 0.08 / 0.38 | 0.43 / 0.61 | -0.00 / 0.26 | 0.34 / 0.50 | -0.04 / 0.23 |
| P: rot. bonds (HGB) | 0.12 / 0.28 | 0.07 / 0.26 | 0.15 / 0.31 | 0.01 / 0.25 | 0.20 / 0.20 | 0.16 / 0.14 |
| P: ring count (HGB) | 0.09 / 0.26 | 0.06 / 0.27 | 0.23 / 0.28 | 0.18 / 0.22 | 0.16 / 0.39 | 0.13 / 0.39 |
| P: 4 descriptors (HGB) | 0.41 / 0.80 | 0.09 / 0.69 | 0.50 / 0.87 | 0.05 / 0.72 | 0.40 / 0.81 | 0.13 / 0.68 |
| P: proxies = tc+4 desc (HGB) | 0.47 / 1.00 | 0.20 / 1.00 | 0.52 / 1.00 | 0.11 / 1.00 | 0.46 / 1.00 | 0.24 / 1.00 |
| T: shared targets (HGB) | 0.09 / 0.10 | 0.09 / 0.09 | 0.07 / 0.07 | 0.07 / 0.06 | 0.12 / 0.18 | 0.12 / 0.17 |
| T: tc + shared targets (HGB) | 0.29 / 0.49 | 0.26 / 0.49 | 0.29 / 0.40 | 0.23 / 0.38 | 0.28 / 0.40 | 0.23 / 0.40 |
| T: proxies + shared targets (HGB) | 0.47 / 0.96 | 0.21 / 0.94 | 0.52 / 0.97 | 0.12 / 0.95 | 0.44 / 0.92 | 0.23 / 0.90 |
| S: ECFP4 (HGB) | 0.57 / 0.53 | 0.24 / 0.23 | 0.61 / 0.63 | 0.21 / 0.22 | 0.45 / 0.53 | 0.11 / 0.23 |
| S: ECFP4 + tc (HGB) | 0.60 / 0.60 | 0.31 / 0.38 | 0.63 / 0.67 | 0.23 / 0.31 | 0.48 / 0.60 | 0.20 / 0.40 |
| S: ECFP4 + proxies (HGB) | 0.60 / 0.66 | 0.30 / 0.46 | 0.60 / 0.72 | 0.19 / 0.39 | 0.49 / 0.70 | 0.23 / 0.54 |

### T7. HGB fit+predict time per fold (s, mean)

| sample | warm ECFP4 | node ECFP4 | warm proxies | node proxies |
|---|---|---|---|---|
| U1000 | 31 | 65 | 0.8 | 2.9 |
| U300 | 36 | 43 | 0.6 | 0.8 |
| H300 | 34 | 46 | 0.5 | 0.6 |
