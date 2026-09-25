**Compute and approximations**
* **Reduced folds (compute budget, shared machine).** See the fold-count table under "Protocol details". The std across folds was 0.00–0.02 for every as-in-paper metric, so the means are stable. S2 test sets are small (630 pairs per fold for D1/D2), so S2 standard deviations are ±0.03–0.09.
* **HGB replaces sklearn's exact `GradientBoostingClassifier`.** It is usually slightly stronger; it would not *lower* the paper's GB numbers.
* **FFNN-orig was reproduced with `isValid=False`** (200 full-batch steps). The paper may have used `isValid=True`, which takes 20 steps per epoch (4,000 in total). Only the 1,000-epoch full-batch sensitivity was run, not 4,000 steps. Longer training does lift FFNN-orig toward the FFNN-fixed level, but that level is itself matched or beaten by random identity vectors (section 3).
* **Score-only baselines use a tuned threshold.** Degree, raw S, SimKNN and DrugBank degree are binarised at the threshold that maximises MCC on the *training* pairs; learned models use a 0.5 probability. AUROC and AUPRC do not depend on thresholds.

**Inputs used as given**
* **node2vec was not re-run.** I used the committed embedding files (`*_hop{1,2}_128_1.2_1.2.txt`; `mp_128_1.2_1.2.txt` for MP, whose hop setting is not recorded). gensim/node2vec are not installed, so there is no estimate of node2vec seed variance, and I could not test whether a single *global* node2vec embedding would change the cold-start picture.
* **The `sim_arr` fusion was not re-created.** KEGG SIMCOMP2 was not queried. The D1/D2 `sim_arr` does not match fingerprints recomputed from the current `drugs.json` (section 1), so the exact 2022 similarity pipeline is unknown.
* **CROssBAR was not accessed.** The new METU MongoDB was not used; all data came from the repo snapshot.

**Scope**
* **HIN-DDI was not re-run.** The HIN-DDI numbers quoted in the report come from Tanvir et al. (2021) under their own protocol, and the MP labels are HIN-DDI's DDI list (only 96% overlap with the repo's DrugBank file). Comparing DDNet with those numbers is not like for like. The HIN-DDI paper also reports percentages, while the DDNet report gives fractions.
* **Not evaluated:** the z-score-filtered embedding files (`*_z_*`), hop-2 under the leakage-free protocols, and hyperparameter tuning for any learner. No learner was tuned, which works in DDNet's favour relative to the baselines only where DDNet has more parameters.
* **Hard-label "AUROC/AUPRC" as the paper computes them** are included only for comparability. They should not be used in a publication.
