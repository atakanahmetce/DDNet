# DDNet (CENG-514, 2022): empirical re-evaluation

*Run on 2026-09-25 against the repo at `/home/user/DDNet`. The repo was only read; nothing was written there. All scripts and outputs are in this folder.*

## What was run

| Script | Task | Output |
|---|---|---|
| `s01_characterize.py` | (5) dataset characterisation: density, degrees vs full DrugBank, similarity graph, whether `sim_arr` carries DDI information | `out/characterize.json`, `out/gdeg_quantiles.csv` |
| `s02_reproduce.py` | (1)+(2) as-in-paper protocol plus trivial baselines under the same protocol | `out/reproduce_<cfg>.json` |
| `s02b_controls.py` | (2) identity controls under the as-in-paper protocol: random vectors in place of node2vec, and an identity MLP | `out/controls_<cfg>.json` |
| `s02c_ffnn_orig_long.py` | sensitivity: the original FFNN trained for up to 1000 epochs | `out/ffnn_orig_long_D1h1.json` |
| `s03_proper.py` | (3) leakage-free protocols: warm / S1 / S2 | `out/proper_<ds>_hop1.json` |
| `s04_cross.py` | (4) cross-dataset D1→D2, embedding alignment and comparability | `out/cross.json` |
| `s05_tables.py`, `s06_build_results.py` | tables and this file | `out/tables.md`, `out/compact.md`, `RESULTS.md` |

`common.py` holds the loaders and metrics; `nets.py` holds the FFNNs. Seeds are fixed (KFold seed 0, torch seed = fold index, model `random_state=0`, random-vector control seed 123).

**Protocol details, and where I departed from the paper (please read):**
* **As-in-paper** (`gen_matrix` + `get_data`):
  * Rows are all ordered pairs over the embedding-file drugs, including self pairs. Features are `[emb(src), emb(tgt), path1..3]` (259-D); labels are the directed pairs in `interactions.txt`, which is stored in both orientations.
  * Features go through the sklearn `Normalizer` (row-wise L2), which is the `get_data` default.
  * I used shuffled 5-fold `KFold` with seed 0. **Fold 1 is reported as "the 80/20 random split"**, since statistically it is the same as `train_test_split(test_size=0.2)`; I did not fit a sixth model for it.
* **"GB" is `HistGradientBoostingClassifier`** (100 iterations, lr 0.1, no early stopping). sklearn's exact `GradientBoostingClassifier` needed about 5 s per tree on MP (roughly 9 min per fit), which did not fit the compute budget.
* **FFNN-orig** is a faithful copy of `Net`, `deep_model(isValid=False)` and `evaluate_score(isDeep=True)`:
  * `Softmax(dim=0)` normalises over the whole batch, followed by `BCELoss`.
  * Training is full-batch Adam (lr 7e-4, wd 1e-5, dropout 0.4) for 200 epochs.
  * A hard label is positive when the softmax output is ≥ 0.5/len(X_test), computed over the whole test set at once. The score used for AUROC is the pre-softmax logit.
* **FFNN-fixed** uses the same layer stack with no softmax: `BCEWithLogitsLoss`, a 0.5 sigmoid threshold, mini-batch 512, 40 epochs, and `StandardScaler` in place of the row normaliser.
* The paper computes AUROC/AUPRC **from hard labels** (`classif_AUC(y, f)` with `f = model.predict`). Both that version ("hard") and the correct score-based version are reported.
* **Compute cuts.** Other agents' jobs shared this 4-core machine, and OpenMP with 4 threads was pathologically slow (HGB took 120 s instead of 6 s), so I cut the fold counts:

  | Configuration | Folds run |
  |---|---|
  | D1/D2 hop-1 (as-in-paper) | 3 of 5 |
  | D1/D2 hop-2 (as-in-paper) | 2 of 5 |
  | D1q1 (as-in-paper) | 5 of 5 |
  | D2q1 (as-in-paper) | 2 of 5 |
  | MP (as-in-paper) | 1 (the 80/20 split) |
  | MP (leakage-free) | {{MP_FOLDS}}, without DDNet-sym RF |

  The fold-to-fold std is about 0.01 wherever it could be measured, so the conclusions do not depend on these cuts.

## 0. Key findings

{{TAKEAWAYS}}

## 1. Dataset characterisation (task 5)

| Quantity | D1 | D2 | MP |
|---|---|---|---|
| Drugs (files) / unordered positive pairs | 179 / 9,275 | 180 / 9,429 | 289 / 18,900 |
| Positive density (unordered, no self pairs) | **0.582** | **0.585** | **0.454** |
| F1 of the all-positive predictor | 0.736 | 0.738 | 0.625 |
| Hard-label "AUPRC" of the all-positive predictor = (1+p)/2 | 0.79 | 0.79 | 0.73 |
| In-dataset DDI degree: min / median / max | 5 / 106 / 172 | 11 / 111 / 168 | 0 / 140 / 254 |
| Global DrugBank degree of the chosen drugs: median (DrugBank-wide median = 578) | **1,258** | **1,265** | **1,158** |
| Percentile of that median within all 4,418 DrugBank drugs | 85th | 85th | 82nd |
| Share of drugs in the DrugBank top degree quartile | 71% | 71% | 63% |
| Spearman(in-dataset degree, DrugBank degree) | 0.95 | 0.96 | 0.93 |
| AUROC of in-dataset degree product (oracle upper bound of degree signal) | 0.83 | 0.85 | 0.87 |
| AUROC of *external* DrugBank-degree product | 0.81 | 0.84 | 0.84 |
| AUROC of raw fused similarity S | 0.58 | 0.59 | 0.61 |
| AUROC of Morgan cosine recomputed from `drugs.json` | 0.56 | 0.58 | 0.59 |
| AUROC of path1 / path2 / path3 | 0.56 / 0.59 / 0.59 | 0.57 / 0.62 / 0.62 | 0.54 / 0.61 / 0.62 |
| Pearson(S, recomputed Morgan cosine) | **0.40** | **0.45** | 0.90 |
| Spearman with the label-derived DDI-profile Jaccard: S vs Morgan cosine | 0.17 vs 0.14 | 0.21 vs 0.16 | 0.25 vs 0.20 |
| Similarity graph used (`*_sim3.edgelist`, weight > **0.3**): edges / density | 5,260 / 0.33 | 5,782 / 0.36 | 4,966 / 0.12 |
| Mean hop-1 ego size / mean hop-2 ego size (fraction of graph) | 60 / 179 (**100%**) | 65 / 180 (**100%**) | 35 / 153 (53%) |
| At the paper's stated threshold 0.5: edges / isolated drugs / components | 416 / 24 / 28 | 538 / 17 / 20 | 489 / 61 / 92 |
| DDI rate on similarity edges (> 0.5) vs off them | 0.81 vs 0.58 | 0.78 vs 0.58 | 0.79 vs 0.45 |

DrugBank-wide degree quantiles (min / q25 / median / q75 / max): all 4,418 drugs 1 / 124 / 578 / 969 / 2,492. The chosen drugs: D1 108 / 912 / 1,258 / 1,592 / 2,492; D2 86 / 869 / 1,265 / 1,627 / 2,479; MP 0 / 778 / 1,158 / 1,516 / 2,492. Pairwise overlaps: D1∩D2 = 46 drugs, D1∩MP = 59, D2∩MP = 63.

What this shows:
* **The label matrices are dense, positives are the majority class, and the drugs are heavily popularity-biased.** Most drugs sit in the top quartile of DrugBank degree, and in-dataset degree is almost the same ranking as global DrugBank degree (ρ = 0.93–0.96). A pure degree product therefore scores AUROC 0.83–0.87.
* **`sim_arr` shows no sign of DDI leakage.**
  * Its AUROC is 0.58–0.61, only slightly above Morgan cosine's 0.56–0.59.
  * Its correlation with the label-derived DDI-profile similarity is only marginally higher than that of the plain fingerprint.
  * The component that is not Morgan (`2S − cos`) has the same AUROC as S.
  * The modest enrichment of DDIs on similarity edges (0.8 vs 0.58) is the known "similar drugs interact alike" effect.
* **The D1/D2 `sim_arr` cannot be reproduced from `drugs.json` SMILES.**
  * MP's `sim_arr` matches `0.5·cos(Morgan r=2, 1024) + 0.5·X` with X in [0, 1] (the SIMCOMP part), with correlation 0.90.
  * For D1/D2, the correlation with any standard fingerprint similarity is only 0.4–0.5 (Morgan r=1/2/3, 1024/2048 bits, FCFP4, MACCS, RDKit), and `S − 0.5·cos` is negative for 10% of pairs.
  * So either different SMILES or a different fingerprint pipeline was used in 2022. This needs fixing before any paper.
* **The repo does not match the report.**
  * The edgelists use threshold 0.3 (`sim3`), not the 0.5 the report states.
  * At 0.3, the hop-2 ego graph of every D1/D2 drug is the *whole* graph, so every hop-2 "local" embedding is trained on the same graph. This explains the hop-2 collapse in the report's supplementary PCA figure (the mean off-diagonal cosine is 0.64–0.69 for hop-2 versus 0.44 for hop-1).
  * D1 has 179 drugs in the files, not the 176 the report gives.
  * The MP labels (from HIN-DDI) overlap the repo's DrugBank file only partially: 36,426 of 37,799 directed pairs.
* **The "outlier removal" drops most drugs.** The report says IQR outlier detection "yielded better results". The repo's IQR-filtered embedding files (`*_q_*`) keep only 71 of 179 (D1) and 73 of 180 (D2) drugs. They are evaluated below as D1q1/D2q1, and their positive rate rises to 0.66–0.69.

## 2. Reproduction as in the paper (task 1)

{{COMPACT_A}}

Findings:
* **RF and GB come out at or above the report.** Under the as-in-paper protocol they reach F1 0.78–0.84 and MCC 0.48–0.66, against the report's 0.53–0.85 and 0.26–0.52. The report's RF on MP (F1 0.53, MCC 0.26) is far below a plain `RandomForest(100, balanced)` (0.82, 0.66). It was probably the `RandomizedSearchCV` path, whose `min_samples_split` grid of fractions 0.05–1.0 forces very shallow trees.
* **The committed FFNN code does not reproduce the report's FFNN numbers.**
  * `Softmax(dim=0)` normalises over the batch, and the 0.5/N threshold makes a prediction depend on the rest of the test batch.
  * With 200 full-batch epochs it predicts 70–98% positives, giving MCC 0.14–0.52 against the report's 0.24–0.72.
  * Longer training helps: see the sensitivity check below.
* **A corrected sigmoid FFNN (FFNN-fixed) does reach the report's FFNN numbers**: MCC 0.62–0.72, F1 0.83–0.88, hard-label AUPRC 0.87–0.92. **Section 3 shows these numbers do not come from the graph embeddings.**
* **Hard-label AUPRC inflates the scores.** It comes from a 3-point precision-recall curve, and a model that predicts every pair positive already scores 0.79 on D1/D2 and 0.73 on MP.

Sensitivity of the original FFNN to training length (D1 hop-1, fold 1). In the repo, `deep_model(isValid=True)` takes 20 optimiser steps per epoch.

{{FFNN_LONG}}

## 3. Trivial baselines and identity controls under the same (leaky) protocol (task 2)

{{COMPACT_B}}

Findings:
* **Degree and one-hot identity nearly match DDNet RF/GB.**
  * Degree uses a per-drug positive rate computed from training pairs only; one-hot identity uses LR or RF.
  * Degree reaches AUROC 0.82–0.87 and MCC 0.47–0.56; identity LR/RF reach AUROC 0.81–0.87. DDNet RF/GB reach 0.86–0.91 and MCC 0.54–0.66.
* **The graph embeddings add nothing beyond drug identity.**
  * With node2vec vectors replaced by i.i.d. Gaussian vectors and the same FFNN-fixed, the model scores *higher* than DDNet: D1 AUROC 0.956 / MCC 0.77 vs 0.937 / 0.71; D2 0.950 / 0.75 vs 0.937 / 0.72.
  * An MLP on one-hot drug identity with no features at all is best: AUROC 0.97, MCC 0.83–0.85, F1 0.93.
  * So the report's best numbers (MCC ≈ 0.7, F1 ≈ 0.9) are reachable by memorising drug identities in a dense 58% matrix. This is matrix completion over popular drugs, helped by mirrored pairs landing in train and test.
* The path features alone are weak (AUROC 0.64–0.70), and raw similarity is weaker still (0.57–0.61).

## 4. Leakage-free protocols (task 3)

These use unordered pairs with no self pairs, symmetric features `[e_a+e_b, e_a⊙e_b, |e_a−e_b|, path]`, or the paper's concatenated layout trained on both orientations and scored as the mean of both orientations. A mirrored pair can never straddle folds because it is a single row.
* **warm**: random 5-fold split over pairs.
* **S1**: node-level 5-fold split. Training uses pairs whose drugs are both in the training folds; testing uses pairs with exactly one held-out drug.
* **S2**: the same trained model, tested on pairs with both drugs held out.
* **Morgan FP** uses `[fp_a+fp_b (0/1/2 per bit), Tanimoto]` from `drugs.json` SMILES with RDKit's 1024-bit ECFP4.
* **SimKNN** averages the training labels between the top-10 S-neighbours of a and of b.
* **"[leaky] DrugBank global degree"** uses DDI counts from outside the dataset. It is a diagnostic, not a legitimate method.

{{COMPACT_C}}

Findings:
{{PROPER_FINDINGS}}

## 5. Cross-dataset D1 → D2, the paper's "new drug" experiment (task 4)

{{CROSS_TABLES}}

Findings:
{{CROSS_FINDINGS}}

## 6. What this means for publication (empirical side only)

{{PUBLISH}}

## 7. Caveats, and what I could not run

{{CAVEATS}}

## 8. Rerunning

```
cd reeval
python s01_characterize.py
python s02_reproduce.py MP D1h1 D1h2 D2h1 D2h2 D1q1 D2q1   # env MAXFOLDS=k to run the first k folds; RF_JOBS, TORCH_THREADS
python s02b_controls.py D1h1 D2h1 MP
python s02c_ffnn_orig_long.py
python s03_proper.py D1; python s03_proper.py D2; python s03_proper.py MP   # env SYM_RF=0 / WARM_FOLDS=k to lighten
python s04_cross.py
python s05_tables.py && python s06_build_results.py
```
Set `OMP_NUM_THREADS`, or rely on the per-call `threadpool_limits(1)` already in the scripts. On a shared machine, OpenMP with 4 threads was about 20× slower than 1 thread.

## Appendix: full per-configuration tables

{{FULL}}
