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
  | MP (leakage-free) | 3 warm and 5 node folds, without DDNet-sym RF |

  The fold-to-fold std is about 0.01 wherever it could be measured, so the conclusions do not depend on these cuts.

## 0. Key findings

1. **The 2022 numbers are not publishable as they stand.** Four things produce them:
   - the protocol: ordered pairs including self pairs, mirrored pairs split across train and test, and AUROC/AUPRC computed from hard labels;
   - a dense label matrix: 45–58% positives, with positives as the majority class;
   - heavy popularity bias: most drugs are in the top quartile of DrugBank degree;
   - drug-identity memorisation.
2. **Reproduction.** The committed code and data give RF/GB F1 0.78–0.84 and MCC 0.48–0.66, at or above the report. The committed FFNN has a bug: `Softmax(dim=0)` normalises over the batch and the threshold is 0.5/len(X). It does **not** reproduce the report's FFNN (MCC 0.14–0.52 vs 0.24–0.72; about 0.55 even at 700–900 epochs on D1). A corrected sigmoid FFNN matches the report (MCC 0.62–0.72, F1 0.83–0.88).
3. **Controls under the paper's own protocol.** Replacing the node2vec vectors with **random Gaussian vectors scores higher** (MCC 0.75–0.77, AUROC 0.95–0.96, vs 0.69–0.72 / 0.93–0.94). An MLP on one-hot drug identity is best (MCC 0.82–0.88, AUROC 0.96–0.98). Degree and one-hot LR/RF alone give AUROC 0.81–0.87, roughly equal to DDNet RF/GB.
4. **Leakage-free protocols** (unordered pairs, no self pairs, symmetric features or orientation-averaged training):

   | Split | Result |
   |---|---|
   | Warm | Identity MLP 0.97–0.98 AUROC ≫ DDNet 0.90–0.91 ≈ Morgan-FP GBM 0.89–0.90 |
   | S1 (one new drug) | DDNet 0.71–0.78 ≈ Morgan FP 0.73–0.79; degree 0.72–0.74 |
   | S2 (both new) | DDNet 0.56–0.67 ≈ Morgan 0.56–0.66 ≈ SimKNN 0.61–0.66 ≈ raw similarity 0.59–0.61 |

   The *external* DrugBank-degree product (leaky) scores 0.81–0.84 in every split, including S2.
5. **The "new drug" D1→D2 experiment is at chance** (AUROC 0.40–0.53, MCC −0.07 to 0.03). The report's F1 0.64–0.80 and "AUPRC" 0.78–0.83 are no better than predicting every pair positive (F1 0.736, hard-label AUPRC 0.79). The D1 and D2 embedding spaces are unaligned: the same drug has cosine 0.08, no different from unrelated drugs. Procrustes alignment shows the relative geometry is only partly shared.
6. **Data and code do not match the report.**
   - The edge threshold is 0.3, not the stated 0.5; at 0.3 every hop-2 ego graph is the whole D1/D2 graph.
   - The D1/D2 `sim_arr` cannot be reproduced from the current `drugs.json` SMILES (r = 0.40–0.45 with Morgan cosine, against 0.90 for MP).
   - The IQR "outlier-removed" embedding files keep only 71 of 179 and 73 of 180 drugs.
   - The MP labels are HIN-DDI's list, not the repo's DrugBank file.
   - The HIN-DDI comparison mixes protocols and label sets.
7. **With moderate effort, there is no method gain to rescue.** Under proper cold-start splits, the per-ego node2vec features are no better than feeding Morgan fingerprints to a GBM. A paper would need a new contribution. One option is inductive, heterogeneous CROssBAR features (targets, pathways, enzymes/transporters, phenotypes) evaluated at DrugBank scale with S1/S2 splits and degree-controlled baselines. The other is a benchmark or negative-result paper on leakage and popularity bias in DDI prediction, whose novelty the literature review has to confirm.

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

### A. As-in-paper protocol: our re-run vs the 2022 report

| Data | Model | ours F1 | ours MCC | ours AUPRC (hard labels, paper way) | ours AUROC (scores) | paper F1 | paper MCC | paper AUPRC | folds |
|---|---|---|---|---|---|---|---|---|---|
| MP | DDNet RF | 0.819 | 0.663 | 0.856 | 0.912 | 0.53 | 0.26 | 0.67 | 1 |
| MP | DDNet GB(HGB) | 0.787 | 0.606 | 0.833 | 0.890 | 0.73 | – | 0.78 | 1 |
| MP | DDNet FFNN-orig | 0.754 | 0.515 | 0.799 | 0.883 | 0.85 | 0.72 | 0.87 | 1 |
| MP | DDNet FFNN-fixed | 0.829 | 0.691 | 0.871 | 0.933 | – | – | – | 1 |
| D1h1 | DDNet RF | 0.806 | 0.546 | 0.864 | 0.857 | 0.82 | 0.37 | 0.85 | 3 |
| D1h1 | DDNet GB(HGB) | 0.815 | 0.537 | 0.859 | 0.862 | 0.84 | 0.46 | 0.88 | 3 |
| D1h1 | DDNet FFNN-orig | 0.768 | 0.324 | 0.812 | 0.793 | 0.90 | 0.71 | 0.92 | 3 |
| D1h1 | DDNet FFNN-fixed | 0.867 | 0.709 | 0.919 | 0.937 | – | – | – | 3 |
| D2h1 | DDNet RF | 0.826 | 0.582 | 0.877 | 0.880 | 0.83 | 0.49 | 0.87 | 3 |
| D2h1 | DDNet GB(HGB) | 0.836 | 0.588 | 0.876 | 0.879 | 0.85 | 0.52 | 0.88 | 3 |
| D2h1 | DDNet FFNN-orig | 0.783 | 0.373 | 0.823 | 0.801 | 0.91 | 0.68 | 0.93 | 3 |
| D2h1 | DDNet FFNN-fixed | 0.875 | 0.716 | 0.922 | 0.937 | – | – | – | 3 |
| D1h2 | DDNet RF | 0.783 | 0.484 | 0.844 | 0.825 | 0.79 | 0.32 | 0.83 | 2 |
| D1h2 | DDNet GB(HGB) | 0.803 | 0.507 | 0.850 | 0.842 | 0.83 | 0.48 | 0.87 | 2 |
| D1h2 | DDNet FFNN-orig | 0.738 | 0.143 | 0.792 | 0.687 | 0.79 | 0.24 | 0.82 | 2 |
| D1h2 | DDNet FFNN-fixed | 0.833 | 0.622 | 0.889 | 0.897 | – | – | – | 2 |
| D2h2 | DDNet RF | 0.818 | 0.556 | 0.868 | 0.862 | 0.81 | 0.28 | 0.84 | 2 |
| D2h2 | DDNet GB(HGB) | 0.827 | 0.564 | 0.869 | 0.868 | 0.84 | 0.44 | 0.87 | 2 |
| D2h2 | DDNet FFNN-orig | 0.760 | 0.266 | 0.806 | 0.744 | 0.86 | 0.54 | 0.89 | 2 |
| D2h2 | DDNet FFNN-fixed | 0.863 | 0.678 | 0.906 | 0.923 | – | – | – | 2 |



Findings:
* **RF and GB come out at or above the report.** Under the as-in-paper protocol they reach F1 0.78–0.84 and MCC 0.48–0.66, against the report's 0.53–0.85 and 0.26–0.52. The report's RF on MP (F1 0.53, MCC 0.26) is far below a plain `RandomForest(100, balanced)` (0.82, 0.66). It was probably the `RandomizedSearchCV` path, whose `min_samples_split` grid of fractions 0.05–1.0 forces very shallow trees.
* **The committed FFNN code does not reproduce the report's FFNN numbers.**
  * `Softmax(dim=0)` normalises over the batch, and the 0.5/N threshold makes a prediction depend on the rest of the test batch.
  * With 200 full-batch epochs it predicts 70–98% positives, giving MCC 0.14–0.52 against the report's 0.24–0.72.
  * Longer training helps: see the sensitivity check below.
* **A corrected sigmoid FFNN (FFNN-fixed) does reach the report's FFNN numbers**: MCC 0.62–0.72, F1 0.83–0.88, hard-label AUPRC 0.87–0.92. **Section 3 shows these numbers do not come from the graph embeddings.**
* **Hard-label AUPRC inflates the scores.** It comes from a 3-point precision-recall curve, and a model that predicts every pair positive already scores 0.79 on D1/D2 and 0.73 on MP.

Sensitivity of the original FFNN to training length (D1 hop-1, fold 1). In the repo, `deep_model(isValid=True)` takes 20 optimiser steps per epoch.

| epochs (full-batch steps) | F1 | MCC | Precision | AUROC (scores) | hard-label AUPRC | predicted-positive rate |
|---|---|---|---|---|---|---|
| 100 | 0.733 | 0.090 | 0.578 | 0.666 | 0.789 | 0.994 |
| 200 | 0.767 | 0.325 | 0.638 | 0.787 | 0.811 | 0.867 |
| 300 | 0.793 | 0.431 | 0.677 | 0.836 | 0.829 | 0.813 |
| 400 | 0.803 | 0.465 | 0.692 | 0.850 | 0.836 | 0.795 |
| 500 | 0.816 | 0.512 | 0.720 | 0.865 | 0.848 | 0.753 |
| 600 | 0.820 | 0.523 | 0.725 | 0.881 | 0.850 | 0.747 |
| 700 | 0.830 | 0.558 | 0.750 | 0.885 | 0.860 | 0.712 |
| 800 | 0.830 | 0.556 | 0.750 | 0.890 | 0.860 | 0.711 |
| 900 | 0.826 | 0.545 | 0.745 | 0.883 | 0.857 | 0.714 |
| 1000 | 0.830 | 0.561 | 0.758 | 0.883 | 0.862 | 0.697 |

(Report, D1 hop-1 FFNN: F1 0.90, MCC 0.71, AUPRC 0.92, precision 0.86. True positive rate 0.58.)

## 3. Trivial baselines and identity controls under the same (leaky) protocol (task 2)

### B. Same (leaky) protocol: DDNet vs trivial baselines and identity controls (mean over folds)

| Method | MP AUROC / MCC / F1 / AUPRC-hard | D1h1 AUROC / MCC / F1 / AUPRC-hard | D2h1 AUROC / MCC / F1 / AUPRC-hard | D1q1 AUROC / MCC / F1 / AUPRC-hard | D2q1 AUROC / MCC / F1 / AUPRC-hard |
|---|---|---|---|---|---|
| All-positive | 0.500 / 0.000 / 0.624 / 0.727 | 0.500 / 0.000 / 0.732 / 0.789 | 0.500 / 0.000 / 0.738 / 0.792 | 0.500 / 0.000 / 0.818 / 0.846 | 0.500 / 0.000 / 0.793 / 0.828 |
| Raw similarity S | 0.605 / 0.139 / 0.628 / 0.714 | 0.572 / 0.113 / 0.579 / 0.719 | 0.584 / 0.156 / 0.632 / 0.747 | 0.618 / 0.218 / 0.791 / 0.849 | 0.628 / 0.245 / 0.693 / 0.819 |
| Path features only (HGB) | 0.641 / 0.219 / 0.497 / 0.649 | 0.687 / 0.231 / 0.724 / 0.786 | 0.698 / 0.280 / 0.738 / 0.798 | 0.799 / 0.386 / 0.842 / 0.875 | 0.825 / 0.440 / 0.831 / 0.869 |
| Degree (product) | 0.870 / 0.557 / 0.759 / 0.813 | 0.818 / 0.469 / 0.752 / 0.841 | 0.842 / 0.513 / 0.780 / 0.859 | 0.862 / 0.540 / 0.838 / 0.912 | 0.853 / 0.532 / 0.813 / 0.897 |
| One-hot identity LR | 0.866 / 0.545 / 0.755 / 0.807 | 0.816 / 0.440 / 0.775 / 0.832 | 0.839 / 0.488 / 0.798 / 0.849 | 0.856 / 0.440 / 0.848 / 0.885 | 0.849 / 0.474 / 0.838 / 0.876 |
| One-hot identity RF | 0.861 / 0.548 / 0.748 / 0.809 | 0.808 / 0.446 / 0.754 / 0.833 | 0.835 / 0.508 / 0.770 / 0.858 | 0.867 / 0.561 / 0.843 / 0.918 | 0.862 / 0.573 / 0.832 / 0.908 |
| DDNet RF | 0.912 / 0.663 / 0.819 / 0.856 | 0.857 / 0.546 / 0.806 / 0.864 | 0.880 / 0.582 / 0.826 / 0.877 | 0.870 / 0.552 / 0.862 / 0.910 | 0.876 / 0.565 / 0.843 / 0.902 |
| DDNet GB(HGB) | 0.890 / 0.606 / 0.787 / 0.833 | 0.862 / 0.537 / 0.815 / 0.859 | 0.879 / 0.588 / 0.836 / 0.876 | 0.890 / 0.559 / 0.875 / 0.907 | 0.887 / 0.578 / 0.863 / 0.900 |
| Embeddings only (HGB) | 0.903 / 0.638 / 0.804 / 0.846 | 0.888 / 0.594 / 0.834 / 0.876 | 0.898 / 0.626 / 0.850 / 0.888 | 0.910 / 0.632 / 0.893 / 0.921 | 0.904 / 0.658 / 0.888 / 0.917 |
| DDNet FFNN-orig | 0.883 / 0.515 / 0.754 / 0.799 | 0.793 / 0.324 / 0.768 / 0.812 | 0.801 / 0.373 / 0.783 / 0.823 | 0.824 / 0.335 / 0.841 / 0.867 | 0.809 / 0.358 / 0.823 / 0.854 |
| DDNet FFNN-fixed | 0.933 / 0.691 / 0.829 / 0.871 | 0.937 / 0.709 / 0.867 / 0.919 | 0.937 / 0.716 / 0.875 / 0.922 | 0.946 / 0.725 / 0.914 / 0.946 | 0.922 / 0.638 / 0.881 / 0.914 |
| RandomVec + path HGB (control) | 0.883 / 0.588 / 0.777 / 0.825 | 0.863 / 0.542 / 0.814 / 0.862 | 0.878 / 0.576 / 0.830 / 0.872 | – | – |
| RandomVec + path FFNN-fixed (control) | 0.955 / 0.773 / 0.878 / 0.902 | 0.956 / 0.774 / 0.899 / 0.938 | 0.950 / 0.752 / 0.893 / 0.930 | – | – |
| Identity MLP (FFNN-fixed on one-hot) | 0.983 / 0.880 / 0.933 / 0.953 | 0.971 / 0.848 / 0.935 / 0.956 | 0.964 / 0.821 / 0.925 / 0.947 | – | – |



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

### C. Leakage-free protocols: AUROC (mean ± std over folds)

| Method | D1 warm | D1 S1 | D1 S2 | D2 warm | D2 S1 | D2 S2 | MP warm | MP S1 | MP S2 |
|---|---|---|---|---|---|---|---|---|---|
| DDNet-sym RF | 0.805 ± 0.009 | 0.710 ± 0.024 | 0.600 ± 0.035 | 0.837 ± 0.007 | 0.764 ± 0.024 | 0.669 ± 0.050 | – | – | – |
| DDNet-sym HGB | 0.836 ± 0.007 | 0.712 ± 0.024 | 0.592 ± 0.039 | 0.860 ± 0.007 | 0.764 ± 0.013 | 0.659 ± 0.039 | 0.868 ± 0.005 | 0.757 ± 0.016 | 0.616 ± 0.028 |
| DDNet-sym FFNN-fixed | 0.903 ± 0.006 | 0.695 ± 0.019 | 0.561 ± 0.028 | 0.905 ± 0.006 | 0.744 ± 0.025 | 0.623 ± 0.035 | 0.903 ± 0.003 | 0.720 ± 0.016 | 0.563 ± 0.043 |
| DDNet-concat both-orient. HGB | 0.893 ± 0.005 | 0.747 ± 0.029 | 0.597 ± 0.044 | 0.906 ± 0.006 | 0.784 ± 0.007 | 0.661 ± 0.043 | 0.902 ± 0.005 | 0.776 ± 0.008 | 0.631 ± 0.020 |
| Emb-sym only HGB | 0.834 ± 0.007 | 0.711 ± 0.026 | 0.599 ± 0.045 | 0.861 ± 0.006 | 0.761 ± 0.017 | 0.651 ± 0.034 | 0.867 ± 0.004 | 0.756 ± 0.018 | 0.612 ± 0.030 |
| RandomVec-sym + path HGB (control) | 0.802 ± 0.008 | 0.684 ± 0.023 | 0.549 ± 0.064 | 0.834 ± 0.008 | 0.714 ± 0.009 | 0.588 ± 0.048 | 0.838 ± 0.001 | 0.710 ± 0.030 | 0.544 ± 0.065 |
| RandomVec-sym + path FFNN-fixed (control) | 0.899 ± 0.008 | 0.638 ± 0.024 | 0.503 ± 0.079 | 0.885 ± 0.007 | 0.665 ± 0.011 | 0.540 ± 0.031 | 0.883 ± 0.002 | 0.652 ± 0.012 | 0.507 ± 0.025 |
| Identity MLP (FFNN-fixed on multi-hot) | 0.976 ± 0.001 | 0.671 ± 0.009 | 0.498 ± 0.002 | 0.972 ± 0.002 | 0.659 ± 0.031 | 0.500 ± 0.000 | 0.982 ± 0.001 | 0.714 ± 0.016 | 0.500 ± 0.000 |
| Identity LR | 0.817 ± 0.006 | 0.715 ± 0.022 | 0.500 ± 0.000 | 0.844 ± 0.007 | 0.730 ± 0.019 | 0.500 ± 0.000 | 0.864 ± 0.004 | 0.741 ± 0.014 | 0.500 ± 0.000 |
| Degree (product) | 0.815 ± 0.005 | 0.715 ± 0.022 | 0.500 ± 0.000 | 0.842 ± 0.009 | 0.730 ± 0.019 | 0.500 ± 0.000 | 0.864 ± 0.004 | 0.741 ± 0.014 | 0.500 ± 0.000 |
| Morgan FP HGB | 0.886 ± 0.006 | 0.732 ± 0.031 | 0.564 ± 0.051 | 0.901 ± 0.007 | 0.789 ± 0.015 | 0.662 ± 0.054 | 0.899 ± 0.004 | 0.772 ± 0.012 | 0.621 ± 0.023 |
| SimKNN (k=10) | 0.759 ± 0.013 | 0.695 ± 0.027 | 0.608 ± 0.037 | 0.786 ± 0.008 | 0.736 ± 0.020 | 0.658 ± 0.055 | 0.792 ± 0.006 | 0.736 ± 0.024 | 0.649 ± 0.056 |
| Raw similarity S | 0.577 ± 0.008 | 0.574 ± 0.038 | 0.588 ± 0.086 | 0.594 ± 0.008 | 0.595 ± 0.024 | 0.601 ± 0.082 | 0.614 ± 0.006 | 0.610 ± 0.012 | 0.610 ± 0.034 |
| Path only HGB | 0.628 ± 0.009 | 0.602 ± 0.015 | 0.576 ± 0.041 | 0.656 ± 0.007 | 0.632 ± 0.011 | 0.621 ± 0.050 | 0.621 ± 0.000 | 0.612 ± 0.016 | 0.602 ± 0.040 |
| [leaky] DrugBank global degree product | 0.811 ± 0.004 | 0.807 ± 0.014 | 0.817 ± 0.041 | 0.836 ± 0.008 | 0.836 ± 0.014 | 0.830 ± 0.050 | 0.843 ± 0.003 | 0.844 ± 0.007 | 0.841 ± 0.016 |

MCC (mean over folds) for the same methods:

| Method | D1 warm | D1 S1 | D1 S2 | D2 warm | D2 S1 | D2 S2 | MP warm | MP S1 | MP S2 |
|---|---|---|---|---|---|---|---|---|---|
| DDNet-sym RF | 0.442 | 0.289 | 0.100 | 0.505 | 0.390 | 0.261 | – | – | – |
| DDNet-sym HGB | 0.493 | 0.300 | 0.119 | 0.552 | 0.385 | 0.229 | 0.557 | 0.370 | 0.182 |
| DDNet-sym FFNN-fixed | 0.629 | 0.284 | 0.083 | 0.644 | 0.367 | 0.179 | 0.623 | 0.333 | 0.122 |
| DDNet-concat both-orient. HGB | 0.604 | 0.355 | 0.107 | 0.643 | 0.419 | 0.231 | 0.631 | 0.390 | 0.202 |
| Emb-sym only HGB | 0.485 | 0.300 | 0.119 | 0.553 | 0.384 | 0.216 | 0.558 | 0.367 | 0.171 |
| RandomVec-sym + path HGB (control) | 0.426 | 0.251 | 0.067 | 0.494 | 0.312 | 0.108 | 0.498 | 0.281 | 0.070 |
| RandomVec-sym + path FFNN-fixed (control) | 0.628 | 0.198 | -0.020 | 0.598 | 0.248 | 0.044 | 0.592 | 0.205 | 0.022 |
| Identity MLP (FFNN-fixed on multi-hot) | 0.846 | 0.263 | 0.000 | 0.822 | 0.250 | 0.000 | 0.869 | 0.306 | 0.000 |
| Identity LR | 0.435 | 0.286 | 0.000 | 0.494 | 0.334 | 0.000 | 0.542 | 0.326 | 0.000 |
| Degree (product) | 0.457 | 0.305 | 0.000 | 0.510 | 0.333 | 0.000 | 0.547 | 0.335 | 0.000 |
| Morgan FP HGB | 0.598 | 0.325 | 0.082 | 0.628 | 0.414 | 0.224 | 0.618 | 0.393 | 0.178 |
| SimKNN (k=10) | 0.382 | 0.273 | 0.144 | 0.429 | 0.342 | 0.229 | 0.429 | 0.348 | 0.227 |
| Raw similarity S | 0.117 | 0.107 | 0.102 | 0.156 | 0.159 | 0.159 | 0.155 | 0.140 | 0.138 |
| Path only HGB | 0.154 | 0.113 | 0.062 | 0.219 | 0.173 | 0.163 | 0.187 | 0.176 | 0.180 |
| [leaky] DrugBank global degree product | 0.455 | 0.449 | 0.459 | 0.491 | 0.492 | 0.486 | 0.514 | 0.515 | 0.513 |

Folds used: D1: warm 5, node 5, D2: warm 5, node 5, MP: warm 3, node 5


Findings:
* **Warm split: identity memorisation wins.** Mirrored pairs can no longer leak here, but the dense matrix can still be completed from drug identity. With no features at all, the identity MLP reaches **AUROC 0.97–0.98 and MCC 0.82–0.87**. The best DDNet variants (FFNN-fixed on symmetric features, or HGB on both-orientation concatenation) reach 0.90–0.91 / MCC 0.60–0.64. That is no better than random vectors plus path features with the same FFNN (0.88–0.90) or Morgan fingerprints with HGB (0.89–0.90). Degree and identity LR reach 0.82–0.86.
* **S1, one new drug: DDNet ties Morgan fingerprints and is only slightly above degree.**
  * DDNet: 0.71–0.76 AUROC with symmetric features and HGB; 0.75–0.78 with the concatenated layout.
  * Degree (the known drug's training rate): 0.72–0.74.
  * Morgan FP + HGB: 0.73–0.79.
  * The best DDNet variant beats degree by 0.03–0.05 AUROC and is within ±0.015 of Morgan, which is inside the fold std. MCC is 0.30–0.42 for every method.
  * The random-vector control drops to 0.68–0.71. So the node2vec vectors do carry some structural-neighbourhood signal, but no more than the fingerprint they are built from.
* **S2, both drugs new: all structure-based methods are weak.**
  * AUROC: DDNet 0.56–0.67, Morgan 0.56–0.66, SimKNN 0.61–0.66, raw S 0.59–0.61, path features alone 0.58–0.62.
  * MCC is 0.08–0.26, and degree and identity are exactly 0.5.
  * The fold std is 0.03–0.09, so DDNet, fingerprints and SimKNN cannot be told apart.
* **Adding the path features to the embeddings changes nothing** (Emb-sym-only HGB ≈ DDNet-sym HGB in every split). The symmetric Hadamard/abs-difference encoding is usually slightly *worse* than the paper's own concatenation, provided the latter is trained on both orientations and averaged.
* **Popularity outscores every structure-based method.** The *external* DrugBank degree of the two drugs gives AUROC 0.81–0.84 and MCC 0.46–0.52 in every split, including S2. That is 0.15–0.25 AUROC above any structure-based method on truly new drugs. Most of the learnable signal in these benchmarks is how well-studied a drug is, which is not known for a truly new drug. Any cold-start claim on this data must control for degree, for example with degree-matched negatives or degree-stratified metrics.

## 5. Cross-dataset D1 → D2, the paper's "new drug" experiment (task 4)


### hop-1 (train all D1 ordered pairs, test all D2 ordered pairs; D2 pair types: both drugs also in D1 = 2116, one = 12328, none = 17956)

| Method | all F1 | all MCC | all AUROC | all AUPRC(AP) | all AUPRC(hard) | both-shared AUROC | one-shared AUROC | none-shared AUROC | none-shared MCC | paper MCC / F1 / AUPRC / Prec |
|---|---|---|---|---|---|---|---|---|---|---|
| DDNet RF | 0.578 | 0.033 | 0.526 | 0.600 | 0.707 | 0.510 | 0.532 | 0.520 | 0.026 | 0.16 / 0.80 / 0.83 / 0.67 |
| DDNet GB(HGB) | 0.619 | -0.029 | 0.464 | 0.547 | 0.718 | 0.569 | 0.490 | 0.451 | -0.048 | 0.20 / 0.76 / 0.83 / 0.73 |
| DDNet FFNN-orig | 0.000 | -0.063 | 0.460 | 0.540 | 0.291 | 0.441 | 0.448 | 0.467 | -0.063 | 0.13 / 0.64 / 0.79 / 0.73 |
| DDNet FFNN-fixed | 0.698 | -0.011 | 0.451 | 0.552 | 0.764 | 0.417 | 0.444 | 0.460 | -0.024 |  |
| Path only (HGB) | 0.702 | 0.100 | 0.606 | 0.679 | 0.768 | 0.618 | 0.601 | 0.600 | 0.106 |  |
| Embeddings only (HGB) | 0.458 | -0.060 | 0.453 | 0.548 | 0.647 | 0.465 | 0.459 | 0.455 | -0.061 |  |
| Degree from D1 (product) | 0.724 | 0.080 | 0.615 | 0.690 | 0.782 | 0.779 | 0.662 | 0.500 | 0.000 |  |
| Raw similarity S | 0.580 | 0.149 | 0.586 | 0.628 | 0.730 | 0.574 | 0.610 | 0.572 | 0.136 |  |
| All-positive | 0.736 | 0.000 | 0.500 | 0.582 | 0.791 | 0.500 | 0.500 | 0.500 | 0.000 |  |

### hop-2 (train all D1 ordered pairs, test all D2 ordered pairs; D2 pair types: both drugs also in D1 = 2116, one = 12328, none = 17956)

| Method | all F1 | all MCC | all AUROC | all AUPRC(AP) | all AUPRC(hard) | both-shared AUROC | one-shared AUROC | none-shared AUROC | none-shared MCC | paper MCC / F1 / AUPRC / Prec |
|---|---|---|---|---|---|---|---|---|---|---|
| DDNet RF | 0.606 | -0.066 | 0.444 | 0.540 | 0.709 | 0.478 | 0.433 | 0.452 | -0.055 | 0.13 / 0.72 / 0.78 / 0.60 |
| DDNet GB(HGB) | 0.358 | -0.025 | 0.493 | 0.574 | 0.627 | 0.559 | 0.493 | 0.487 | -0.028 | 0.06 / 0.12 / 0.64 / 0.68 |
| DDNet FFNN-orig | 0.736 | 0.009 | 0.405 | 0.509 | 0.791 | 0.378 | 0.411 | 0.405 | 0.008 | 0.10 / 0.79 / 0.83 / 0.66 |
| DDNet FFNN-fixed | 0.734 | 0.016 | 0.443 | 0.534 | 0.789 | 0.582 | 0.463 | 0.427 | -0.011 |  |
| Path only (HGB) | 0.702 | 0.100 | 0.606 | 0.679 | 0.768 | 0.618 | 0.601 | 0.600 | 0.106 |  |
| Embeddings only (HGB) | 0.118 | 0.035 | 0.534 | 0.610 | 0.630 | 0.512 | 0.527 | 0.527 | 0.031 |  |
| Degree from D1 (product) | 0.724 | 0.080 | 0.615 | 0.690 | 0.782 | 0.779 | 0.662 | 0.500 | 0.000 |  |
| Raw similarity S | 0.580 | 0.149 | 0.586 | 0.628 | 0.730 | 0.574 | 0.610 | 0.572 | 0.136 |  |
| All-positive | 0.736 | 0.000 | 0.500 | 0.582 | 0.791 | 0.500 | 0.500 | 0.500 | 0.000 |  |

### Alignment of the 46 shared drugs

| hop | cos(same drug D1 vs D2) mean [min,max] | cos(different drugs) mean | Procrustes residual (true) | Procrustes residual (perm. null mean, 5%) | perm p | held-out cos matched / mismatched | held-out mean rank of true match (chance) |
|---|---|---|---|---|---|---|---|
| 1 | 0.076 [-0.08, 0.28] | 0.078 | 0.294 | 0.591, 0.568 | 0.005 | 0.660 / 0.480 | 4.5 (12.0) |
| 2 | -0.019 [-0.07, 0.04] | -0.022 | 0.254 | 0.596, 0.553 | 0.005 | 0.900 / 0.723 | 6.9 (12.0) |

### Are per-ego embeddings comparable within a dataset?

| dataset/hop | Spearman(cos emb, S) | NN@10 overlap emb vs S (chance) | AUROC of cos(emb) for DDI | mean off-diag cos | Spearman(norm, sim-degree) | Spearman(norm, DDI-degree) | global spectral emb: Spearman(cos,S) / NN@10 |
|---|---|---|---|---|---|---|---|
| D1_hop1 | 0.726 | 3.04 (0.56) | 0.603 | 0.437 | 0.818 | 0.236 | 0.273 / 4.17 |
| D1_hop2 | 0.542 | 1.41 (0.56) | 0.563 | 0.638 | -0.058 | -0.003 | 0.273 / 4.17 |
| D2_hop1 | 0.689 | 2.69 (0.56) | 0.612 | 0.446 | 0.745 | 0.161 | 0.262 / 4.42 |
| D2_hop2 | 0.532 | 1.47 (0.56) | 0.618 | 0.690 | -0.140 | -0.191 | 0.262 / 4.42 |
| MP_hop1 | 0.694 | 5.67 (0.35) | 0.596 | 0.130 | 0.892 | 0.230 | 0.119 / 6.17 |

hop1_vs_hop2_same_drug_cos_D1: -0.001

hop1_vs_hop2_same_drug_cos_D2: 0.670


Findings:
* **As in the paper** (train on all D1 ordered pairs with D1 embeddings, test on all D2 ordered pairs with D2 embeddings), every DDNet model is **at or below chance**:

  | Model | hop-1 AUROC | hop-1 MCC | hop-2 AUROC | hop-2 MCC |
  |---|---|---|---|---|
  | RF | 0.53 | 0.03 | 0.44 | −0.07 |
  | GB (HGB) | 0.46 | −0.03 | 0.49 | −0.03 |
  | FFNN-orig | 0.46 | −0.06 | 0.41 | 0.01 |
  | FFNN-fixed | 0.45 | −0.01 | 0.44 | 0.02 |

  * FFNN-orig predicts *no* positives at hop-1 (F1 0) and *all* positives at hop-2 (F1 0.736).
  * The only transferable signals are the path features (AUROC 0.61, MCC 0.10) and each drug's degree carried over from D1. The degree gives AUROC 0.78 on pairs where both drugs also appear in D1, 0.66 where one does, and exactly 0.5 for the truly new pairs.
* **The report's figures here are no better than the all-positive predictor.** It gave F1 0.64–0.80 and "AUPRC" 0.78–0.83. The all-positive predictor scores F1 0.736 and hard-label AUPRC 0.79 on this test set. The report's MCC of 0.06–0.20 could not be reproduced; ours is −0.07 to 0.03.
* **Why it fails: the D1 and D2 embedding spaces are not aligned.**
  * For the 46 drugs in both sets, cos(D1 vector, D2 vector) averages 0.076 at hop-1 and −0.019 at hop-2. That is no different from pairs of *different* drugs (0.078 / −0.022; Mann–Whitney p = 0.60 / 0.39). Within D1 alone the mean cosine is 0.49 at hop-1.
  * A classifier fit on D1's coordinates therefore sees D2 vectors as arbitrary points.
  * The *relative* geometry is partly shared. An orthogonal Procrustes map fit on 23 shared drugs and tested on the other 23 raises the matched cosine to 0.66 (mismatched 0.48). It ranks the true partner 4.5th of 23 on average (chance 12); the permutation p is 0.005. The embeddings would need an explicit alignment step, a single shared embedding, or inductive features to transfer at all.
* **Within one dataset, the per-ego embeddings are comparable.** Across pairs, cos(e_a, e_b) tracks the similarity S: Spearman 0.73 / 0.69 / 0.69 for D1 / D2 / MP at hop-1, and 0.53–0.54 at hop-2. The top-10 cosine neighbours overlap the top-10 S neighbours on 3.0 drugs for D1 (chance 0.56) and 5.7 for MP (chance 0.35).
  * The vector norm tracks the drug's similarity-graph degree, i.e. ego-subgraph size: Spearman 0.82 / 0.74 / 0.89 at hop-1.
  * The mechanism is plausibly that word2vec runs on near-identical ego subgraphs with the same seed give near-identical coordinates. Because the geometry is kept, the vectors behave like a smoothed fingerprint neighbourhood. That fits their cold-start results being on par with Morgan fingerprints (section 4).
  * The 32-D Laplacian-eigenmap "global" embedding listed in the table is a weak reference only; it was not tuned.

## 6. What this means for publication (empirical side only)

**Can the current results be published? No.** Every headline number has a simpler explanation:
* The MP FFNN (F1 0.85 / MCC 0.72) and the D1/D2 FFNN (F1 0.90–0.91) are matched or beaten under the *same* protocol by random identity vectors (MCC 0.75–0.77) and by an identity MLP (0.82–0.88).
* The RF/GB numbers are matched by per-drug degree (AUROC 0.82–0.87).
* The "new drug" result is at chance.
* The comparison with HIN-DDI is not like for like: different protocol, a different label list, and hard-label AUPRC.

Any reviewer who adds a degree baseline or a cold-start split will see this.

**Can moderate effort turn it into a paper? Not by keeping the DDNet idea and fixing the evaluation.** Under S1/S2 the method adds nothing over Morgan fingerprints: it is ±0.015 AUROC of them in S1 and in S2, where everything sits at 0.56–0.67. Only two routes look viable, and both are new work rather than polishing:
1. **An inductive DDI model built on CROssBAR's heterogeneous knowledge graph.**
   - Features: drug→target/enzyme/transporter/pathway/disease/phenotype edges, which exist for new drugs before any DDI is known. The one author who is a CROssBAR co-author, and the new METU MongoDB, are an advantage here.
   - Data: full DrugBank, about 4.4k drugs and 1.38M pairs, not 180-drug subsets.
   - Evaluation: S1/S2 (and scaffold-based) splits, with degree-controlled negatives or degree-stratified reporting.
   - Baselines: degree, identity MLP, Morgan+GBM, SimKNN, and current GNN/KG-embedding DDI models.

   The bar is set by the S2 numbers here: structure-only features give AUROC 0.60–0.67, while external degree alone gives 0.84.
2. **A short benchmark or negative-result note.** It would quantify how ordered-pair leakage, self pairs, hard-label AUPRC, dense popularity-biased subsets and identity memorisation inflate DDI link-prediction results; this re-evaluation already contains most of the tables. It is only publishable if the literature does not already cover it. Several cold-start and leakage critiques of DDI prediction exist, so the literature review must check this before any writing.

Before any submission, fix these regardless of route: the FFNN softmax bug, the unreproducible D1/D2 similarity matrix, the 0.3 vs 0.5 threshold mismatch, the drug count (176 vs 179), and the IQR filter that drops about 60% of drugs.

## 7. Caveats, and what I could not run

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

### Reproduction tables (as-in-paper protocol)

#### MP — `mp_128_1.2_1.2.txt`, 289 drugs, 83521 ordered rows (incl. self), positive rate 0.453, 1 folds

| Method | F1 | MCC | Precision | AUPRC (paper way, hard labels) | AUROC (hard) | AUROC (scores) | AUPRC (scores, AP) | fold-1 = 80/20 split F1 / MCC | paper MCC / F1 / AUPRC / Prec |
|---|---|---|---|---|---|---|---|---|---|
| DDNet RF | 0.819 | 0.663 | 0.803 | 0.856 | 0.833 | 0.912 | 0.894 | 0.819 / 0.663 | 0.26 / 0.53 / 0.67 / – |
| DDNet GB(HGB) | 0.787 | 0.606 | 0.775 | 0.833 | 0.804 | 0.890 | 0.872 | 0.787 / 0.606 | – / 0.73 / 0.78 / – |
| DDNet FFNN-orig | 0.754 | 0.515 | 0.623 | 0.799 | 0.738 | 0.883 | 0.859 | 0.754 / 0.515 | 0.72 / 0.85 / 0.87 / – |
| DDNet FFNN-fixed | 0.829 | 0.691 | 0.841 | 0.871 | 0.845 | 0.933 | 0.927 | 0.829 / 0.691 |  |
| All-positive | 0.624 | 0.000 | 0.453 | 0.727 | 0.500 | 0.500 | 0.453 | 0.624 / 0.000 |  |
| Degree (product) | 0.759 | 0.557 | 0.754 | 0.813 | 0.779 | 0.870 | 0.850 | 0.759 / 0.557 |  |
| Degree (sum) | 0.753 | 0.552 | 0.762 | 0.811 | 0.776 | 0.865 | 0.849 | 0.753 / 0.552 |  |
| Raw similarity S | 0.628 | 0.139 | 0.483 | 0.714 | 0.551 | 0.605 | 0.544 | 0.628 / 0.139 |  |
| One-hot identity LR | 0.755 | 0.545 | 0.740 | 0.807 | 0.773 | 0.866 | 0.848 | 0.755 / 0.545 |  |
| One-hot identity RF | 0.748 | 0.548 | 0.767 | 0.809 | 0.773 | 0.861 | 0.835 | 0.748 / 0.548 |  |
| Path features only (HGB) | 0.497 | 0.219 | 0.617 | 0.649 | 0.601 | 0.641 | 0.608 | 0.497 / 0.219 |  |
| Embeddings only (HGB) | 0.804 | 0.638 | 0.793 | 0.846 | 0.819 | 0.903 | 0.886 | 0.804 / 0.638 |  |
| RandomVec + path HGB (control) [1 folds] | 0.777 | 0.588 | 0.766 | 0.825 | 0.794 | 0.883 | 0.866 | 0.777 / 0.588 |  |
| RandomVec + path FFNN-fixed (control) [1 folds] | 0.878 | 0.773 | 0.859 | 0.902 | 0.888 | 0.955 | 0.949 | 0.878 / 0.773 |  |
| Identity MLP (FFNN-fixed on one-hot) [1 folds] | 0.933 | 0.880 | 0.952 | 0.953 | 0.938 | 0.983 | 0.978 | 0.933 / 0.880 |  |

FFNN-orig predicted-positive rate: 0.695


#### D1h1 — `cb1_hop1_128_1.2_1.2.txt`, 179 drugs, 32041 ordered rows (incl. self), positive rate 0.579, 3 folds

| Method | F1 | MCC | Precision | AUPRC (paper way, hard labels) | AUROC (hard) | AUROC (scores) | AUPRC (scores, AP) | fold-1 = 80/20 split F1 / MCC | paper MCC / F1 / AUPRC / Prec |
|---|---|---|---|---|---|---|---|---|---|
| DDNet RF | 0.806 ± 0.009 | 0.546 ± 0.020 | 0.812 ± 0.012 | 0.864 ± 0.007 | 0.774 ± 0.010 | 0.857 ± 0.006 | 0.891 ± 0.006 | 0.802 / 0.534 | 0.37 / 0.82 / 0.85 / 0.71 |
| DDNet GB(HGB) | 0.815 ± 0.007 | 0.537 ± 0.016 | 0.780 ± 0.009 | 0.859 ± 0.006 | 0.762 ± 0.008 | 0.862 ± 0.005 | 0.898 ± 0.005 | 0.808 / 0.520 | 0.46 / 0.84 / 0.88 / 0.80 |
| DDNet FFNN-orig | 0.768 ± 0.003 | 0.324 ± 0.009 | 0.641 ± 0.005 | 0.812 ± 0.002 | 0.613 ± 0.005 | 0.793 ± 0.009 | 0.826 ± 0.010 | 0.766 / 0.321 | 0.71 / 0.90 / 0.92 / 0.86 |
| DDNet FFNN-fixed | 0.867 ± 0.005 | 0.709 ± 0.010 | 0.911 ± 0.017 | 0.919 ± 0.006 | 0.858 ± 0.006 | 0.937 ± 0.002 | 0.956 ± 0.002 | 0.866 / 0.699 |  |
| All-positive | 0.732 ± 0.002 | 0.000 ± 0.000 | 0.577 ± 0.003 | 0.789 ± 0.001 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.577 ± 0.003 | 0.730 / 0.000 |  |
| Degree (product) | 0.752 ± 0.006 | 0.469 ± 0.015 | 0.807 ± 0.012 | 0.841 ± 0.006 | 0.737 ± 0.008 | 0.818 ± 0.004 | 0.864 ± 0.006 | 0.746 / 0.455 |  |
| Degree (sum) | 0.734 ± 0.021 | 0.472 ± 0.006 | 0.832 ± 0.030 | 0.844 ± 0.005 | 0.737 ± 0.003 | 0.821 ± 0.004 | 0.867 ± 0.005 | 0.743 / 0.464 |  |
| Raw similarity S | 0.579 ± 0.021 | 0.113 ± 0.011 | 0.636 ± 0.014 | 0.719 ± 0.004 | 0.557 ± 0.005 | 0.572 ± 0.008 | 0.617 ± 0.009 | 0.587 / 0.103 |  |
| One-hot identity LR | 0.775 ± 0.005 | 0.440 ± 0.011 | 0.745 ± 0.006 | 0.832 ± 0.004 | 0.715 ± 0.005 | 0.816 ± 0.004 | 0.865 ± 0.005 | 0.773 / 0.435 |  |
| One-hot identity RF | 0.754 ± 0.004 | 0.446 ± 0.009 | 0.781 ± 0.007 | 0.833 ± 0.004 | 0.725 ± 0.005 | 0.808 ± 0.004 | 0.854 ± 0.005 | 0.753 / 0.442 |  |
| Path features only (HGB) | 0.724 ± 0.003 | 0.231 ± 0.002 | 0.645 ± 0.004 | 0.786 ± 0.002 | 0.603 ± 0.001 | 0.687 ± 0.006 | 0.752 ± 0.010 | 0.721 / 0.229 |  |
| Embeddings only (HGB) | 0.834 ± 0.005 | 0.594 ± 0.012 | 0.813 ± 0.007 | 0.876 ± 0.004 | 0.793 ± 0.006 | 0.888 ± 0.006 | 0.916 ± 0.006 | 0.831 / 0.588 |  |
| Embeddings only (RF) | 0.808 ± 0.005 | 0.548 ± 0.012 | 0.812 ± 0.009 | 0.864 ± 0.005 | 0.775 ± 0.007 | 0.858 ± 0.004 | 0.892 ± 0.005 | 0.804 / 0.537 |  |
| RandomVec + path HGB (control) [5 folds] | 0.814 ± 0.004 | 0.542 ± 0.010 | 0.792 ± 0.007 | 0.862 ± 0.003 | 0.767 ± 0.006 | 0.863 ± 0.002 | 0.903 ± 0.002 | 0.809 / 0.531 |  |
| RandomVec + path FFNN-fixed (control) [5 folds] | 0.899 ± 0.005 | 0.774 ± 0.007 | 0.929 ± 0.009 | 0.938 ± 0.002 | 0.890 ± 0.003 | 0.956 ± 0.001 | 0.967 ± 0.001 | 0.903 / 0.781 |  |
| Identity MLP (FFNN-fixed on one-hot) [5 folds] | 0.935 ± 0.001 | 0.848 ± 0.003 | 0.941 ± 0.008 | 0.956 ± 0.002 | 0.925 ± 0.003 | 0.971 ± 0.001 | 0.968 ± 0.003 | 0.936 / 0.849 |  |

FFNN-orig predicted-positive rate: 0.863 ± 0.006


#### D1h2 — `cb1_hop2_128_1.2_1.2.txt`, 179 drugs, 32041 ordered rows (incl. self), positive rate 0.579, 2 folds

| Method | F1 | MCC | Precision | AUPRC (paper way, hard labels) | AUROC (hard) | AUROC (scores) | AUPRC (scores, AP) | fold-1 = 80/20 split F1 / MCC | paper MCC / F1 / AUPRC / Prec |
|---|---|---|---|---|---|---|---|---|---|
| DDNet RF | 0.783 ± 0.007 | 0.484 ± 0.014 | 0.778 ± 0.002 | 0.844 ± 0.003 | 0.742 ± 0.006 | 0.825 ± 0.005 | 0.864 ± 0.002 | 0.788 / 0.494 | 0.32 / 0.79 / 0.83 / 0.70 |
| DDNet GB(HGB) | 0.803 ± 0.000 | 0.507 ± 0.003 | 0.766 ± 0.007 | 0.850 ± 0.002 | 0.747 ± 0.003 | 0.842 ± 0.001 | 0.882 ± 0.003 | 0.803 / 0.505 | 0.48 / 0.83 / 0.87 / 0.79 |
| DDNet FFNN-orig | 0.738 ± 0.000 | 0.143 ± 0.012 | 0.587 ± 0.000 | 0.792 ± 0.000 | 0.522 ± 0.003 | 0.687 ± 0.000 | 0.738 ± 0.002 | 0.738 / 0.151 | 0.24 / 0.79 / 0.82 / 0.65 |
| DDNet FFNN-fixed | 0.833 ± 0.004 | 0.622 ± 0.014 | 0.859 ± 0.013 | 0.889 ± 0.006 | 0.814 ± 0.007 | 0.897 ± 0.007 | 0.929 ± 0.005 | 0.830 / 0.613 |  |
| All-positive | 0.731 ± 0.001 | 0.000 ± 0.000 | 0.576 ± 0.002 | 0.788 ± 0.001 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.576 ± 0.002 | 0.730 / 0.000 |  |
| Degree (product) | 0.749 ± 0.005 | 0.461 ± 0.008 | 0.801 ± 0.005 | 0.838 ± 0.004 | 0.733 ± 0.004 | 0.816 ± 0.002 | 0.860 ± 0.001 | 0.746 / 0.455 |  |
| Degree (sum) | 0.746 ± 0.003 | 0.470 ± 0.008 | 0.815 ± 0.006 | 0.841 ± 0.004 | 0.738 ± 0.004 | 0.818 ± 0.003 | 0.864 ± 0.002 | 0.743 / 0.464 |  |
| Raw similarity S | 0.591 ± 0.006 | 0.108 ± 0.007 | 0.628 ± 0.005 | 0.720 ± 0.004 | 0.554 ± 0.004 | 0.568 ± 0.002 | 0.612 ± 0.003 | 0.587 / 0.103 |  |

FFNN-orig predicted-positive rate: 0.975 ± 0.004


#### D2h1 — `cb2_hop1_128_1.2_1.2.txt`, 180 drugs, 32400 ordered rows (incl. self), positive rate 0.582, 3 folds

| Method | F1 | MCC | Precision | AUPRC (paper way, hard labels) | AUROC (hard) | AUROC (scores) | AUPRC (scores, AP) | fold-1 = 80/20 split F1 / MCC | paper MCC / F1 / AUPRC / Prec |
|---|---|---|---|---|---|---|---|---|---|
| DDNet RF | 0.826 ± 0.007 | 0.582 ± 0.016 | 0.826 ± 0.006 | 0.877 ± 0.005 | 0.791 ± 0.008 | 0.880 ± 0.009 | 0.910 ± 0.008 | 0.824 / 0.584 | 0.49 / 0.83 / 0.87 / 0.78 |
| DDNet GB(HGB) | 0.836 ± 0.005 | 0.588 ± 0.010 | 0.808 ± 0.009 | 0.876 ± 0.005 | 0.788 ± 0.005 | 0.879 ± 0.007 | 0.912 ± 0.008 | 0.832 / 0.580 | 0.52 / 0.85 / 0.88 / 0.81 |
| DDNet FFNN-orig | 0.783 ± 0.005 | 0.373 ± 0.005 | 0.670 ± 0.006 | 0.823 ± 0.004 | 0.645 ± 0.001 | 0.801 ± 0.008 | 0.839 ± 0.011 | 0.777 / 0.368 | 0.68 / 0.91 / 0.93 / 0.88 |
| DDNet FFNN-fixed | 0.875 ± 0.004 | 0.716 ± 0.005 | 0.909 ± 0.012 | 0.922 ± 0.002 | 0.862 ± 0.003 | 0.937 ± 0.001 | 0.956 ± 0.001 | 0.871 / 0.721 |  |
| All-positive | 0.738 ± 0.005 | 0.000 ± 0.000 | 0.584 ± 0.006 | 0.792 ± 0.003 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.584 ± 0.006 | 0.732 / 0.000 |  |
| Degree (product) | 0.780 ± 0.004 | 0.513 ± 0.015 | 0.826 ± 0.012 | 0.859 ± 0.006 | 0.760 ± 0.008 | 0.842 ± 0.008 | 0.882 ± 0.008 | 0.779 / 0.514 |  |
| Degree (sum) | 0.787 ± 0.012 | 0.516 ± 0.020 | 0.817 ± 0.008 | 0.859 ± 0.006 | 0.760 ± 0.009 | 0.843 ± 0.008 | 0.884 ± 0.008 | 0.789 / 0.518 |  |
| Raw similarity S | 0.632 ± 0.006 | 0.156 ± 0.016 | 0.655 ± 0.013 | 0.747 ± 0.007 | 0.579 ± 0.009 | 0.584 ± 0.010 | 0.628 ± 0.012 | 0.626 / 0.146 |  |
| One-hot identity LR | 0.798 ± 0.006 | 0.488 ± 0.016 | 0.767 ± 0.011 | 0.849 ± 0.006 | 0.738 ± 0.008 | 0.839 ± 0.008 | 0.882 ± 0.008 | 0.794 / 0.481 |  |
| One-hot identity RF | 0.770 ± 0.007 | 0.508 ± 0.021 | 0.834 ± 0.011 | 0.858 ± 0.006 | 0.758 ± 0.010 | 0.835 ± 0.008 | 0.881 ± 0.007 | 0.775 / 0.521 |  |
| Path features only (HGB) | 0.738 ± 0.000 | 0.280 ± 0.006 | 0.673 ± 0.005 | 0.798 ± 0.002 | 0.629 ± 0.002 | 0.698 ± 0.001 | 0.759 ± 0.006 | 0.737 / 0.287 |  |
| Embeddings only (HGB) | 0.850 ± 0.003 | 0.626 ± 0.009 | 0.828 ± 0.004 | 0.888 ± 0.002 | 0.809 ± 0.005 | 0.898 ± 0.005 | 0.923 ± 0.007 | 0.850 / 0.633 |  |
| Embeddings only (RF) | 0.827 ± 0.007 | 0.583 ± 0.016 | 0.826 ± 0.009 | 0.877 ± 0.005 | 0.791 ± 0.008 | 0.879 ± 0.008 | 0.909 ± 0.009 | 0.826 / 0.584 |  |
| RandomVec + path HGB (control) [5 folds] | 0.830 ± 0.006 | 0.576 ± 0.017 | 0.805 ± 0.008 | 0.872 ± 0.005 | 0.784 ± 0.008 | 0.878 ± 0.007 | 0.913 ± 0.005 | 0.824 / 0.563 |  |
| RandomVec + path FFNN-fixed (control) [5 folds] | 0.893 ± 0.003 | 0.752 ± 0.006 | 0.911 ± 0.018 | 0.930 ± 0.004 | 0.878 ± 0.005 | 0.950 ± 0.003 | 0.963 ± 0.003 | 0.891 / 0.755 |  |
| Identity MLP (FFNN-fixed on one-hot) [5 folds] | 0.925 ± 0.002 | 0.821 ± 0.006 | 0.926 ± 0.005 | 0.947 ± 0.002 | 0.911 ± 0.003 | 0.964 ± 0.003 | 0.963 ± 0.004 | 0.928 / 0.830 |  |

FFNN-orig predicted-positive rate: 0.821 ± 0.003


#### D2h2 — `cb2_hop2_128_1.2_1.2.txt`, 180 drugs, 32400 ordered rows (incl. self), positive rate 0.582, 2 folds

| Method | F1 | MCC | Precision | AUPRC (paper way, hard labels) | AUROC (hard) | AUROC (scores) | AUPRC (scores, AP) | fold-1 = 80/20 split F1 / MCC | paper MCC / F1 / AUPRC / Prec |
|---|---|---|---|---|---|---|---|---|---|
| DDNet RF | 0.818 ± 0.008 | 0.556 ± 0.015 | 0.808 ± 0.012 | 0.868 ± 0.008 | 0.777 ± 0.008 | 0.862 ± 0.010 | 0.892 ± 0.013 | 0.812 / 0.546 | 0.28 / 0.81 / 0.84 / 0.71 |
| DDNet GB(HGB) | 0.827 ± 0.007 | 0.564 ± 0.011 | 0.795 ± 0.011 | 0.869 ± 0.006 | 0.776 ± 0.006 | 0.868 ± 0.006 | 0.900 ± 0.010 | 0.823 / 0.557 | 0.44 / 0.84 / 0.87 / 0.78 |
| DDNet FFNN-orig | 0.760 ± 0.001 | 0.266 ± 0.024 | 0.628 ± 0.001 | 0.806 ± 0.001 | 0.583 ± 0.014 | 0.744 ± 0.003 | 0.784 ± 0.006 | 0.759 / 0.283 | 0.54 / 0.86 / 0.89 / 0.79 |
| DDNet FFNN-fixed | 0.863 ± 0.001 | 0.678 ± 0.008 | 0.874 ± 0.030 | 0.906 ± 0.009 | 0.840 ± 0.008 | 0.923 ± 0.002 | 0.945 ± 0.006 | 0.864 / 0.672 |  |
| All-positive | 0.736 ± 0.006 | 0.000 ± 0.000 | 0.583 ± 0.008 | 0.791 ± 0.004 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.583 ± 0.008 | 0.732 / 0.000 |  |
| Degree (product) | 0.782 ± 0.004 | 0.521 ± 0.010 | 0.830 ± 0.013 | 0.860 ± 0.007 | 0.764 ± 0.006 | 0.846 ± 0.007 | 0.883 ± 0.011 | 0.779 / 0.514 |  |
| Degree (sum) | 0.794 ± 0.006 | 0.526 ± 0.011 | 0.817 ± 0.012 | 0.861 ± 0.007 | 0.765 ± 0.006 | 0.846 ± 0.007 | 0.885 ± 0.011 | 0.789 / 0.518 |  |
| Raw similarity S | 0.630 ± 0.005 | 0.147 ± 0.001 | 0.648 ± 0.008 | 0.743 ± 0.006 | 0.574 ± 0.001 | 0.579 ± 0.000 | 0.622 ± 0.010 | 0.626 / 0.146 |  |

FFNN-orig predicted-positive rate: 0.895 ± 0.020


#### D1q1 — `cb1_q_hop1_128_1.2_1.2.txt`, 71 drugs, 5041 ordered rows (incl. self), positive rate 0.692, 5 folds

| Method | F1 | MCC | Precision | AUPRC (paper way, hard labels) | AUROC (hard) | AUROC (scores) | AUPRC (scores, AP) | fold-1 = 80/20 split F1 / MCC | paper MCC / F1 / AUPRC / Prec |
|---|---|---|---|---|---|---|---|---|---|
| DDNet RF | 0.862 ± 0.007 | 0.552 ± 0.020 | 0.862 ± 0.008 | 0.910 ± 0.002 | 0.776 ± 0.005 | 0.870 ± 0.010 | 0.936 ± 0.008 | 0.868 / 0.569 |  |
| DDNet GB(HGB) | 0.875 ± 0.007 | 0.559 ± 0.020 | 0.840 ± 0.012 | 0.907 ± 0.006 | 0.761 ± 0.010 | 0.890 ± 0.009 | 0.950 ± 0.004 | 0.871 / 0.548 |  |
| DDNet FFNN-orig | 0.841 ± 0.008 | 0.335 ± 0.038 | 0.743 ± 0.014 | 0.867 ± 0.007 | 0.608 ± 0.016 | 0.824 ± 0.008 | 0.909 ± 0.011 | 0.838 / 0.355 |  |
| DDNet FFNN-fixed | 0.914 ± 0.007 | 0.725 ± 0.027 | 0.919 ± 0.016 | 0.946 ± 0.007 | 0.865 ± 0.016 | 0.946 ± 0.010 | 0.977 ± 0.005 | 0.925 / 0.770 |  |
| All-positive | 0.818 ± 0.007 | 0.000 ± 0.000 | 0.692 ± 0.010 | 0.846 ± 0.005 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.692 ± 0.010 | 0.810 / 0.000 |  |
| Degree (product) | 0.838 ± 0.015 | 0.540 ± 0.034 | 0.888 ± 0.014 | 0.912 ± 0.007 | 0.784 ± 0.017 | 0.862 ± 0.012 | 0.923 ± 0.005 | 0.840 / 0.531 |  |
| Degree (sum) | 0.831 ± 0.011 | 0.553 ± 0.029 | 0.909 ± 0.013 | 0.918 ± 0.005 | 0.795 ± 0.015 | 0.865 ± 0.012 | 0.926 ± 0.005 | 0.824 / 0.565 |  |
| Raw similarity S | 0.791 ± 0.010 | 0.218 ± 0.037 | 0.745 ± 0.020 | 0.849 ± 0.005 | 0.596 ± 0.022 | 0.618 ± 0.013 | 0.739 ± 0.010 | 0.806 / 0.231 |  |
| One-hot identity LR | 0.848 ± 0.003 | 0.440 ± 0.007 | 0.800 ± 0.014 | 0.885 ± 0.005 | 0.697 ± 0.008 | 0.856 ± 0.011 | 0.923 ± 0.005 | 0.844 / 0.431 |  |
| One-hot identity RF | 0.843 ± 0.007 | 0.561 ± 0.018 | 0.898 ± 0.011 | 0.918 ± 0.005 | 0.796 ± 0.009 | 0.867 ± 0.011 | 0.935 ± 0.007 | 0.844 / 0.567 |  |
| Path features only (HGB) | 0.842 ± 0.006 | 0.386 ± 0.026 | 0.775 ± 0.009 | 0.875 ± 0.005 | 0.660 ± 0.013 | 0.799 ± 0.013 | 0.894 ± 0.008 | 0.837 / 0.397 |  |
| Embeddings only (HGB) | 0.893 ± 0.008 | 0.632 ± 0.026 | 0.867 ± 0.008 | 0.921 ± 0.004 | 0.802 ± 0.009 | 0.910 ± 0.014 | 0.951 ± 0.007 | 0.893 / 0.640 |  |
| Embeddings only (RF) | 0.861 ± 0.007 | 0.549 ± 0.021 | 0.861 ± 0.012 | 0.909 ± 0.006 | 0.775 ± 0.011 | 0.872 ± 0.009 | 0.938 ± 0.007 | 0.861 / 0.545 |  |

FFNN-orig predicted-positive rate: 0.903 ± 0.012


#### D2q1 — `cb2_q_hop1_128_1.2_1.2.txt`, 73 drugs, 5329 ordered rows (incl. self), positive rate 0.664, 2 folds

| Method | F1 | MCC | Precision | AUPRC (paper way, hard labels) | AUROC (hard) | AUROC (scores) | AUPRC (scores, AP) | fold-1 = 80/20 split F1 / MCC | paper MCC / F1 / AUPRC / Prec |
|---|---|---|---|---|---|---|---|---|---|
| DDNet RF | 0.843 ± 0.014 | 0.565 ± 0.049 | 0.866 ± 0.017 | 0.902 ± 0.010 | 0.789 ± 0.025 | 0.876 ± 0.011 | 0.935 ± 0.006 | 0.833 / 0.530 |  |
| DDNet GB(HGB) | 0.863 ± 0.011 | 0.578 ± 0.027 | 0.836 ± 0.014 | 0.900 ± 0.008 | 0.779 ± 0.014 | 0.887 ± 0.003 | 0.941 ± 0.002 | 0.871 / 0.597 |  |
| DDNet FFNN-orig | 0.823 ± 0.007 | 0.358 ± 0.018 | 0.723 ± 0.009 | 0.854 ± 0.005 | 0.627 ± 0.007 | 0.809 ± 0.016 | 0.875 ± 0.004 | 0.828 / 0.370 |  |
| DDNet FFNN-fixed | 0.881 ± 0.015 | 0.638 ± 0.052 | 0.861 ± 0.034 | 0.914 ± 0.016 | 0.812 ± 0.033 | 0.922 ± 0.006 | 0.959 ± 0.001 | 0.892 / 0.675 |  |
| All-positive | 0.793 ± 0.005 | 0.000 ± 0.000 | 0.657 ± 0.007 | 0.828 ± 0.003 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.657 ± 0.007 | 0.796 / 0.000 |  |
| Degree (product) | 0.813 ± 0.020 | 0.532 ± 0.020 | 0.877 ± 0.013 | 0.897 ± 0.000 | 0.777 ± 0.007 | 0.853 ± 0.001 | 0.903 ± 0.001 | 0.799 / 0.518 |  |
| Degree (sum) | 0.818 ± 0.001 | 0.554 ± 0.028 | 0.892 ± 0.019 | 0.904 ± 0.007 | 0.790 ± 0.015 | 0.857 ± 0.000 | 0.906 ± 0.001 | 0.817 / 0.534 |  |
| Raw similarity S | 0.693 ± 0.016 | 0.245 ± 0.040 | 0.762 ± 0.010 | 0.819 ± 0.007 | 0.628 ± 0.020 | 0.628 ± 0.026 | 0.726 ± 0.016 | 0.682 / 0.217 |  |
| One-hot identity LR | 0.838 ± 0.004 | 0.474 ± 0.003 | 0.788 ± 0.006 | 0.876 ± 0.003 | 0.717 ± 0.001 | 0.849 ± 0.004 | 0.903 ± 0.004 | 0.841 / 0.476 |  |
| One-hot identity RF | 0.832 ± 0.003 | 0.573 ± 0.019 | 0.891 ± 0.015 | 0.908 ± 0.004 | 0.798 ± 0.011 | 0.862 ± 0.002 | 0.923 ± 0.005 | 0.833 / 0.560 |  |
| Path features only (HGB) | 0.831 ± 0.007 | 0.440 ± 0.024 | 0.771 ± 0.006 | 0.869 ± 0.001 | 0.695 ± 0.004 | 0.825 ± 0.018 | 0.901 ± 0.007 | 0.826 / 0.423 |  |
| Embeddings only (HGB) | 0.888 ± 0.004 | 0.658 ± 0.003 | 0.865 ± 0.001 | 0.917 ± 0.002 | 0.819 ± 0.001 | 0.904 ± 0.005 | 0.937 ± 0.002 | 0.891 / 0.660 |  |

FFNN-orig predicted-positive rate: 0.869 ± 0.001


### Leakage-free protocols (unordered pairs, no self pairs)

#### D1 (hop-1 embeddings) — 179 drugs, 15931 unordered pairs; per node-fold: train≈10182, S1≈5126, S2≈623 pairs; warm folds=5, node folds=5

| Method | warm AUROC | warm AUPRC | warm F1 | warm MCC | S1 AUROC | S1 AUPRC | S1 MCC | S2 AUROC | S2 AUPRC | S2 MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| DDNet-sym RF | 0.805 ± 0.009 | 0.850 ± 0.010 | 0.772 ± 0.008 | 0.442 ± 0.017 | 0.710 ± 0.024 | 0.762 ± 0.030 | 0.289 ± 0.035 | 0.600 ± 0.035 | 0.680 ± 0.074 | 0.100 ± 0.050 |
| DDNet-sym HGB | 0.836 ± 0.007 | 0.878 ± 0.007 | 0.801 ± 0.008 | 0.493 ± 0.022 | 0.712 ± 0.024 | 0.761 ± 0.034 | 0.300 ± 0.030 | 0.592 ± 0.039 | 0.671 ± 0.070 | 0.119 ± 0.077 |
| DDNet-sym FFNN-fixed | 0.903 ± 0.006 | 0.934 ± 0.005 | 0.832 ± 0.006 | 0.629 ± 0.013 | 0.695 ± 0.019 | 0.744 ± 0.036 | 0.284 ± 0.038 | 0.561 ± 0.028 | 0.629 ± 0.072 | 0.083 ± 0.043 |
| DDNet-concat both-orient. HGB | 0.893 ± 0.005 | 0.926 ± 0.004 | 0.840 ± 0.003 | 0.604 ± 0.010 | 0.747 ± 0.029 | 0.794 ± 0.033 | 0.355 ± 0.043 | 0.597 ± 0.044 | 0.683 ± 0.068 | 0.107 ± 0.046 |
| RandomVec-sym + path HGB (control) | 0.802 ± 0.008 | 0.854 ± 0.008 | 0.777 ± 0.006 | 0.426 ± 0.012 | 0.684 ± 0.023 | 0.740 ± 0.042 | 0.251 ± 0.024 | 0.549 ± 0.064 | 0.626 ± 0.085 | 0.067 ± 0.085 |
| RandomVec-sym + path FFNN-fixed (control) | 0.899 ± 0.008 | 0.930 ± 0.007 | 0.832 ± 0.008 | 0.628 ± 0.015 | 0.638 ± 0.024 | 0.696 ± 0.048 | 0.198 ± 0.028 | 0.503 ± 0.079 | 0.597 ± 0.112 | -0.020 ± 0.107 |
| Identity MLP (FFNN-fixed on multi-hot) | 0.976 ± 0.001 | 0.984 ± 0.002 | 0.934 ± 0.006 | 0.846 ± 0.015 | 0.671 ± 0.009 | 0.737 ± 0.030 | 0.263 ± 0.020 | 0.498 ± 0.002 | 0.582 ± 0.069 | 0.000 ± 0.000 |
| Emb-sym only HGB | 0.834 ± 0.007 | 0.877 ± 0.007 | 0.798 ± 0.007 | 0.485 ± 0.016 | 0.711 ± 0.026 | 0.762 ± 0.032 | 0.300 ± 0.033 | 0.599 ± 0.045 | 0.682 ± 0.069 | 0.119 ± 0.043 |
| Path only HGB | 0.628 ± 0.009 | 0.697 ± 0.007 | 0.694 ± 0.008 | 0.154 ± 0.015 | 0.602 ± 0.015 | 0.674 ± 0.037 | 0.113 ± 0.029 | 0.576 ± 0.041 | 0.665 ± 0.083 | 0.062 ± 0.051 |
| Morgan FP HGB | 0.886 ± 0.006 | 0.921 ± 0.005 | 0.836 ± 0.008 | 0.598 ± 0.018 | 0.732 ± 0.031 | 0.782 ± 0.048 | 0.325 ± 0.052 | 0.564 ± 0.051 | 0.644 ± 0.094 | 0.082 ± 0.066 |
| Identity LR | 0.817 ± 0.006 | 0.871 ± 0.005 | 0.776 ± 0.007 | 0.435 ± 0.014 | 0.715 ± 0.022 | 0.765 ± 0.037 | 0.286 ± 0.043 | 0.500 ± 0.000 | 0.582 ± 0.069 | 0.000 ± 0.000 |
| Degree (product) | 0.815 ± 0.005 | 0.868 ± 0.005 | 0.743 ± 0.015 | 0.457 ± 0.009 | 0.715 ± 0.022 | 0.765 ± 0.037 | 0.305 ± 0.031 | 0.500 ± 0.000 | 0.582 ± 0.069 | 0.000 ± 0.000 |
| Degree (sum) | 0.817 ± 0.006 | 0.870 ± 0.005 | 0.735 ± 0.014 | 0.453 ± 0.007 | 0.715 ± 0.022 | 0.765 ± 0.037 | 0.301 ± 0.024 | 0.500 ± 0.000 | 0.582 ± 0.069 | 0.000 ± 0.000 |
| Raw similarity S | 0.577 ± 0.008 | 0.640 ± 0.007 | 0.577 ± 0.050 | 0.117 ± 0.013 | 0.574 ± 0.038 | 0.639 ± 0.047 | 0.107 ± 0.051 | 0.588 ± 0.086 | 0.652 ± 0.109 | 0.102 ± 0.102 |
| SimKNN (k=10) | 0.759 ± 0.013 | 0.814 ± 0.011 | 0.732 ± 0.020 | 0.382 ± 0.022 | 0.695 ± 0.027 | 0.762 ± 0.025 | 0.273 ± 0.044 | 0.608 ± 0.037 | 0.693 ± 0.067 | 0.144 ± 0.078 |
| [leaky] DrugBank global degree product | 0.811 ± 0.004 | 0.861 ± 0.003 | 0.772 ± 0.003 | 0.455 ± 0.005 | 0.807 ± 0.014 | 0.856 ± 0.021 | 0.449 ± 0.023 | 0.817 ± 0.041 | 0.864 ± 0.049 | 0.459 ± 0.080 |

Positive rate: warm 0.582, S1 0.582, S2 0.582 (= AUPRC of a random ranker).

#### D2 (hop-1 embeddings) — 180 drugs, 16110 unordered pairs; per node-fold: train≈10296, S1≈5184, S2≈630 pairs; warm folds=5, node folds=5

| Method | warm AUROC | warm AUPRC | warm F1 | warm MCC | S1 AUROC | S1 AUPRC | S1 MCC | S2 AUROC | S2 AUPRC | S2 MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| DDNet-sym RF | 0.837 ± 0.007 | 0.874 ± 0.008 | 0.801 ± 0.008 | 0.505 ± 0.011 | 0.764 ± 0.024 | 0.809 ± 0.036 | 0.390 ± 0.031 | 0.669 ± 0.050 | 0.720 ± 0.081 | 0.261 ± 0.055 |
| DDNet-sym HGB | 0.860 ± 0.007 | 0.897 ± 0.009 | 0.823 ± 0.007 | 0.552 ± 0.012 | 0.764 ± 0.013 | 0.811 ± 0.024 | 0.385 ± 0.018 | 0.659 ± 0.039 | 0.719 ± 0.076 | 0.229 ± 0.039 |
| DDNet-sym FFNN-fixed | 0.905 ± 0.006 | 0.933 ± 0.007 | 0.842 ± 0.009 | 0.644 ± 0.013 | 0.744 ± 0.025 | 0.794 ± 0.031 | 0.367 ± 0.033 | 0.623 ± 0.035 | 0.688 ± 0.062 | 0.179 ± 0.073 |
| DDNet-concat both-orient. HGB | 0.906 ± 0.006 | 0.935 ± 0.006 | 0.857 ± 0.008 | 0.643 ± 0.015 | 0.784 ± 0.007 | 0.831 ± 0.021 | 0.419 ± 0.012 | 0.661 ± 0.043 | 0.724 ± 0.078 | 0.231 ± 0.062 |
| RandomVec-sym + path HGB (control) | 0.834 ± 0.008 | 0.877 ± 0.010 | 0.803 ± 0.005 | 0.494 ± 0.007 | 0.714 ± 0.009 | 0.762 ± 0.019 | 0.312 ± 0.016 | 0.588 ± 0.048 | 0.656 ± 0.063 | 0.108 ± 0.065 |
| RandomVec-sym + path FFNN-fixed (control) | 0.885 ± 0.007 | 0.921 ± 0.006 | 0.828 ± 0.007 | 0.598 ± 0.016 | 0.665 ± 0.011 | 0.716 ± 0.009 | 0.248 ± 0.019 | 0.540 ± 0.031 | 0.608 ± 0.046 | 0.044 ± 0.056 |
| Identity MLP (FFNN-fixed on multi-hot) | 0.972 ± 0.002 | 0.981 ± 0.002 | 0.925 ± 0.006 | 0.822 ± 0.013 | 0.659 ± 0.031 | 0.734 ± 0.030 | 0.250 ± 0.047 | 0.500 ± 0.000 | 0.583 ± 0.038 | 0.000 ± 0.000 |
| Emb-sym only HGB | 0.861 ± 0.006 | 0.896 ± 0.006 | 0.823 ± 0.009 | 0.553 ± 0.015 | 0.761 ± 0.017 | 0.808 ± 0.029 | 0.384 ± 0.023 | 0.651 ± 0.034 | 0.714 ± 0.075 | 0.216 ± 0.055 |
| Path only HGB | 0.656 ± 0.007 | 0.721 ± 0.010 | 0.716 ± 0.007 | 0.219 ± 0.012 | 0.632 ± 0.011 | 0.702 ± 0.016 | 0.173 ± 0.019 | 0.621 ± 0.050 | 0.693 ± 0.058 | 0.163 ± 0.073 |
| Morgan FP HGB | 0.901 ± 0.007 | 0.930 ± 0.008 | 0.851 ± 0.007 | 0.628 ± 0.013 | 0.789 ± 0.015 | 0.836 ± 0.019 | 0.414 ± 0.029 | 0.662 ± 0.054 | 0.732 ± 0.068 | 0.224 ± 0.084 |
| Identity LR | 0.844 ± 0.007 | 0.892 ± 0.009 | 0.801 ± 0.006 | 0.494 ± 0.010 | 0.730 ± 0.019 | 0.774 ± 0.019 | 0.334 ± 0.022 | 0.500 ± 0.000 | 0.583 ± 0.038 | 0.000 ± 0.000 |
| Degree (product) | 0.842 ± 0.009 | 0.888 ± 0.010 | 0.776 ± 0.009 | 0.510 ± 0.014 | 0.730 ± 0.019 | 0.773 ± 0.019 | 0.333 ± 0.026 | 0.500 ± 0.000 | 0.583 ± 0.038 | 0.000 ± 0.000 |
| Degree (sum) | 0.843 ± 0.007 | 0.891 ± 0.009 | 0.779 ± 0.008 | 0.517 ± 0.017 | 0.730 ± 0.019 | 0.773 ± 0.019 | 0.331 ± 0.039 | 0.500 ± 0.000 | 0.583 ± 0.038 | 0.000 ± 0.000 |
| Raw similarity S | 0.594 ± 0.008 | 0.652 ± 0.013 | 0.636 ± 0.033 | 0.156 ± 0.011 | 0.595 ± 0.024 | 0.655 ± 0.035 | 0.159 ± 0.023 | 0.601 ± 0.082 | 0.667 ± 0.104 | 0.159 ± 0.094 |
| SimKNN (k=10) | 0.786 ± 0.008 | 0.828 ± 0.012 | 0.768 ± 0.010 | 0.429 ± 0.014 | 0.736 ± 0.020 | 0.785 ± 0.035 | 0.342 ± 0.023 | 0.658 ± 0.055 | 0.729 ± 0.078 | 0.229 ± 0.099 |
| [leaky] DrugBank global degree product | 0.836 ± 0.008 | 0.882 ± 0.010 | 0.768 ± 0.017 | 0.491 ± 0.016 | 0.836 ± 0.014 | 0.882 ± 0.017 | 0.492 ± 0.026 | 0.830 ± 0.050 | 0.869 ± 0.056 | 0.486 ± 0.079 |

Positive rate: warm 0.585, S1 0.586, S2 0.583 (= AUPRC of a random ranker).

#### MP (hop-1 embeddings) — 289 drugs, 41616 unordered pairs; per node-fold: train≈26611, S1≈13363, S2≈1642 pairs; warm folds=3, node folds=5

| Method | warm AUROC | warm AUPRC | warm F1 | warm MCC | S1 AUROC | S1 AUPRC | S1 MCC | S2 AUROC | S2 AUPRC | S2 MCC |
|---|---|---|---|---|---|---|---|---|---|---|
| DDNet-sym HGB | 0.868 ± 0.005 | 0.847 ± 0.009 | 0.752 ± 0.005 | 0.557 ± 0.012 | 0.757 ± 0.016 | 0.705 ± 0.040 | 0.370 ± 0.031 | 0.616 ± 0.028 | 0.573 ± 0.082 | 0.182 ± 0.053 |
| DDNet-sym FFNN-fixed | 0.903 ± 0.003 | 0.895 ± 0.006 | 0.777 ± 0.003 | 0.623 ± 0.015 | 0.720 ± 0.016 | 0.670 ± 0.025 | 0.333 ± 0.033 | 0.563 ± 0.043 | 0.530 ± 0.065 | 0.122 ± 0.060 |
| DDNet-concat both-orient. HGB | 0.902 ± 0.005 | 0.888 ± 0.009 | 0.801 ± 0.007 | 0.631 ± 0.013 | 0.776 ± 0.008 | 0.726 ± 0.034 | 0.390 ± 0.012 | 0.631 ± 0.020 | 0.599 ± 0.080 | 0.202 ± 0.049 |
| RandomVec-sym + path HGB (control) | 0.838 ± 0.001 | 0.812 ± 0.006 | 0.720 ± 0.002 | 0.498 ± 0.004 | 0.710 ± 0.030 | 0.647 ± 0.058 | 0.281 ± 0.046 | 0.544 ± 0.065 | 0.513 ± 0.115 | 0.070 ± 0.100 |
| RandomVec-sym + path FFNN-fixed (control) | 0.883 ± 0.002 | 0.871 ± 0.006 | 0.762 ± 0.006 | 0.592 ± 0.005 | 0.652 ± 0.012 | 0.590 ± 0.029 | 0.205 ± 0.024 | 0.507 ± 0.025 | 0.478 ± 0.079 | 0.022 ± 0.061 |
| Identity MLP (FFNN-fixed on multi-hot) | 0.982 ± 0.001 | 0.981 ± 0.002 | 0.927 ± 0.004 | 0.869 ± 0.007 | 0.714 ± 0.016 | 0.642 ± 0.033 | 0.306 ± 0.024 | 0.500 ± 0.000 | 0.460 ± 0.066 | 0.000 ± 0.000 |
| Emb-sym only HGB | 0.867 ± 0.004 | 0.845 ± 0.010 | 0.753 ± 0.005 | 0.558 ± 0.011 | 0.756 ± 0.018 | 0.706 ± 0.038 | 0.367 ± 0.037 | 0.612 ± 0.030 | 0.572 ± 0.082 | 0.171 ± 0.054 |
| Path only HGB | 0.621 ± 0.000 | 0.575 ± 0.004 | 0.475 ± 0.003 | 0.187 ± 0.006 | 0.612 ± 0.016 | 0.562 ± 0.037 | 0.176 ± 0.024 | 0.602 ± 0.040 | 0.564 ± 0.097 | 0.180 ± 0.063 |
| Morgan FP HGB | 0.899 ± 0.004 | 0.885 ± 0.008 | 0.793 ± 0.002 | 0.618 ± 0.005 | 0.772 ± 0.012 | 0.722 ± 0.036 | 0.393 ± 0.018 | 0.621 ± 0.023 | 0.586 ± 0.073 | 0.178 ± 0.053 |
| Identity LR | 0.864 ± 0.004 | 0.848 ± 0.008 | 0.754 ± 0.007 | 0.542 ± 0.012 | 0.741 ± 0.014 | 0.669 ± 0.032 | 0.326 ± 0.018 | 0.500 ± 0.000 | 0.460 ± 0.066 | 0.000 ± 0.000 |
| Degree (product) | 0.864 ± 0.004 | 0.845 ± 0.008 | 0.754 ± 0.003 | 0.547 ± 0.007 | 0.741 ± 0.014 | 0.669 ± 0.032 | 0.335 ± 0.029 | 0.500 ± 0.000 | 0.460 ± 0.066 | 0.000 ± 0.000 |
| Degree (sum) | 0.860 ± 0.003 | 0.845 ± 0.008 | 0.745 ± 0.001 | 0.541 ± 0.006 | 0.741 ± 0.014 | 0.669 ± 0.032 | 0.319 ± 0.022 | 0.500 ± 0.000 | 0.460 ± 0.066 | 0.000 ± 0.000 |
| Raw similarity S | 0.614 ± 0.006 | 0.563 ± 0.004 | 0.505 ± 0.004 | 0.155 ± 0.008 | 0.610 ± 0.012 | 0.557 ± 0.033 | 0.140 ± 0.015 | 0.610 ± 0.034 | 0.565 ± 0.085 | 0.138 ± 0.061 |
| SimKNN (k=10) | 0.792 ± 0.006 | 0.769 ± 0.009 | 0.681 ± 0.013 | 0.429 ± 0.007 | 0.736 ± 0.024 | 0.693 ± 0.044 | 0.348 ± 0.032 | 0.649 ± 0.056 | 0.616 ± 0.097 | 0.227 ± 0.085 |
| [leaky] DrugBank global degree product | 0.843 ± 0.003 | 0.820 ± 0.008 | 0.747 ± 0.001 | 0.514 ± 0.008 | 0.844 ± 0.007 | 0.821 ± 0.022 | 0.515 ± 0.014 | 0.841 ± 0.016 | 0.817 ± 0.060 | 0.513 ± 0.044 |

Positive rate: warm 0.455, S1 0.453, S2 0.460 (= AUPRC of a random ranker).


