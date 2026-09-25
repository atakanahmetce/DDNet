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
