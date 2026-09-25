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
