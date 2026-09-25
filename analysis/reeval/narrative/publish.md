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
