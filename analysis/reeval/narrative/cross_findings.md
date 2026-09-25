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
