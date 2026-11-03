### Diagnostic: official MERGE split (artist overlap)

| Configuration | Seeds | Jamendo PR-AUC | Jamendo ROC-AUC | MERGE R² | MERGE valence R² | MERGE arousal R² | MERGE RMSE |
|---|---|---|---|---|---|---|---|
| Official split: + XLM-R lyrics (empty slot) | 3 | 0.1308 ± 0.0045 | 0.7403 ± 0.0047 | 0.5208 ± 0.0040 | 0.5475 ± 0.0049 | 0.4941 ± 0.0041 | 0.9698 ± 0.0046 |
| Official split: + XLM-R + rhyme density | 3 | 0.1325 ± 0.0012 | 0.7412 ± 0.0040 | 0.5173 ± 0.0036 | 0.5395 ± 0.0052 | 0.4950 ± 0.0101 | 0.9760 ± 0.0029 |

Mean ± SD over seeds. Jamendo: mood-selected checkpoint; MERGE: VA-selected checkpoint; official tvt_70_15_15 MERGE test split.

