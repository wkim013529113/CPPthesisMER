### Core configurations

| Configuration | Seeds | Jamendo PR-AUC | Jamendo ROC-AUC | MERGE R² | MERGE valence R² | MERGE arousal R² | MERGE RMSE |
|---|---|---|---|---|---|---|---|
| Audio baseline (Jamendo + MERGE) | 3 | 0.1261 ± 0.0036 | 0.7339 ± 0.0035 | 0.4733 ± 0.0020 | 0.4044 ± 0.0021 | 0.5422 ± 0.0025 | 1.1252 ± 0.0020 |
| + GlobalMood multitask | 3 | 0.1267 ± 0.0013 | 0.7359 ± 0.0025 | 0.4768 ± 0.0047 | 0.4143 ± 0.0188 | 0.5393 ± 0.0095 | 1.1187 ± 0.0113 |
| + XLM-R lyrics (empty 4-dim slot) | 3 | 0.1276 ± 0.0002 | 0.7368 ± 0.0008 | 0.5516 ± 0.0066 | 0.5725 ± 0.0036 | 0.5307 ± 0.0096 | 0.9976 ± 0.0058 |
| + XLM-R + rhyme density | 3 | 0.1274 ± 0.0017 | 0.7375 ± 0.0017 | 0.5519 ± 0.0038 | 0.5705 ± 0.0098 | 0.5334 ± 0.0121 | 0.9984 ± 0.0057 |

Mean ± SD over seeds. Jamendo: mood-selected checkpoint; MERGE: VA-selected checkpoint; artist-disjoint MERGE test split.

### Controls

| Configuration | Seeds | Jamendo PR-AUC | Jamendo ROC-AUC | MERGE R² | MERGE valence R² | MERGE arousal R² | MERGE RMSE |
|---|---|---|---|---|---|---|---|
| Control: shuffled rhyme vectors | 3 | 0.1285 ± 0.0021 | 0.7373 ± 0.0004 | 0.5513 ± 0.0042 | 0.5704 ± 0.0081 | 0.5323 ± 0.0146 | 0.9989 ± 0.0035 |
| Control: random 4-dim noise | 3 | 0.1281 ± 0.0009 | 0.7386 ± 0.0020 | 0.5482 ± 0.0097 | 0.5723 ± 0.0036 | 0.5240 ± 0.0227 | 0.9998 ± 0.0042 |
| Control: non-rhyme lyric features | 3 | 0.1273 ± 0.0015 | 0.7382 ± 0.0016 | 0.5478 ± 0.0038 | 0.5617 ± 0.0086 | 0.5338 ± 0.0161 | 1.0056 ± 0.0022 |

Mean ± SD over seeds. Jamendo: mood-selected checkpoint; MERGE: VA-selected checkpoint; artist-disjoint MERGE test split.

### Exploratory ablations (single seed)

| Configuration | Seeds | Jamendo PR-AUC | Jamendo ROC-AUC | MERGE R² | MERGE valence R² | MERGE arousal R² | MERGE RMSE |
|---|---|---|---|---|---|---|---|
| Ablation: rhyme without coverage | 1 | 0.1292 | 0.7354 | 0.5505 | 0.5752 | 0.5257 | 0.9969 |
| Ablation: coverage only | 1 | 0.1296 | 0.7376 | 0.5429 | 0.5679 | 0.5178 | 1.0054 |
| Ablation: end-rhyme density only | 1 | 0.1296 | 0.7372 | 0.5455 | 0.5718 | 0.5191 | 1.0017 |
| Ablation: internal-rhyme density only | 1 | 0.1283 | 0.7355 | 0.5440 | 0.5638 | 0.5243 | 1.0068 |
| Ablation: rhymes per line only | 1 | 0.1312 | 0.7374 | 0.5524 | 0.5775 | 0.5274 | 0.9945 |

Mean ± SD over seeds. Jamendo: mood-selected checkpoint; MERGE: VA-selected checkpoint; artist-disjoint MERGE test split.

