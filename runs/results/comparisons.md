### Paired comparisons on MERGE test (A − B)

| A − B | Question | Metric | Seeds | Δ | 95% CI | p |
|---|---|---|---|---|---|---|
| C4 − C3 | Rhyme beyond XLM-R semantics (primary) | valence R² | 3 | -0.0020 | [-0.0149, +0.0103] | 0.7616 |
| C4 − C3 | Rhyme beyond XLM-R semantics (primary) | arousal R² | 3 | +0.0027 | [-0.0062, +0.0110] | 0.5360 |
| C4 − C3 | Rhyme beyond XLM-R semantics (primary) | combined R² | 3 | +0.0004 | [-0.0073, +0.0076] | 0.9248 |
| C4 − K1 | Song-specific rhyme vs. rhyme distribution | valence R² | 3 | +0.0001 | [-0.0120, +0.0119] | 0.9722 |
| C4 − K1 | Song-specific rhyme vs. rhyme distribution | arousal R² | 3 | +0.0011 | [-0.0065, +0.0087] | 0.7672 |
| C4 − K1 | Song-specific rhyme vs. rhyme distribution | combined R² | 3 | +0.0006 | [-0.0067, +0.0077] | 0.8590 |
| C4 − K2 | Rhyme vs. any extra 4-dim input | valence R² | 3 | -0.0019 | [-0.0132, +0.0091] | 0.7466 |
| C4 − K2 | Rhyme vs. any extra 4-dim input | arousal R² | 3 | +0.0094 | [-0.0039, +0.0226] | 0.1616 |
| C4 − K2 | Rhyme vs. any extra 4-dim input | combined R² | 3 | +0.0038 | [-0.0050, +0.0128] | 0.3976 |
| C4 − K3 | Rhyme vs. non-rhyme lyric structure | valence R² | 3 | +0.0088 | [-0.0035, +0.0204] | 0.1528 |
| C4 − K3 | Rhyme vs. non-rhyme lyric structure | arousal R² | 3 | -0.0005 | [-0.0084, +0.0077] | 0.9048 |
| C4 − K3 | Rhyme vs. non-rhyme lyric structure | combined R² | 3 | +0.0042 | [-0.0031, +0.0114] | 0.2594 |
| C3 − C2 | XLM-R lyrics beyond audio multitask | valence R² | 3 | +0.1582 | [+0.1012, +0.2169] | 0.0000 |
| C3 − C2 | XLM-R lyrics beyond audio multitask | arousal R² | 3 | -0.0087 | [-0.0360, +0.0194] | 0.5326 |
| C3 − C2 | XLM-R lyrics beyond audio multitask | combined R² | 3 | +0.0748 | [+0.0418, +0.1083] | 0.0000 |
| C2 − C1 | GlobalMood multitask beyond audio baseline | valence R² | 3 | +0.0099 | [+0.0008, +0.0190] | 0.0318 |
| C2 − C1 | GlobalMood multitask beyond audio baseline | arousal R² | 3 | -0.0029 | [-0.0106, +0.0053] | 0.4842 |
| C2 − C1 | GlobalMood multitask beyond audio baseline | combined R² | 3 | +0.0035 | [-0.0026, +0.0098] | 0.2464 |

Δ = mean over paired seeds of R²(A) − R²(B); CI and p from 10000 paired bootstrap resamples of test songs.

### Pre-registered criteria (valence R²)

- C4 > C3: CI includes 0 or below
- C4 > K1: CI includes 0 or below
- C4 > K2: CI includes 0 or below
- C4 > K3: CI includes 0 or below
- Rhyme effect: **not detected** under these controls.
