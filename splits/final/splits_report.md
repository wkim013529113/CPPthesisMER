# Final splits — rerun-v2

## MERGE: artist-disjoint, stratified by quadrant

  2000 songs, 1044 artists, 1044 groups; largest group 29 songs (slayer)

  split   n       Q1    Q2    Q3    Q4   val_mean aro_mean
  train  1400    350   350   350   350   0.5204   0.4737
  val     300     75    75    75    75   0.5177   0.4718
  test    300     75    75    75    75   0.5104   0.4803
  train∩val: artists 0  artist+title 0
  train∩test: artists 0  artist+title 0
  val∩test: artists 0  artist+title 0
  same-audio MT0009795009 -> ['train'], A066-140 -> ['train']
  same-audio MT0011933875 -> ['train'], A114-120 -> ['train']
  wrote dataset/merge/MERGE_Bimodal_Balanced/tvt_dataframes/tvt_artist_70_15_15/

  rhyme stats (new train split, .npy order):
    end_rhyme_density        mean 0.1857  std 0.1605
    internal_rhyme_density   mean 0.0391  std 0.0467
    rhymes_per_line          mean 0.4702  std 0.4492
    cmudict_coverage         mean 0.9824  std 0.0297

## GlobalMood: loader replica (rng 42, 80/10/10)

  train: 900
  val: 112
  test: 113

## Jamendo: MTG split-0 pickles, test duplicates removed

  train: 2580
  val: 1067
  test: 1163  (removed 2 duplicates)
