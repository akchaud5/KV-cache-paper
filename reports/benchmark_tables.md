## Evaluation Summary

| Config             |   Compression× |   Throughput× |   Attention Coverage |   Key Cosine |   Value Cosine |   Selected Cosine |
|:-------------------|---------------:|--------------:|---------------------:|-------------:|---------------:|------------------:|
| baseline_full.yaml |        1.5137  |       1       |             0.979124 |     0.930889 |       0.930906 |          1        |
| hybrid_6x.yaml     |        4.38109 |       1.82609 |             0.89303  |     0.799024 |       0.792869 |          0.997816 |
| hybrid_8x.yaml     |        3.41415 |       2.08696 |             0.905143 |     0.881146 |       0.840766 |          0.98921  |
| hybrid_12x.yaml    |        3.47339 |       3.11111 |             0.959196 |     0.921266 |       0.861319 |          0.984258 |

## Hyperparameter Sweep (Hybrid 8×)

|   Guard |   Bits |   Compression× |   Attention Coverage |   Key Cosine |
|--------:|-------:|---------------:|---------------------:|-------------:|
|     0.4 |      2 |        4.6772  |             0.905143 |     0.87635  |
|     0.4 |      3 |        4.5458  |             0.905143 |     0.876349 |
|     0.4 |      4 |        4.30396 |             0.905143 |     0.88774  |
|     0.5 |      2 |        3.94672 |             0.905143 |     0.878739 |
|     0.5 |      3 |        3.85274 |             0.905143 |     0.87874  |
|     0.5 |      4 |        3.6776  |             0.905143 |     0.88828  |
|     0.6 |      2 |        3.41415 |             0.905143 |     0.881146 |
|     0.6 |      3 |        3.3436  |             0.905143 |     0.881146 |
|     0.6 |      4 |        3.21089 |             0.905143 |     0.888821 |
|     0.7 |      2 |        3.00784 |             0.905143 |     0.883576 |
|     0.7 |      3 |        2.95294 |             0.905143 |     0.883577 |
|     0.7 |      4 |        2.84895 |             0.905143 |     0.889363 |

## Benchmark Accuracy (Synthetic)

| Config     | Dataset   | Task          |   Baseline |    Score |      Delta |
|:-----------|:----------|:--------------|-----------:|---------:|-----------:|
| baseline   | LongBench | multifield_qa |       0.62 | 0.62     |  0         |
| baseline   | LongBench | narrative_qa  |       0.57 | 0.57     |  0         |
| baseline   | LongBench | gov_report    |       0.44 | 0.44     |  0         |
| baseline   | LongBench | math_facts    |       0.69 | 0.69     |  0         |
| hybrid_12x | LongBench | multifield_qa |       0.62 | 0.591153 | -0.0288468 |
| hybrid_12x | LongBench | narrative_qa  |       0.57 | 0.54348  | -0.0265204 |
| hybrid_12x | LongBench | gov_report    |       0.44 | 0.419528 | -0.0204719 |
| hybrid_12x | LongBench | math_facts    |       0.69 | 0.657896 | -0.0321037 |
| hybrid_6x  | LongBench | multifield_qa |       0.62 | 0.548853 | -0.0711471 |
| hybrid_6x  | LongBench | narrative_qa  |       0.57 | 0.504591 | -0.0654094 |
| hybrid_6x  | LongBench | gov_report    |       0.44 | 0.389509 | -0.0504915 |
| hybrid_6x  | LongBench | math_facts    |       0.69 | 0.61082  | -0.0791798 |
| hybrid_8x  | LongBench | multifield_qa |       0.62 | 0.565287 | -0.0547125 |
| hybrid_8x  | LongBench | narrative_qa  |       0.57 | 0.5197   | -0.0503002 |
| hybrid_8x  | LongBench | gov_report    |       0.44 | 0.401172 | -0.0388283 |
| hybrid_8x  | LongBench | math_facts    |       0.69 | 0.62911  | -0.0608898 |

