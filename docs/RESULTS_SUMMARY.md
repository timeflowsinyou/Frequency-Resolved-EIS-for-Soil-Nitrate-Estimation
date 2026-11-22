# Reference Results Summary

## Primary grouped five-fold evaluation

Population: 19 soil specimens, six nitrate stages per specimen, 20 technical EIS measurements per stage, and 201 measured frequencies from 20 Hz to 1 MHz.

| Model | Coefficient of determination | RMSE (mg/L) | MAE (mg/L) |
|---|---:|---:|---:|
| Direct five-channel CNN | 0.851378 | 1.597343 | 1.282630 |
| Direct EIS–physical fusion CNN | 0.858814 | 1.556873 | 1.308818 |
| Admittance-based CNN | 0.873068 | 1.476194 | 1.148082 |
| Spectral level/shape CNN | 0.891099 | 1.367329 | 1.059820 |
| Fixed equal-weight ensemble | 0.898334 | 1.321130 | 1.037155 |

Separating density and water content from the frequency axis reduced RMSE by 0.0405 mg/L relative to direct five-channel convolution. The larger gains came from the physically structured admittance and spectral level/shape representations. The fixed representation ensemble reduced RMSE by 0.2762 mg/L relative to the direct five-channel model.

## Ten-network-seed grouped-CV stability

| Model | Mean coefficient of determination | Sample SD | Mean RMSE (mg/L) | RMSE sample SD (mg/L) |
|---|---:|---:|---:|---:|
| Admittance-based CNN | 0.835876 | 0.022014 | 1.675168 | 0.112815 |
| Spectral level/shape CNN | 0.857082 | 0.036903 | 1.555527 | 0.194141 |
| Fixed equal-weight ensemble | 0.870402 | 0.020803 | 1.487387 | 0.118275 |

The fixed ensemble improved RMSE relative to the admittance-based model in all ten network seeds and ranked first among the predefined models in eight seeds.

## Ten-network-seed historical transfer

| Model | Mean coefficient of determination | Sample SD | Mean RMSE (mg/L) | RMSE sample SD (mg/L) |
|---|---:|---:|---:|---:|
| Admittance-based CNN | 0.930349 | 0.017446 | 1.114611 | 0.144812 |
| Spectral level/shape CNN | 0.879891 | 0.061379 | 1.432209 | 0.370756 |
| Fixed equal-weight ensemble | 0.922474 | 0.030195 | 1.166633 | 0.218103 |

The transfer population contains three historical Z-series soils. These specimens were excluded from the frozen full-cohort model fit but had been examined in earlier development analyses.

## Interpretation boundary

Training-seed variability, between-soil heterogeneity, and within-stage technical-repeat variability are distinct. The ten-seed distribution is conditional on the current data, exclusions, folds, model settings, and hyperparameters. It is not a confidence interval for generalization to a new soil population.

