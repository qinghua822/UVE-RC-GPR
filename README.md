# UVE RC GPR model weights and testing instructions

This supplementary review package supplies the fitted parameters and minimal inference procedure for predicting oilseed rape leaf nitrogen concentration (LNC), leaf potassium concentration (LKC), and LNC/LKC. Full training source code and the original dataset are not included.

## Model corresponding to the manuscript

The model was fitted on 108 training samples. The same weights were used to predict the 46-sample test set and the 21-sample independent validation set. Those evaluation samples were excluded from feature statistics, scaling, and model fitting. The independent-validation membership was confirmed by the author.

The target-specific reflectance input counts are 30 for LNC, 30 for LKC, and 10 for LNC/LKC. `wavelengths_nm` stores the union of required wavelengths, and the three rows of `feature_mask` identify each target's own input subset. Each target uses only its own mask. This is a single fitted predictor, not a 100-model ensemble or a report of mean performance across 100 repetitions. The paired independent-GPR baseline coefficients are stored in the same archive.

## Files

- `uve_rc_gpr_30_30_10.npz`: fixed fitted parameters, target-specific masks, and metadata.
- `predict.py`: NumPy-only inference and self-test.
- `requirements.txt`: inference dependency.
- `example_spectra.csv`: two artificial spectra for checking the prediction procedure.
- `expected_predictions.csv`: reference outputs for those two spectra and both models.
- `README.md`: these testing instructions.

The artificial spectra have no measured target values. They verify implementation consistency only and must not be presented as independent validation or as evidence of predictive accuracy. The original 175-sample dataset, measured targets, training code, and figure files are intentionally omitted from this minimal review archive. Standardized support vectors inside the NPZ are required GP prediction parameters.

## Install and check

Extract the entire archive into one folder before running scripts. Do not launch a single file from a ZIP preview or a temporary extraction directory. Run the following commands from the extracted folder:

```bash
python -m pip install -r requirements.txt
python predict.py --self-test
python predict.py --input example_spectra.csv --output example_predictions.csv
python predict.py --input example_spectra.csv --output example_baseline_predictions.csv --baseline
```

The supplied decoder was checked using Python 3.9 with NumPy 2.0.2 and Python 3.12 with NumPy 2.3.5. A passing self-test verifies the fitting-sample count, target feature counts, and agreement of both models' predictions with the supplied references using relative and absolute tolerances of 1e-8.

## Test other spectra

Prepare a UTF-8 CSV with one sample per row, a unique nonempty `sample_id`, and the same numeric wavelength headers as `example_spectra.csv`. All required wavelength columns must be present; their order may differ. Supply finite, strictly positive reflectance values in percentage units: 20 means 20% reflectance. Multiply fractional reflectance by 100 before saving the CSV. Do not calculate spectral ratios or apply additional standardization.

```bash
python predict.py --input your_spectra.csv --output your_predictions.csv
```

Outputs are `LNC`, `LKC`, `LNC_LKC`, `derived_ratio`, and `consistency_error`. LNC and LKC are dry-mass percentages. The directly predicted nutrient ratio and remaining outputs are dimensionless. `derived_ratio = LNC / LKC`; `consistency_error = LNC_LKC - derived_ratio`. The consistency penalty is soft, so the discrepancy can be nonzero. Add `--baseline` to use the paired independent-GPR coefficients.

## Prediction definition

Load the archive with `numpy.load(..., allow_pickle=False)`. Standardize inputs using the saved means and scales. The support vectors are already standardized. For target j, compute squared distances only over the entries selected by `feature_mask[j]`:

```text
z = (x - feature_mean) / feature_scale
d2[j,i] = sum_k feature_mask[j,k] * (z[k] - support[i,k])**2
K[j,i] = amplitudes[0,j] * exp(-d2[j,i] / (2 * length_scales[0,j]**2))
q[j] = sum_i K[j,i] * coefficients[0,i,j]
t[j] = q[j] * target_scale[j] + target_mean[j]
y[j] = lower[j] + (upper[j]-lower[j]) / (1+exp(-t[j]))
```

Use amplitudes directly, without squaring, and do not add observation noise to cross-covariances. For baseline inference substitute `baseline_coefficients`. Target order is LNC, LKC, LNC/LKC. Only mean predictions are supplied; no uncertainty intervals are provided.

