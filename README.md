# MXene WGSR Eads Predictor

Streamlit web UI for WGS-domain MXene adsorption-energy prediction.

## What the app does

- Accepts MXene descriptors such as `M1`, `M2`, `X`, `T1_type`, `Mol`, formation energy, Bader charges and structure descriptors.
- Applies the same feature engineering and imputation logic used in the paper workflow.
- Predicts adsorption energy `Eads` in eV using the TOPSIS ensemble model.
- Supports single-sample prediction and CSV/XLSX batch prediction.
- Reports missing descriptors that were imputed.

## Scientific scope

The model is intended for WGS-domain MXene adsorption systems. It should not be used as a reliable dopant-domain screening engine because the paper diagnostics showed weak direct transfer to the dopant domain.

## Files

```text
MXene-WGSR-Eads-Predictor/
├── app.py
├── model/
│   ├── model_bundle.joblib
│   └── model_metadata.json
├── data/
│   ├── example_input.csv
│   └── single_prediction_template.csv
├── scripts/
│   └── train_export_model.py
├── .streamlit/
│   └── config.toml
├── requirements.txt
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Upload this folder to a GitHub repository.
2. Go to Streamlit Community Cloud.
3. Create a new app from the repository.
4. Set the main file path to `app.py`.
5. Deploy.

## Input columns

The batch prediction file should include:

```text
Formula, Stacking, M1, M2, X, T1_type, Mol, a_ang, N_Layers, Layer_dist_MX, Layer_dist_MM, Length_MX, Length_MT1, E_Form, Bader_M1, Bader_M2, Bader_X, Bader_T1, Band_gap_PBE_ev
```

Missing numeric descriptors are imputed by the trained preprocessing pipeline. Missing categorical descriptors are filled as `None`.
