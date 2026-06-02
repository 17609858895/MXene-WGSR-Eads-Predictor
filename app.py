from __future__ import annotations

from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_ROOT = Path(__file__).resolve().parent
MODEL_PATH = APP_ROOT / "model" / "model_bundle.joblib"
METADATA_PATH = APP_ROOT / "model" / "model_metadata.json"
EXAMPLE_PATH = APP_ROOT / "data" / "example_input.csv"

ELEMENTS = {
    "H": (1, 1, 1, 2.20, 31), "C": (6, 14, 2, 2.55, 76), "N": (7, 15, 2, 3.04, 71),
    "O": (8, 16, 2, 3.44, 66), "F": (9, 17, 2, 3.98, 57), "S": (16, 16, 3, 2.58, 105),
    "Cl": (17, 17, 3, 3.16, 102), "Br": (35, 17, 4, 2.96, 120),
    "Sc": (21, 3, 4, 1.36, 170), "Ti": (22, 4, 4, 1.54, 160), "V": (23, 5, 4, 1.63, 153),
    "Cr": (24, 6, 4, 1.66, 139), "Mn": (25, 7, 4, 1.55, 139), "Fe": (26, 8, 4, 1.83, 132),
    "Co": (27, 9, 4, 1.88, 126), "Ni": (28, 10, 4, 1.91, 124), "Cu": (29, 11, 4, 1.90, 132),
    "Zn": (30, 12, 4, 1.65, 122), "Y": (39, 3, 5, 1.22, 190), "Zr": (40, 4, 5, 1.33, 175),
    "Nb": (41, 5, 5, 1.60, 164), "Mo": (42, 6, 5, 2.16, 154), "Tc": (43, 7, 5, 1.90, 147),
    "Ru": (44, 8, 5, 2.20, 146), "Rh": (45, 9, 5, 2.28, 142), "Pd": (46, 10, 5, 2.20, 139),
    "Ag": (47, 11, 5, 1.93, 145), "Hf": (72, 4, 6, 1.30, 175), "Ta": (73, 5, 6, 1.50, 170),
    "W": (74, 6, 6, 2.36, 162), "Re": (75, 7, 6, 1.90, 151), "Os": (76, 8, 6, 2.20, 144),
    "Ir": (77, 9, 6, 2.20, 141), "Pt": (78, 10, 6, 2.28, 136), "Au": (79, 11, 6, 2.54, 136),
}
ALIASES = {"Rd": "Rh"}
MOL_PROPS = {
    "H": (1.008, 1, 1, 1), "H2": (2.016, 2, 2, 0), "H2O": (18.015, 3, 8, 0),
    "CO": (28.010, 2, 10, 0), "CO2": (44.010, 3, 16, 0), "OH": (17.007, 2, 7, 1),
    "O": (15.999, 1, 6, 2), "O2": (31.998, 2, 12, 2),
}

NUMERIC_BASE = [
    "a_ang", "N_Layers", "Layer_dist_MX", "Layer_dist_MM", "Length_MX", "Length_MT1",
    "E_Form", "Bader_M1", "Bader_M2", "Bader_X", "Bader_T1", "Band_gap_PBE_ev",
]
CATEGORICAL_BASE = ["Stacking", "M1", "M2", "X", "T1_type", "Mol"]
RAW_COLUMNS = ["Formula", *CATEGORICAL_BASE, *NUMERIC_BASE]


st.set_page_config(page_title="MXene Eads Predictor", page_icon="MX", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #f5f8f7; color: #1f2933; }
    .block-container { padding-top: 1.6rem; padding-bottom: 2.2rem; max-width: 1180px; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dbe7ea;
        border-radius: 10px;
        padding: 14px 16px;
        min-height: 104px;
        box-shadow: 0 8px 22px rgba(31, 41, 51, 0.045);
    }
    div[data-testid="stMetricLabel"] p { color: #526371; font-weight: 700; }
    div[data-testid="stMetricValue"] { color: #123c55; }
    .hero {
        background: linear-gradient(135deg, #123c55 0%, #0f766e 68%, #8bbf9f 100%);
        padding: 30px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 18px 42px rgba(18, 60, 85, 0.18);
        position: relative;
    }
    .hero h1 { font-size: 2.15rem; margin: 0 0 0.42rem 0; letter-spacing: 0; line-height: 1.14; }
    .hero p { font-size: 1.02rem; opacity: 0.95; max-width: 900px; margin: 0; line-height: 1.58; }
    .version-pill {
        position: absolute;
        top: 16px;
        right: 18px;
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.35);
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 0.78rem;
        font-weight: 750;
        color: #ffffff;
    }
    .soft-card {
        background: #ffffff;
        border: 1px solid #dbe7ea;
        border-radius: 12px;
        padding: 15px 17px;
        margin: 8px 0 16px 0;
        box-shadow: 0 8px 22px rgba(31, 41, 51, 0.04);
    }
    .warning-card {
        background: #fff8eb;
        border: 1px solid #f1d7a5;
        border-radius: 10px;
        padding: 13px 15px;
        color: #5b4214;
        margin: 16px 0 16px 0;
    }
    .section-title {
        color: #123c55;
        font-size: 0.98rem;
        font-weight: 800;
        margin: 0.1rem 0 0.55rem 0;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid #e3ecef;
    }
    .small-note { color: #607180; font-size: 0.92rem; line-height: 1.5; }
    div[data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #dbe7ea;
        border-radius: 14px;
        padding: 18px 18px 10px 18px;
        box-shadow: 0 10px 26px rgba(31, 41, 51, 0.055);
    }
    div[data-testid="stTextInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label {
        color: #334e5c;
        font-size: 0.86rem;
        font-weight: 750;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #ffffff;
        border: 1px solid #dbe7ea;
        border-radius: 9px 9px 0 0;
        padding: 8px 14px;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 9px;
        min-height: 42px;
        font-weight: 750;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundle() -> dict:
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata() -> dict:
    if METADATA_PATH.exists():
        return pd.read_json(METADATA_PATH, typ="series").to_dict()
    return {}


@st.cache_data
def load_example() -> pd.DataFrame:
    return pd.read_csv(EXAMPLE_PATH)


def elem_tuple(symbol) -> tuple[float, float, float, float, float]:
    if pd.isna(symbol) or str(symbol).strip() in {"", "None", "nan"}:
        return (np.nan,) * 5
    symbol = ALIASES.get(str(symbol).strip(), str(symbol).strip())
    if symbol in ELEMENTS:
        return ELEMENTS[symbol]
    if symbol in {"OH", "HO"}:
        return tuple(np.nanmean([ELEMENTS["O"], ELEMENTS["H"]], axis=0))
    if symbol == "NH":
        return tuple(np.nanmean([ELEMENTS["N"], ELEMENTS["H"]], axis=0))
    return (np.nan,) * 5


def mol_tuple(mol) -> tuple[float, float, float, float]:
    if pd.isna(mol):
        return (np.nan,) * 4
    return MOL_PROPS.get(str(mol).strip(), (np.nan,) * 4)


def make_features(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    raw = df.copy()
    for col in RAW_COLUMNS:
        if col not in raw.columns:
            raw[col] = np.nan
    x = pd.DataFrame(index=raw.index)
    missing_report: list[str] = []
    for col in NUMERIC_BASE:
        x[col] = pd.to_numeric(raw[col], errors="coerce")
        x[f"{col}_missing"] = x[col].isna().astype(int)
        if x[col].isna().any():
            missing_report.append(col)
    for col in CATEGORICAL_BASE:
        x[col] = raw[col].fillna("None").astype(str)
        if raw[col].isna().any():
            missing_report.append(col)
    for col in ["M1", "M2", "X", "T1_type", "Dopant"]:
        if col in raw.columns:
            props = np.array([elem_tuple(v) for v in raw[col]])
        else:
            props = np.full((len(raw), 5), np.nan)
        for i, prop_name in enumerate(["Z", "group", "period", "en", "radius"]):
            x[f"{col}_{prop_name}"] = props[:, i]
            x[f"{col}_{prop_name}_missing"] = np.isnan(props[:, i]).astype(int)
    mol_props = np.array([mol_tuple(v) for v in raw["Mol"]])
    for i, prop_name in enumerate(["mw", "atoms", "valence", "radical_e"]):
        x[f"Mol_{prop_name}"] = mol_props[:, i]
        x[f"Mol_{prop_name}_missing"] = np.isnan(mol_props[:, i]).astype(int)
    x["has_termination"] = raw["T1_type"].notna().astype(int)
    x["is_doped_domain"] = raw.get("Dopant", pd.Series(index=raw.index)).notna().astype(int)
    for col in feature_columns:
        if col not in x.columns:
            x[col] = np.nan
    return x[feature_columns], sorted(set(missing_report))


def predict(df: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, list[str]]:
    x, missing = make_features(df, bundle["feature_columns"])
    pred_matrix = []
    for name in bundle["weights"]:
        pred_matrix.append(bundle["models"][name].predict(x))
    weights = np.array([bundle["weights"][name] for name in bundle["weights"]], dtype=float)
    ensemble = np.column_stack(pred_matrix) @ weights
    out = df.copy()
    out["Predicted_E_ads_eV"] = ensemble
    best = bundle.get("best_single_model")
    if best in bundle["models"]:
        out[f"{best}_E_ads_eV"] = bundle["models"][best].predict(x)
    out["Model"] = "TOPSIS ensemble"
    return out, missing


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="prediction")
    output.seek(0)
    return output.getvalue()


bundle = load_bundle()
metadata = load_metadata()
example = load_example()

st.markdown(
    """
    <div class="hero">
      <div class="version-pill">Aligned UI v2</div>
      <h1>MXene WGSR E<sub>ads</sub> Predictor</h1>
      <p>Interpretable WGS-domain adsorption-energy prediction for MXene catalyst descriptors. The app uses the same preprocessing and TOPSIS ensemble model as the paper workflow.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
metric_cols[0].metric("Target", "Eads", "eV")
metric_cols[1].metric("Primary model", "TOPSIS", "ensemble")
metric_cols[2].metric("Hold-out R2", f"{metadata.get('TOPSIS_ensemble', {}).get('R2', np.nan):.3f}")
metric_cols[3].metric("Hold-out RMSE", f"{metadata.get('TOPSIS_ensemble', {}).get('RMSE', np.nan):.3f} eV")

st.markdown(
    """
    <div class="warning-card">
    <b>Applicability domain:</b> prediction is intended for WGS-domain MXene systems.
    Missing descriptors are imputed by the training pipeline. Dopant-domain transfer was weak in the study, so this tool should not be used as a reliable dopant-screening engine.
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["Single prediction", "Batch prediction", "Model information", "Input template"])

with tabs[0]:
    st.markdown(
        '<div class="soft-card"><b>Single-sample input</b><br><span class="small-note">Enter one MXene adsorption system. Numeric descriptors use the paper descriptor names and units; missing values in batch files are imputed by the training pipeline.</span></div>',
        unsafe_allow_html=True,
    )
    defaults = example.iloc[0].to_dict()
    with st.form("single_prediction_form", clear_on_submit=False):
        comp_col, struct_col, elec_col = st.columns(3, gap="large")
        with comp_col:
            st.markdown('<div class="section-title">Composition and adsorbate</div>', unsafe_allow_html=True)
            formula = st.text_input("Formula", value=str(defaults.get("Formula", "")))
            stacking = st.selectbox("Stacking", sorted(example["Stacking"].dropna().astype(str).unique()), index=0)
            m1 = st.selectbox("M1", sorted(example["M1"].dropna().astype(str).unique()), index=0)
            m2_options = ["None", *sorted(example["M2"].dropna().astype(str).unique())]
            m2 = st.selectbox("M2", m2_options, index=0)
            x_atom = st.selectbox("X", sorted(example["X"].dropna().astype(str).unique()), index=0)
            t1 = st.selectbox("T1_type", ["None", *sorted(example["T1_type"].dropna().astype(str).unique())], index=0)
            mol = st.selectbox("Mol", sorted(example["Mol"].dropna().astype(str).unique()), index=0)
        with struct_col:
            st.markdown('<div class="section-title">Structure descriptors</div>', unsafe_allow_html=True)
            values = {}
            for col in ["a_ang", "N_Layers", "Layer_dist_MX", "Layer_dist_MM", "Length_MX", "Length_MT1"]:
                default = defaults.get(col, np.nan)
                values[col] = st.number_input(col, value=float(default) if pd.notna(default) else 0.0, format="%.5f")
        with elec_col:
            st.markdown('<div class="section-title">Electronic descriptors</div>', unsafe_allow_html=True)
            for col in ["E_Form", "Bader_M1", "Bader_M2", "Bader_X", "Bader_T1", "Band_gap_PBE_ev"]:
                default = defaults.get(col, np.nan)
                values[col] = st.number_input(col, value=float(default) if pd.notna(default) else 0.0, format="%.5f")
        submitted = st.form_submit_button("Predict Eads", type="primary", use_container_width=True)

    if submitted:
        row = {"Formula": formula, "Stacking": stacking, "M1": m1, "M2": np.nan if m2 == "None" else m2, "X": x_atom, "T1_type": np.nan if t1 == "None" else t1, "Mol": mol, **values}
        result, missing = predict(pd.DataFrame([row]), bundle)
        value = float(result["Predicted_E_ads_eV"].iloc[0])
        res_col, note_col = st.columns([0.35, 0.65], gap="large")
        with res_col:
            st.metric("Predicted Eads", f"{value:.4f} eV", "TOPSIS ensemble")
        with note_col:
            st.markdown('<div class="soft-card"><b>Prediction note</b><br><span class="small-note">Use this value for WGS-domain MXene adsorption-energy estimation. For dopant-domain screening, rely on DFT-labelled trends rather than direct model transfer.</span></div>', unsafe_allow_html=True)
        if missing:
            st.info("Missing descriptors were imputed: " + ", ".join(missing))
        st.dataframe(result, use_container_width=True)
        st.download_button("Download single prediction", to_excel_bytes(result), "single_prediction.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tabs[1]:
    st.markdown("Upload a CSV or XLSX file with the descriptor columns shown in the template. Extra columns are preserved.")
    uploaded = st.file_uploader("Upload descriptor file", type=["csv", "xlsx"])
    st.download_button("Download example CSV", example.to_csv(index=False).encode("utf-8-sig"), "example_input.csv", "text/csv")
    if uploaded is not None:
        data = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
        st.dataframe(data.head(20), use_container_width=True)
        required_missing = [c for c in RAW_COLUMNS if c not in data.columns]
        if required_missing:
            st.warning("Columns not found and will be imputed or set to None: " + ", ".join(required_missing))
        if st.button("Run batch prediction", type="primary", use_container_width=True):
            result, missing = predict(data, bundle)
            st.success(f"Completed {len(result)} predictions.")
            if missing:
                st.info("Missing descriptors were imputed: " + ", ".join(missing))
            st.dataframe(result.head(100), use_container_width=True)
            st.download_button("Download prediction Excel", to_excel_bytes(result), "mxene_eads_predictions.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tabs[2]:
    st.markdown("#### Model card")
    st.write(f"Training domain: {bundle.get('training_domain')}")
    st.write(f"Best single model: {bundle.get('best_single_model')}")
    st.write("Ensemble weights")
    st.dataframe(pd.DataFrame({"model": list(bundle["weights"].keys()), "weight": list(bundle["weights"].values())}), use_container_width=True)
    st.write("Hold-out metrics")
    rows = []
    for key in ["TOPSIS_ensemble", bundle.get("best_single_model")]:
        if key in metadata:
            rows.append({"model": key, **metadata[key]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tabs[3]:
    st.markdown("#### Required descriptor columns")
    st.code(", ".join(RAW_COLUMNS))
    st.dataframe(example, use_container_width=True)
