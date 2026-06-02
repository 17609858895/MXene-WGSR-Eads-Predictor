# Paper Raw Data and Code

This folder contains the raw dataset and reproducible code used for the MXene WGS-domain adsorption-energy paper workflow.

## Folder structure

```text
paper_raw_data_and_code/
├── data/
│   └── raw/
│       └── Comprehensive Dataset on MXene Properties for Catalyst Design (2024).xlsx
├── code/
│   ├── figures/
│   │   ├── generate_all_figures.py
│   │   ├── generate_all_figures.ipynb
│   │   ├── generate_subfigures_fig2to8.ipynb
│   │   ├── build_subfigure_ppts.py
│   │   └── extract_panels_from_combined_pdf.py
│   ├── tables/
│   │   ├── generate_paper_tables.py
│   │   └── generate_paper_tables.ipynb
│   └── model_export/
│       └── train_export_model.py
├── docs/
│   ├── 新论文框架_修订版_吸附能主线.md
│   └── 图片说明.md
└── requirements-paper.txt
```

## Reproduce paper figures

Run from the repository root:

```bash
python paper_raw_data_and_code/code/figures/generate_all_figures.py
```

The generated figure files are written to:

```text
paper_raw_data_and_code/outputs/figures/
```

## Reproduce paper tables

Run from the repository root:

```bash
python paper_raw_data_and_code/code/tables/generate_paper_tables.py
```

The generated table files are written to:

```text
paper_raw_data_and_code/outputs/tables/
```

## Re-export the model bundle

Run from the repository root:

```bash
python paper_raw_data_and_code/code/model_export/train_export_model.py
```

The generated model bundle and example inputs are written to:

```text
paper_raw_data_and_code/outputs/model_export/
```

The deployed Streamlit app uses the model artifacts in the repository-level `model/` folder. The export script in this folder writes to `outputs/model_export/` to avoid overwriting the currently deployed model unless the user intentionally copies those files.

## Scientific scope

The reproducible workflow focuses on WGS-domain MXene adsorption-energy prediction. Dopant-domain analysis is included for transfer diagnostics and DFT-labelled trend interpretation, not for standalone dopant screening.
