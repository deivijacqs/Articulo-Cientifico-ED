# Physics-informed neural differential equations for urban air-pollution dynamics

## Target journal and manuscript status

- **Target journal:** *Environmental Modelling & Software* (Elsevier).
- **Current manuscript title:** “A physics-informed neural differential equation for hourly urban ozone forecasting: A reduced-order dynamical approach”.
- **Manuscript language:** English.
- **Manuscript status:** Full draft prepared and formatted for preliminary academic review by the supervising instructor. The manuscript has not yet been submitted to the journal.
- **Authorship:** Two authors affiliated with Universidad Técnica del Norte, Faculty of Engineering in Applied Sciences, Data Science and Artificial Intelligence Program, Ibarra, Ecuador. Author names and correspondence details are maintained in the manuscript file and are not duplicated in this repository metadata.
- **Editorial workflow:** Complete the instructor's scientific and editorial review before preparing the journal submission package. Journal formatting and submission requirements must be checked against the current official Guide for Authors.

The editable Word manuscript is maintained separately during drafting and review. It is intentionally not included in this repository at this stage. A PDF may be generated as a read-only review copy when needed; neither format should be treated as the definitive submitted version until the authors approve it.

## Project overview

This repository contains the code and selected outputs for a single-station case study of hourly ambient ozone (O3) in London, United Kingdom. The study combines OpenAQ observations with historical meteorological variables from the Open-Meteo Historical Weather API and compares persistence, conventional multilayer perceptrons (MLPs), and a neural model constrained by a simplified dynamical equation.

The experiment focuses on three forecast horizons (1, 6, and 24 hours) and on the predictive behaviour and effective dynamical parameters of the proposed model. It is not a multi-city study and does not estimate direct emission rates.

## Study scope and data

- Pollutant: ground-level ozone (O3).
- Monitoring site: Bexley–Belvedere West, London, United Kingdom.
- OpenAQ sensor: `1304560`.
- Hourly observation period: 28 April 2021 to 31 December 2023.
- Meteorological inputs: 2-m air temperature, relative humidity, dew-point temperature, 10-m wind speed and direction, shortwave radiation, cloud cover, and precipitation.
- Forecast horizons: 1, 6, and 24 hours.
- Data partitions: chronological training, validation, and test sets.

The weather data were retrieved from the Open-Meteo Historical Weather API for the monitoring-site coordinates. Project metadata identifies the meteorological product as ERA5. The downloaded JSON filename includes `era5_land`, so the precise product provenance should be checked against the original API request and metadata before making a more specific claim about the product.

## Models

The evaluated approaches include:

1. Persistence.
2. MLP without meteorological predictors.
3. MLP with meteorological predictors.
4. Neural dynamical model with a simplified physical structure.

The physical model uses

\[
\frac{dC}{dt}=q_\theta(x_t)-kC(t), \qquad k>0,
\]

where \(C(t)\) is the ozone concentration, \(q_\theta\) is a non-negative learned effective source term, and \(k\) is a positive effective removal coefficient. The source term is assumed constant over each forecast horizon in the analytical forecast solution.

These quantities are effective model parameters. They should not be interpreted as direct measurements of emissions, chemically isolated production, or an independently identified atmospheric ozone lifetime. A single concentration time series does not, by itself, identify the individual physical processes contributing to the observed concentration.

## Environment and installation

The project was developed on Windows with Python 3.11 in a Conda environment. From Anaconda Prompt or a compatible terminal:

```bash
conda env create -f environment.yml
conda activate urban-air-pinn
```

The reproducible environment is defined by `environment.yml` and `requirements-pip.txt`. The latter specifies the CPU PyTorch package used by the training scripts. `environment-lock-win-64.txt` records a platform-specific Windows export and is provided as a reference, not as a cross-platform lockfile.

If the OpenAQ download script is used, create a local `.env` file based on `.env.example` and set `OPENAQ_API_KEY`. Never commit `.env` or share its contents. The meteorological download script uses the Open-Meteo archive endpoint and does not require a CDS account.

## Reproduction workflow

Run commands from the repository root with the Conda environment activated. The raw data and generated intermediate datasets are intentionally not included in this repository; the corresponding acquisition and processing steps must be run locally.

### 1. Environment check

```bash
python scripts/00_check_environment.py
```

### 2. Ozone observations

```bash
python scripts/08_download_historical_o3.py
python scripts/09_audit_historical_o3.py
python scripts/10_inspect_o3_flags.py
python scripts/11_validate_o3_quality.py
python scripts/12_inspect_o3_peaks.py
python scripts/13_build_o3_model_dataset.py
python scripts/14_validate_o3_model_dataset.py
python scripts/15_split_o3_temporal.py
python scripts/16_evaluate_o3_baselines.py
python scripts/17_compare_o3_baselines_paired.py
```

The OpenAQ download requires a valid API key. Review the quality-control outputs before proceeding.

### 3. Meteorological data and merged dataset

```bash
python scripts/18_download_historical_weather.py
python scripts/19_merge_o3_weather.py
```

### 4. Baselines and neural models

```bash
python scripts/20_train_o3_neural_network.py
python scripts/21_ablation_o3_meteorology.py
python scripts/22_evaluate_multihorizon_baselines.py
python scripts/23_train_o3_pinn.py
python scripts/24_verificar_predicciones.py
```

The training and evaluation scripts expect the intermediate datasets produced by the previous steps. Preserve the chronological partitions and do not use the test partition for model selection.

### 5. Diagnostics, robustness analyses, and figures

The scripts numbered 31 and 33–45 generate the selected temporal-dependence, uncertainty, stability, calibration, paired-comparison, parameter, and figure outputs. They use the detailed prediction file produced by the model-evaluation workflow, except where a downstream script consumes a table generated by an earlier diagnostic script.

Run the relevant scripts in dependency order:

```bash
python scripts/31_dependencia_temporal_agregada.py
python scripts/33_bootstrap_jerarquico_bloques.py
python scripts/34_block_subsampling_temporal.py
python scripts/35_diagnostico_estabilidad_subsampling.py
python scripts/36_diagnostico_episodios_influyentes.py
python scripts/37_estratificacion_concentracion.py
python scripts/38_estratificacion_cinco_cuantiles.py
python scripts/39_diagnostico_calibracion_errores.py
python scripts/40_calibracion_global_condicional.py
python scripts/41_comparacion_pareada_modelos.py
python scripts/42_figura_bootstrap_jerarquico.py
python scripts/43_figuras_estabilidad_temporal.py
python scripts/44_figura_autocorrelacion_perdidas.py
python scripts/45_figura_parametros_fisicos.py
```

Some diagnostic scripts are independent and can be run separately once their input prediction file exists. The order above ensures that the principal downstream figure scripts have their required tables.

## Repository contents

- `scripts/`: acquisition, processing, modelling, diagnostics, and figure-generation scripts.
- `docs/`: data dictionary and scientific design documentation.
- Configuration files from the initial exploratory workflow are not included because they do not describe the consolidated London ozone experiment.
- `data/`: local raw and processed data directories. Data files are excluded from version control.
- `outputs/tables/`: selected summary tables; large intermediate datasets and detailed predictions are excluded.
- `outputs/figures/`: selected figures in PNG and PDF formats.
- `models/`: local model artifacts; excluded from version control.
- `logs/`: local execution logs; excluded from version control.
- `manuscript/`: manuscript organization notes.

## Reproducibility and interpretation

The scripts use fixed random seeds where specified and preserve chronological partitions. Results should be interpreted in light of the single-site design, temporal dependence, missing observations, and the assumptions of the simplified dynamical equation. The learned source and removal terms are effective quantities and do not establish causal atmospheric mechanisms.

Do not commit API credentials, private notes, raw monitoring data, large intermediate datasets, trained model binaries, or local environment files. Consult `.gitignore` before staging files.
