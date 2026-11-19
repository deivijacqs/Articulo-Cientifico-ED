# Manuscript workspace

This directory is reserved for the scientific manuscript and its publication-ready supporting material.

## Scope of the manuscript

The manuscript should describe the consolidated single-station London ozone experiment. It should not reuse the initial multi-city/multi-pollutant framing unless a new experiment is conducted to support that broader scope.

The study compares persistence, MLP baselines, and a neural model with a simplified dynamical structure for hourly ozone forecasts at 1-, 6-, and 24-hour horizons.

## Manuscript standards

- Write the manuscript in academic English.
- Describe the study design, data provenance, quality control, chronological partitions, model architecture, loss function, and evaluation procedures precisely.
- Distinguish validation diagnostics from final test evaluation.
- Report uncertainty and temporal dependence where relevant.
- Interpret learned dynamical quantities as effective model parameters, not direct emissions or independently identified atmospheric rates.
- Avoid causal claims that are not supported by the design.
- Verify every reference and DOI against the original scientific publication before submission.

## Tables and figures

Use only outputs whose provenance can be traced to a script and a documented input. Each table should be introduced in the manuscript text and include a self-contained caption. Each figure should have a caption, a specified placement in the manuscript, and a traceable source filename. Retain vector PDF versions where available and use PNG versions for review or systems that require raster images.

## Reproducibility record

Record the code revision, environment, data source and retrieval date, sensor identifier, study period, quality-control rules, partition boundaries, random seeds, model settings, and script versions used to generate each reported result.

Do not place API credentials, private conversation notes, or raw data in this directory or in the public-facing repository.

## Target journal and current status

- **Target journal:** *Environmental Modelling & Software* (Elsevier).
- **Working title:** “A physics-informed neural differential equation for hourly urban ozone forecasting: A reduced-order dynamical approach”.
- **Current stage:** Complete manuscript draft prepared for preliminary review by the supervising instructor; not yet submitted.
- **Authors:** Two authors from Universidad Técnica del Norte, Faculty of Engineering in Applied Sciences, Data Science and Artificial Intelligence Program, Ibarra, Ecuador.
- **Next milestone:** Incorporate the instructor's scientific and editorial feedback, obtain both authors' approval, and then prepare the final submission files according to the journal's current Guide for Authors.

The preliminary review copies are intended to be stored in `manuscript/review/` as a DOCX (editable) and PDF (read-only). They are being shared publicly only to facilitate the supervising instructor's review. They are not submitted, accepted, or final publication versions. After feedback is received, replace or remove these copies as appropriate and update the manuscript metadata, figures, tables, and related outputs. The final public research record should distinguish the approved/submitted manuscript from the code and selected outputs that support reproducibility.
