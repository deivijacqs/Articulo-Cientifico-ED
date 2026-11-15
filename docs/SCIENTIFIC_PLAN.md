# Scientific design of the consolidated experiment

## Study scope

The consolidated experiment is a single-site case study of hourly ambient ozone (O3) at Bexley–Belvedere West, London, United Kingdom, using OpenAQ sensor `1304560`. The observation period runs from 28 April 2021 to 31 December 2023. Historical meteorological predictors are obtained from the Open-Meteo Historical Weather API and aligned to the ozone time series.

The study does not constitute a multi-city transfer-learning experiment and does not cover PM2.5, PM10, or NO2 in its final modelling analysis.

## Research objective

The objective is to assess whether a neural forecasting model incorporating a simplified dynamical structure can provide competitive ozone forecasts and yield effective dynamical parameters, relative to persistence and conventional MLP baselines, at 1-, 6-, and 24-hour horizons.

The analysis addresses both predictive performance and the temporal/conditional behaviour of model errors. It does not claim that the learned parameters identify individual atmospheric processes or direct emission rates.

## Data and predictors

Ozone observations are obtained from OpenAQ. Meteorological predictors include 2-m temperature, relative humidity, dew-point temperature, 10-m wind speed and direction, shortwave radiation, cloud cover, and precipitation. Wind direction is encoded using sine and cosine components. Calendar variables are represented cyclically.

The processed data are divided chronologically into training, validation, and test partitions. Scaling parameters are fitted on the training partition only. Validation is used for model selection and diagnostic comparisons as specified by the scripts. The test partition is reserved for final out-of-sample evaluation and must not guide model or hyperparameter selection.

## Models

The evaluated models are:

1. Persistence.
2. MLP without meteorological predictors.
3. MLP with meteorological predictors.
4. Neural dynamical model with a simplified physical structure.

The physical model is based on

\[
\frac{dC}{dt}=q_\theta(x_t)-kC(t), \qquad k>0,
\]

where \(C(t)\) denotes ozone concentration, \(q_\theta(x_t)\) is a non-negative neural source term, and \(k\) is a positive learned coefficient. The analytical forecast assumes that the source term remains constant during the forecast horizon:

\[
C(t+h)=e^{-kh}C(t)+\frac{q_\theta(x_t)}{k}\left(1-e^{-kh}\right).
\]

The physical residual is evaluated using the one-hour transition specified in the training implementation. The loss combines forecast error and a physics-informed residual penalty, with the weight and optimization details fixed in the training script.

## Parameter interpretation and identifiability

The learned \(q_\theta\) and \(k\) are effective parameters of a reduced-order model. A single observed concentration series does not generally allow independent identification of emissions, chemical production, transport, deposition, and other atmospheric processes. The parameters may compensate for one another and should not be interpreted as direct emission estimates, isolated chemical production rates, or a measured atmospheric lifetime.

Meteorological associations are predictive/model-based relationships and should not be presented as causal effects without additional evidence.

## Evaluation design

Forecast horizons are 1, 6, and 24 hours. Performance is assessed using standard error metrics, including MAE and RMSE, with R² reported where appropriate. Comparisons should be made on common valid timestamps when paired inference is intended.

The analysis includes:
- comparison with persistence and MLP baselines;
- meteorological ablation;
- multi-horizon evaluation;
- paired model-error comparisons;
- temporal dependence and autocorrelation diagnostics;
- block-based and hierarchical uncertainty analyses;
- temporal stability and influential-episode diagnostics;
- concentration-stratified error analysis;
- calibration and residual diagnostics;
- inspection of learned effective dynamical parameters.

Validation-based diagnostic analyses must be distinguished from final test-set evaluation. Confidence intervals from temporal resampling should account for serial dependence and should not be described as independent-observation uncertainty.

## Reproducibility

The scripts define the data-acquisition, quality-control, chronological partitioning, training, evaluation, diagnostic, and figure-generation procedures. Random seeds and model settings are recorded in the relevant scripts. The environment manifests document the computational dependencies.

The raw data, detailed prediction files, and local model artifacts are excluded from version control. A reproduction requires retrieving the source data, recreating the intermediate datasets, and running the scripts in dependency order as documented in the root README.

## Limitations

The experiment is based on one monitoring station and one pollutant. Its findings cannot establish cross-city generalization or broad transferability. The reduced dynamical equation is an approximation and does not explicitly resolve atmospheric chemistry, advection, turbulence, or boundary-layer processes. Data gaps, sensor characteristics, the temporal dependence of errors, and the meteorological reanalysis resolution also constrain interpretation.
