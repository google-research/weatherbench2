# Command line scripts

## Evaluation
Main evaluation script.

```
usage: wb2_evaluation.py [-h] 
                         [--forecast_path FORECAST_PATH] 
                         [--obs_path OBS_PATH]
                         [--climatology_path CLIMATOLOGY_PATH] 
                         [--by_init] 
                         [--evaluate_persistence] 
                         [--evaluate_climatology] 
                         [--evaluate_probabilistic_climatology] 
                         [--probabilistic_climatology_start_year PROBABILISTIC_CLIMATOLOGY_START_YEAR] 
                         [--probabilistic_climatology_end_year PROBABILISTIC_CLIMATOLOGY_END_YEAR] 
                         [--probabilistic_climatology_hour_interval PROBABILISTIC_CLIMATOLOGY_HOUR_INTERVAL]
                         [--add_land_region]
                         [--eval_configs EVAL_CONFIGS]
                         [--ensemble_dim ENSEMBLE_DIM]
                         [--rename_variables RENAME_VARIABLES]
                         [--pressure_level_suffixes]
                         [--levels LEVELS]
                         [--variables VARIABLES]
                         [--derived_variables DERIVED_VARIABLES]
                         [--time_start TIME_START]
                         [--time_stop TIME_STOP]
                         [--output_dir OUTPUT_DIR]
                         [--output_file_prefix OUTPUT_FILE_PREFIX]
                         [--input_chunks INPUT_CHUNKS]
                         [--use_beam]
                         [--beam_runner BEAM_RUNNER]
                         [--fanout FANOUT]
```

_Command options_:

* `--forecast_path`: (required) Path to forecasts to evaluate in Zarr format.
* `--obs_path`: (required) Path to ground-truth to evaluate in Zarr format.
* `--climatology_path`: Path to climatology. Used to compute e.g. ACC.
* `--by_init`: Specifies whether forecasts are in by-init or by-valid format.
* `--evaluate_persistence`: Evaluate persistence forecast, i.e. forecast at t=0
* `--evaluate_climatology`: Evaluate climatology forecast specified in climatology path
* `--evaluate_probabilistic_climatology`: Evaluate probabilistic climatology,
      derived from using each year of the ground-truth dataset as a member
* `--probabilistic_climatology_start_year`: First year of ground-truth to use
      for probabilistic climatology
* `--probabilistic_climatology_end_year`: Last year of ground-truth to use
      for probabilistic climatology
* `--probabilistic_climatology_hour_interval`: Hour interval to compute 
      probabilistic climatology. Default: 6
* `--add_land_region`: Add land-only evaluation. `land_sea_mask` must be in observation'
        'dataset.
* `--eval_configs`: Comma-separated list of evaluation configs to run. See details below. Default: `deterministic`
* `--ensemble_dim`: Ensemble dimension name for ensemble metrics. Default: `number`.
* `--rename_variables`: Dictionary of variable to rename to standard names. E.g. {"2t": "2m_temperature"}
* `--pressure_level_suffixes`: 'Decode pressure levels as suffixes in forecast file. E.g.'
        ' temperature_850'
* `--levels`: Comma delimited list of pressure levels to select for evaluation. Default: `500,700,850`
* `--variables`: Comma delimited list of variables to select from weather. Default: `geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,mean_sea_level_pressure`
* `--derived_variables`: Comma delimited list of derived variables to dynamically compute during evaluation. Default: `None`
* `--time_start`: ISO 8601 timestamp (inclusive) at which to start evaluation. Default: `2020-01-01`
* `--time_stop`: ISO 8601 timestamp (inclusive) at which to stop evaluation. Default: `2020-12-31`
* `--output_dir`: Directory in which to save evaluation results in netCDF format.
* `--output_file_prefix`: Prefix of results filename. If "_" or "-" is desired, please add specifically here.
* `--input_chunks`: chunk sizes overriding input chunks to use for loading forecast and observation data. By default, omitted dimension names are loaded in one chunk. Default: `time=1`
* `--use_beam`: Run evaluation pipeline as beam pipeline. If False, run in memory.
* `--beam_runner`: Beam runner
* `--fanout`: Beam CombineFn fanout. Might be required for large dataset. Default: `None`

*Predefined evaluation configs*

```
deterministic_metrics = {
  'rmse': RMSE(wind_vector_rmse=_wind_vector_rmse()),
  'mse': MSE(),
  'acc': ACC(climatology=climatology),
}

eval_configs = {
  'deterministic': EvalConfig(
      metrics=deterministic_metrics,
      against_analysis=False,
      regions=regions,
      derived_variables=derived_variables,
      evaluate_persistence=EVALUATE_PERSISTENCE.value,
      evaluate_climatology=EVALUATE_CLIMATOLOGY.value,
  ),
  'deterministic_spatial': EvalConfig(
      metrics={'bias': SpatialBias(), 'mse': SpatialMSE()},
      against_analysis=False,
      derived_variables=derived_variables,
      evaluate_persistence=EVALUATE_PERSISTENCE.value,
      evaluate_climatology=EVALUATE_CLIMATOLOGY.value,
  ),
  'deterministic_temporal': EvalConfig(
      metrics=deterministic_metrics,
      against_analysis=False,
      regions=regions,
      derived_variables=derived_variables,
      evaluate_persistence=EVALUATE_PERSISTENCE.value,
      evaluate_climatology=EVALUATE_CLIMATOLOGY.value,
      temporal_mean=False,
  ),
  'probabilistic': EvalConfig(
      metrics={
          'crps': CRPS(ensemble_dim=ENSEMBLE_DIM.value),
          'ensemble_mean_rmse': EnsembleMeanRMSE(
              ensemble_dim=ENSEMBLE_DIM.value
          ),
          'ensemble_stddev': EnsembleStddev(
              ensemble_dim=ENSEMBLE_DIM.value
          ),
      },
      against_analysis=False,
      derived_variables=derived_variables,
      evaluate_probabilistic_climatology=EVALUATE_PROBABILISTIC_CLIMATOLOGY.value,
      probabilistic_climatology_start_year=PROBABILISTIC_CLIMATOLOGY_START_YEAR.value,
      probabilistic_climatology_end_year=PROBABILISTIC_CLIMATOLOGY_END_YEAR.value,
      probabilistic_climatology_hour_interval=PROBABILISTIC_CLIMATOLOGY_HOUR_INTERVAL.value,
  ),
}
```

*Example*

Local evaluation.

TODO

```bash
python wb2_evaluation.py \
  --forecast_path=gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_with_poles_conservative.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_with_poles_conservative.zarr \
  --output_dir=./ \
  --output_file_prefix=hres_vs_era_2020_ \
  --input_chunks=init_time=1 \
  --eval_configs=deterministic,deterministic_spatial,deterministic_temporal \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed
```

TODO: Beam version

## Climatology

```
usage: wb2_compute_climatology.py [-h] 
                                  [--input_path INPUT_PATH] 
                                  [--output_path OUTPUT_PATH]
                                  [--hour_interval HOUR_INTERVAL] 
                                  [--window_size WINDOW_SIZE] 
                                  [--start_year START_YEAR] 
                                  [--end_year END_YEAR] 
                                  [--working_chunks WORKING_CHUNKS] 
                                  [--rechunk_itemsize RECHUNK_ITEMSIZE] 
                                  [--output_chunks OUTPUT_CHUNKS]
                                  [--statistics STATISTICS]
                                  [--add_statistic_suffix]
                                  [--method METHOD]
                                  [--seeps_dry_threshold_mm SEEPS_DRY_THRESHOLD_MM]
                                  [--beam_runner BEAM_RUNNER]

```

_Command options_:

* `--input_path`: (required) Input Zarr path
* `--output_path`: (required) Output Zarr path
* `--hour_interval`: Which intervals to compute hourly climatology for. Default: `1`
* `--window_size`: Window size in days to average over. Default: `61` 
* `--start_year`: Inclusive start year of climatology. Default: `1990`
* `--end_year`: Inclusive end year of climatology. Default: `2019`
* `--working_chunks`: Chunk sizes overriding input chunks to use for computing climatology, e.g., "longitude=10,latitude=10".
* `--rechunk_itemsize`: Itemsize for rechunking.. Default: `4`
* `--output_chunks`: Chunk sizes overriding input chunks to use for storing climatology.
* `--statistics`: List of statistics to compute from "mean", "std", "seeps". Default: `["mean"]`
* `--add_statistic_suffix`: Add suffix of statistic to variable name. Required for >1 statistic.
* `--method`: Computation method to use. "explicit": Stack years first, apply rolling and then compute weighted statistic over (year, rolling_window). "fast": Compute statistic over day-of-year first and then apply weighted smoothing. Mathematically equivalent for mean but different for nonlinear statistics. Default: `explicit`
* `--seeps_dry_threshold_mm`: Dict defining dry threshold for SEEPS quantile computation for each precipitation variable. In mm. Default: `"{'total_precipitation_24hr':0.25, 'total_precipitation_6hr':0.1}"`
* `--beam_runner`: Beam runner. Use `DirectRunner` for local execution.

*Example*

Local evaluation. 

TODO

```bash
python wb2_compute_climatology.py \
  --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
  --output_path=PATH \
  --working_chunks=level=13,longitude=4,latitude=4 \
  --output_chunks=level=13,hour=4 \
  --start_year=1990 \
  --end_year=2019 \
  --hour_interval=6 \
  --statistics=mean,seeps
```

## Compute derived variables

Computes derived variables, adds them to the original dataset and saves it as a new file. See derived_variables.py for a list of available derived variables.

```
usage: wb2_compute_derived_variables.py [-h] 
                                        [--input_path INPUT_PATH] 
                                        [--output_path OUTPUT_PATH]
                                        [--derived_variables DERIVED_VARIABLES] 
                                        [--raw_tp_name RAW_TP_NAME] 
                                        [--rename_raw_tp_name] 
                                        [--working_chunks WORKING_CHUNKS] 
                                        [--rechunk_itemsize RECHUNK_ITEMSIZE] 
                                        [--max_mem_gb MAX_MEM_GB] 
                                        [--runner RUNNER]
```

_Command options_:

* `--input_path`: (required) Input Zarr path
* `--output_path`: (required) Output Zarr path
* `--derived_variables`: (required) Comma delimited list of derived variables to compute. Default: `wind_speed,10m_wind_speed,total_precipitation_6hr,total_precipitation_24hr`
* `--raw_tp_name`: Raw name of total precipitation variables. Use "total_precipitation_6hr" for backwards compatibility. 
* `--rename_raw_tp_name`: Rename raw tp name to "total_precipitation".
* `--working_chunks`: Chunk sizes overriding input chunks to use for computing aggregations e.g., "longitude=10,latitude=10". No need to add prediction_timedelta=-1, this is automatically added for aggregation variables. Default: `None`, i.e. input chunks
* `--rechunk_itemsize`: Itemsize for rechunking. Default: `4`
* `--max_mem_gb`: Max memory for rechunking in GB. Default: `1`
* `--runner`: Beam runner. Use `DirectRunner` for local execution.

*Example*

Local evaluation. 

TODO

```bash
python wb2_compute_derived_variables.py \
  --input_path=gs://weatherbench2/datasets/hres/2016-2022-12h-6h-0p25deg-chunk-1.zarr/ \
  --output_path=PATH \
  --working_chunks=prediction_timedelta=-1 \
  --rename_raw_tp_name=True \
  --raw_tp_name=total_precipitation_6hr \
  --rechunk_itemsize=1 \
```

## Compute Spectra

Computes zonal energy spectra.

```
usage: wb2_compute_zonal_energy_spectrum.py [-h] 
                                            [--input_path INPUT_PATH] 
                                            [--output_path OUTPUT_PATH]
                                            [--base_variables BASE_VARIABLES] 
                                            [--time_dim TIME_DIM] 
                                            [--time_start TIME_START] 
                                            [--time_stop TIME_STOP] 
                                            [--levels LEVELS] 
                                            [--averaging_dims AVERAGING_DIMS]
                                            [--runner RUNNER] 
```

_Command options_:

* `--input_path`: (required) Input Zarr path
* `--output_path`: (required) Output Zarr path
* `--base_variables`: Comma delimited list of variables in --input_path. Each variable VAR results in a VAR_zonal_power_spectrum entry in --output_path. Default: `geopotential,specific_humidity,temperature`
* `--time_dim`: Name for the time dimension to slice data on. Default: `time`
* `--time_start`: ISO 8601 timestamp (inclusive) at which to start evaluation. Default: `2020-01-01'`
* `--time_stop`: ISO 8601 timestamp (inclusive) at which to stop evaluation. Default: `2020-12-31`
* `--levels`: Comma delimited list of pressure levels to compute spectra on. If empty, compute on all levels of --input_path. Default: `500,700,850`
* `--averaging_dims`: Comma delimited list of variables to average over. If empty, do not average. Default: `time`
* `--runner`: Beam runner. Use `DirectRunner` for local execution.

*Example*

Local evaluation. 

TODO

```bash
python wb2_compute_zonal_power_spectrum.py \
  --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr  \
  --output_path=PATH \
  --time_start=2020 \
  --time_stop=2020 \
  --base_variables=geopotential,specific_humidity,temperature,u_component_of_wind,v_component_of_wind,wind_speed,10m_u_component_of_wind,10m_v_component_of_wind,10m_wind_speed,2m_temperature,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr
```

## Compute ensemble mean

To use the ensemble mean in deterministic evaluation, we first must compute the mean and save it in a separate file.

```
usage: wb2_compute_ensemble_mean.py [-h] 
                                    [--input_path INPUT_PATH] 
                                    [--output_path OUTPUT_PATH]
                                    [--realization_name REALIZATION_NAME]
                                    [--runner RUNNER] 
```

_Command options_:

* `--input_path`: (required) Input Zarr path
* `--output_path`: (required) Output Zarr path
* `--realization_name`: Name of realization/member/number dimension. Default: `realization`
* `--runner`: Beam runner. Use `DirectRunner` for local execution.

*Example*

Local evaluation. 

TODO

```bash
python ensemble_mean.py -- \
  --input_path=gs://weatherbench2/datasets/ens/2018-64x32_equiangular_with_poles_conservative.zarr \
  --output_path=PATH \
  --realization_name=number
```

## What about regridding?
We plan to open-source our regridding script soon.

## Expand climatology

`wb2_expand_climatology.py` takes a climatology dataset and expands it into a forecast-like format (`init_time` + `lead_time`). This is not currently used as `wb2_evaluation.py` is able to do this on-the-fly, reducing the number of intermediate steps. We still included the script here in case others find it useful. 

## Init to valid time conversion

`wb2_init_to_valid_time.py` converts a forecasts in init-time convention to valid-time convention. Since currently, we do all evaluation in the init-time format, this script is not used. 