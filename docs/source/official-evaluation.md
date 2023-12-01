(official-evaluation)=
# Official Evaluation

Below, you will find the command line scripts used for the evaluation shown on the official WeatherBench 2 website.

Only 2020 results are shown. Dataflow options are omitted. 

## Deterministic Evaluation

### 64x32 Resolution

```bash
python evaluate.py \
  --forecast_path=gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr \
  --output_dir=$OUTDIR/64x32/deterministic/ \
  --output_file_prefix=hres_vs_era_2020_ \
  --input_chunks=init_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed \
  --compute_seeps=True \
  --use_beam=True
```

```bash
python evaluate.py \
  --forecast_path=gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr \
  --obs_path=gs://weatherbench2/datasets/hres_t0/2016-2022-6h-64x32_equiangular_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr \
  --output_dir=$OUTDIR/64x32/deterministic/ \
  --output_file_prefix=hres_vs_analysis_2020_ \
  --input_chunks=init_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,10m_wind_speed,wind_speed \
  --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/ens/2018-2022-64x32_equiangular_conservative_mean.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr \
  --output_dir=$OUTDIR/64x32/deterministic/ \
  --output_file_prefix=ens_vs_era_2020_ \
  --input_chunks=init_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed \
  --compute_seeps=True \
  --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/ens/2018-2022-64x32_equiangular_conservative_mean.zarr \
  --obs_path=gs://weatherbench2/datasets/hres_t0/2016-2022-6h-64x32_equiangular_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr \
  --output_dir=$OUTDIR/64x32/deterministic/ \
  --output_file_prefix=ens_vs_analysis_2020_ \
  --input_chunks=init_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,10m_wind_speed,wind_speed \
  --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr \
  --output_dir=$OUTDIR/64x32/deterministic/ \
  --output_file_prefix=climatology_vs_era_2020_ \
  --input_chunks=init_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=True \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed \
  --compute_seeps=True \
  --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr \
  --output_dir=$OUTDIR/64x32/deterministic/ \
  --output_file_prefix=persistence_vs_era_2020_ \
  --input_chunks=init_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=False \
  --evaluate_persistence=True \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed \
  --compute_seeps=True \
  --use_beam=True
```

```bash
python evaluate.py -- \
 --forecast_path=gs://weatherbench2/datasets/pangu/2018-2022_0012_64x32_equiangular_conservative.zarr \
 --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
 --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr \
 --output_dir=$OUTDIR/64x32/deterministic/ \
 --output_file_prefix=pangu_vs_era_2020_ \
 --input_chunks=init_time=1 \
 --fanout=27 \
 --regions=all \
 --eval_configs=deterministic,deterministic_temporal \
 --evaluate_climatology=False \
 --evaluate_persistence=False \
 --time_start=2020-01-01 \
 --time_stop=2020-12-31 \
 --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure \
 --add_land_region=False \
 --use_beam=True
```

```bash
python evaluate.py -- \
 --forecast_path=gs://weatherbench2/datasets/pangu/pangu_hres_init/2020_0012_64x32_equiangular_conservative.zarr \
 --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
 --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr \
 --output_dir=$OUTDIR/64x32/deterministic/ \
 --output_file_prefix=pangu_hres_init_vs_era_2020_ \
 --input_chunks=init_time=1 \
 --fanout=27 \
 --regions=all \
 --eval_configs=deterministic,deterministic_temporal \
 --evaluate_climatology=False \
 --evaluate_persistence=False \
 --time_start=2020-01-01 \
 --time_stop=2020-12-31 \
 --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure \
 --add_land_region=False \
 --use_beam=True
```

```bash
python evaluate.py -- \
 --forecast_path=gs://weatherbench2/datasets/pangu_hres_init/2020_0012_64x32_equiangular_conservative.zarr \
 --obs_path=gs://weatherbench2/datasets/hres_t0/2016-2022-6h-64x32_equiangular_conservative.zarr \
 --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr \
 --output_dir=$OUTDIR/64x32/deterministic/ \
 --output_file_prefix=pangu_hres_init_vs_analysis_2020_ \
 --input_chunks=init_time=1 \
 --fanout=27 \
 --regions=all \
 --eval_configs=deterministic,deterministic_temporal \
 --evaluate_climatology=False \
 --evaluate_persistence=False \
 --time_start=2020-01-01 \
 --time_stop=2020-12-31 \
 --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure \
 --add_land_region=False \
 --use_beam=True
```

```bash
python evaluate.py -- \
 --forecast_path=gs://weatherbench2/datasets/keisler/2020-64x32_equiangular_conservative.zarr \
 --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
 --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr \
 --output_dir=$OUTDIR/64x32/deterministic/ \
 --output_file_prefix=keisler_vs_era_2020_ \
 --input_chunks=init_time=1 \
 --fanout=27 \
 --regions=all \
 --eval_configs=deterministic,deterministic_temporal \
 --evaluate_climatology=False \
 --evaluate_persistence=False \
 --time_start=2020-01-01 \
 --time_stop=2020-12-31 \
 --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity \
 --derived_variables=wind_speed \
 --add_land_region=False \
 --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/era5-forecasts/2020-64x32_equiangular_conservative.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr \
  --output_dir=$OUTDIR/64x32/deterministic/ \
  --output_file_prefix=era5-forecasts_vs_era_2020_ \
  --input_chunks=init_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,10m_wind_speed,wind_speed \
  --use_beam=True
```

```bash
python evaluate.py -- \
 --forecast_path=gs://weatherbench2/datasets/sphericalcnn/2020-64x32_equiangular_conservative.zarr \
 --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
 --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr \
 --output_dir=$OUTDIR/64x32/deterministic/ \
 --output_file_prefix=sphericalcnn_vs_era_2020_ \
 --input_chunks=init_time=1 \
 --fanout=27 \
 --regions=all \
 --eval_configs=deterministic,deterministic_temporal \
 --evaluate_climatology=False \
 --evaluate_persistence=False \
 --time_start=2020-01-01 \
 --time_stop=2020-12-31 \
 --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity \
 --derived_variables=wind_speed \
 --add_land_region=False \
 --use_beam=True
```

```bash
python evaluate.py -- \
 --forecast_path=gs://weatherbench2/datasets/fuxi/2020-64x32_equiangular_conservative.zarr \
 --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
 --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr \
 --output_dir=$OUTDIR/64x32/deterministic/ \
 --output_file_prefix=fuxi_vs_era_2020_ \
 --input_chunks=init_time=1 \
 --fanout=27 \
 --regions=all \
 --eval_configs=deterministic,deterministic_temporal \
 --evaluate_climatology=False \
 --evaluate_persistence=False \
 --time_start=2020-01-01 \
 --time_stop=2020-12-31 \
 --levels=500,850 \
 --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,10m_wind_speed,wind_speed \
 --add_land_region=False \
 --use_beam=True
```

### 240x121 Resolution

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
  --output_dir=$OUTDIR/240x121/deterministic/ \
  --output_file_prefix=hres_vs_era_2020_ \
  --input_chunks=init_time=1,lead_time=10 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed \
  --compute_seeps=True \
  --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr \
  --obs_path=gs://weatherbench2/datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
  --lsm_dataset=gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
  --output_dir=$OUTDIR/240x121/deterministic/ \
  --output_file_prefix=hres_vs_analysis_2020_ \
  --input_chunks=init_time=1,lead_time=10 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,10m_wind_speed,wind_speed \
  --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/ens/2018-2022-240x121_equiangular_with_poles_conservative_mean.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
  --output_dir=$OUTDIR/240x121/deterministic/ \
  --output_file_prefix=ens_vs_era_2020_ \
  --input_chunks=init_time=1,lead_time=10 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed \
  --compute_seeps=True \
  --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/ens/2018-2022-240x121_equiangular_with_poles_conservative_mean.zarr \
  --obs_path=gs://weatherbench2/datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
  --lsm_dataset=gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
  --output_dir=$OUTDIR/240x121/deterministic/ \
  --output_file_prefix=ens_vs_analysis_2020_ \
  --input_chunks=init_time=1,lead_time=10 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,10m_wind_speed,wind_speed \
  --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
  --output_dir=$OUTDIR/240x121/deterministic/ \
  --output_file_prefix=climatology_vs_era_2020_ \
  --input_chunks=init_time=1,lead_time=10 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=True \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed \
  --compute_seeps=True \
  --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
  --output_dir=$OUTDIR/240x121/deterministic/ \
  --output_file_prefix=persistence_vs_era_2020_ \
  --input_chunks=init_time=1,lead_time=10 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=False \
  --evaluate_persistence=True \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed \
  --compute_seeps=True \
  --use_beam=True
```

```bash
python evaluate.py -- \
 --forecast_path=gs://weatherbench2/datasets/pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr \
 --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
 --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
 --output_dir=$OUTDIR/240x121/deterministic/ \
 --output_file_prefix=pangu_vs_era_2020_ \
 --input_chunks=init_time=1,lead_time=10 \
 --fanout=27 \
 --regions=all \
 --eval_configs=deterministic,deterministic_temporal \
 --evaluate_climatology=False \
 --evaluate_persistence=False \
 --time_start=2020-01-01 \
 --time_stop=2020-12-31 \
 --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure \
 --add_land_region=False \
 --use_beam=True
```

```bash
python evaluate.py -- \
 --forecast_path=gs://weatherbench2/datasets/pangu_hres_init/2020_0012_240x121_equiangular_with_poles_conservative.zarr \
 --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
 --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
 --output_dir=$OUTDIR/240x121/deterministic/ \
 --output_file_prefix=pangu_hres_init_vs_era_2020_ \
 --input_chunks=init_time=1,lead_time=10 \
 --fanout=27 \
 --regions=all \
 --eval_configs=deterministic,deterministic_temporal \
 --evaluate_climatology=False \
 --evaluate_persistence=False \
 --time_start=2020-01-01 \
 --time_stop=2020-12-31 \
 --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure \
 --add_land_region=False \
 --use_beam=True
```

```bash
python evaluate.py -- \
 --forecast_path=gs://weatherbench2/datasets/pangu_hres_init/2020_0012_240x121_equiangular_with_poles_conservative.zarr \
 --obs_path=gs://weatherbench2/datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
 --lsm_dataset=gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
 --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
 --output_dir=$OUTDIR/240x121/deterministic/ \
 --output_file_prefix=pangu_hres_init_vs_analysis_2020_ \
 --input_chunks=init_time=1,lead_time=10 \
 --fanout=27 \
 --regions=all \
 --eval_configs=deterministic,deterministic_temporal \
 --evaluate_climatology=False \
 --evaluate_persistence=False \
 --time_start=2020-01-01 \
 --time_stop=2020-12-31 \
 --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure \
 --add_land_region=False \
 --use_beam=True
```

```bash
python evaluate.py -- \
 --forecast_path=gs://weatherbench2/datasets/keisler/2020-240x121_equiangular_with_poles_conservative.zarr \
 --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
 --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
 --output_dir=$OUTDIR/240x121/deterministic/ \
 --output_file_prefix=keisler_vs_era_2020_ \
 --input_chunks=init_time=1,lead_time=10 \
 --fanout=27 \
 --regions=all \
 --eval_configs=deterministic,deterministic_temporal \
 --evaluate_climatology=False \
 --evaluate_persistence=False \
 --time_start=2020-01-01 \
 --time_stop=2020-12-31 \
 --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity \
 --derived_variables=wind_speed \
 --add_land_region=False \
 --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/era5-forecasts/2020-240x121_equiangular_with_poles_conservative.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
  --output_dir=$OUTDIR/240x121/deterministic/ \
  --output_file_prefix=era5-forecasts_vs_era_2020_ \
  --input_chunks=init_time=1,lead_time=10 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,10m_wind_speed,wind_speed \
  --use_beam=True
```

```bash
python evaluate.py -- \
 --forecast_path=gs://weatherbench2/datasets/sphericalcnn/2020-240x121_equiangular_with_poles_conservative.zarr \
 --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
 --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
 --output_dir=$OUTDIR/240x121/deterministic/ \
 --output_file_prefix=sphericalcnn_vs_era_2020_ \
 --input_chunks=init_time=1,lead_time=10 \
 --fanout=27 \
 --regions=all \
 --eval_configs=deterministic,deterministic_temporal \
 --evaluate_climatology=False \
 --evaluate_persistence=False \
 --time_start=2020-01-01 \
 --time_stop=2020-12-31 \
 --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity \
 --derived_variables=wind_speed \
 --add_land_region=False \
 --use_beam=True
```

```bash
python evaluate.py -- \
 --forecast_path=gs://weatherbench2/datasets/fuxi/2020-240x121_equiangular_with_poles_conservative.zarr \
 --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
 --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
 --output_dir=$OUTDIR/240x121/deterministic/ \
 --output_file_prefix=fuxi_vs_era_2020_ \
 --input_chunks=init_time=1,lead_time=10 \
 --fanout=27 \
 --regions=all \
 --eval_configs=deterministic,deterministic_temporal \
 --evaluate_climatology=False \
 --evaluate_persistence=False \
 --time_start=2020-01-01 \
 --time_stop=2020-12-31 \
 --levels=500,850 \
 --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,10m_wind_speed,wind_speed \
 --add_land_region=False \
 --use_beam=True
```

### 1440x721 resolution

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr \
  --output_dir=$OUTDIR/1440x721/deterministic/ \
  --output_file_prefix=hres_vs_era_2020_ \
  --input_chunks=init_time=1,lead_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed \
  --compute_seeps=True \
  --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr \
  --obs_path=gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr \
  --lsm_dataset=gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr \
  --output_dir=$OUTDIR/1440x721/deterministic/ \
  --output_file_prefix=hres_vs_analysis_2020_ \
  --input_chunks=init_time=1,lead_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,10m_wind_speed,wind_speed \
  --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/ens/2018-2022-1440x721_mean.zarr/ \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr \
  --output_dir=$OUTDIR/1440x721/deterministic/ \
  --output_file_prefix=ens_vs_era_2020_ \
  --input_chunks=init_time=1,lead_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed \
  --compute_seeps=True \
  --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/ens/2018-2022-1440x721_mean.zarr/ \
  --obs_path=gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr \
  --lsm_dataset=gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr \
  --output_dir=$OUTDIR/1440x721/deterministic/ \
  --output_file_prefix=ens_vs_analysis_2020_ \
  --input_chunks=init_time=1,lead_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,10m_wind_speed,wind_speed \
  --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr \
  --output_dir=$OUTDIR/1440x721/deterministic/ \
  --output_file_prefix=climatology_vs_era_2020_ \
  --input_chunks=init_time=1,lead_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=True \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed \
  --compute_seeps=True \
  --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr \
  --output_dir=$OUTDIR/1440x721/deterministic/ \
  --output_file_prefix=persistence_vs_era_2020_ \
  --input_chunks=init_time=1,lead_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=False \
  --evaluate_persistence=True \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed \
  --compute_seeps=True \
  --use_beam=True
```

```bash
python evaluate.py -- \
 --forecast_path=gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr \
 --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr \
 --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr \
 --output_dir=$OUTDIR/1440x721/deterministic/ \
 --output_file_prefix=pangu_vs_era_2020_ \
 --input_chunks=init_time=1,lead_time=1 \
 --fanout=27 \
 --regions=all \
 --eval_configs=deterministic,deterministic_temporal \
 --evaluate_climatology=False \
 --evaluate_persistence=False \
 --time_start=2020-01-01 \
 --time_stop=2020-12-31 \
 --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure \
 --add_land_region=False \
 --use_beam=True
```

```bash
python evaluate.py -- \
 --forecast_path=gs://weatherbench2/datasets/pangu_hres_init/2020_0012_0p25.zarr \
 --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr \
 --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr \
 --output_dir=$OUTDIR/1440x721/deterministic/ \
 --output_file_prefix=pangu_hres_init_vs_era_2020_ \
 --input_chunks=init_time=1,lead_time=1 \
 --fanout=27 \
 --regions=all \
 --eval_configs=deterministic,deterministic_temporal \
 --evaluate_climatology=False \
 --evaluate_persistence=False \
 --time_start=2020-01-01 \
 --time_stop=2020-12-31 \
 --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure \
 --add_land_region=False \
 --use_beam=True
```

```bash
python evaluate.py -- \
 --forecast_path=gs://weatherbench2/datasets/pangu_hres_init/2020_0012_0p25.zarr \
 --obs_path=gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr \
 --lsm_dataset=gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr \
 --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr \
 --output_dir=$OUTDIR/1440x721/deterministic/ \
 --output_file_prefix=pangu_hres_init_vs_analysis_2020_ \
 --input_chunks=init_time=1,lead_time=1 \
 --fanout=27 \
 --regions=all \
 --eval_configs=deterministic,deterministic_temporal \
 --evaluate_climatology=False \
 --evaluate_persistence=False \
 --time_start=2020-01-01 \
 --time_stop=2020-12-31 \
 --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure \
 --add_land_region=False \
 --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/era5-forecasts/2020-1440x721.zarr/ \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr \
  --output_dir=$OUTDIR/1440x721/deterministic/ \
  --output_file_prefix=era5-forecasts_vs_era_2020_ \
  --input_chunks=init_time=1,lead_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,10m_wind_speed,wind_speed \
  --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/fuxi/2020-1440x721.zarr/ \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr \
  --output_dir=$OUTDIR/1440x721/deterministic/ \
  --output_file_prefix=era5-fuxi_vs_era_2020_ \
  --input_chunks=init_time=1,lead_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=deterministic,deterministic_temporal \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --levels=500,850 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,10m_wind_speed,wind_speed \
  --use_beam=True
```

## Probabilistic evaluation

### 64x32 Resolution

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/ens/2018-2022-64x32_equiangular_conservative.zarr \
  --obs_path=gs://weatherbench2/datasets/hres_t0/2016-2022-6h-64x32_equiangular_conservative.zarr \
  --lsm_dataset=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr \
  --output_dir=$OUTDIR/64x32/probabilistic/ \
  --output_file_prefix=ens_vs_analysis_2020_ \
  --input_chunks=init_time=1,lead_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=probabilistic \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,10m_wind_speed,wind_speed \
  --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/ens/2018-2022-64x32_equiangular_conservative.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr \
  --output_dir=$OUTDIR/64x32/probabilistic/ \
  --output_file_prefix=ens_vs_era_2020_ \
  --input_chunks=init_time=1,lead_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=probabilistic \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed \
  --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/ens/2018-2022-64x32_equiangular_conservative.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr \
  --output_dir=$OUTDIR/64x32/probabilistic/ \
  --output_file_prefix=climatology_vs_era_2020_ \
  --input_chunks=init_time=1,lead_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=probabilistic \
  --evaluate_probabilistic_climatology=True \
  --probabilistic_climatology_start_year=1990 \
  --probabilistic_climatology_end_year=2019 \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed \
  --use_beam=True
```

### 240x121 Resolution

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr \
  --obs_path=gs://weatherbench2/datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
  --lsm_dataset=gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
  --output_dir=$OUTDIR/240x121/probabilistic/ \
  --output_file_prefix=ens_vs_analysis_2020_ \
  --input_chunks=init_time=1,lead_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=probabilistic \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,10m_wind_speed,wind_speed \
  --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
  --output_dir=$OUTDIR/240x121/probabilistic/ \
  --output_file_prefix=ens_vs_era_2020_ \
  --input_chunks=init_time=1,lead_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=probabilistic \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed \
  --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
  --output_dir=$OUTDIR/240x121/probabilistic/ \
  --output_file_prefix=climatology_vs_era_2020_ \
  --input_chunks=init_time=1,lead_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=probabilistic \
  --evaluate_probabilistic_climatology=True \
  --probabilistic_climatology_start_year=1990 \
  --probabilistic_climatology_end_year=2019 \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed \
  --use_beam=True
```

### 1440x721 Resolution

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/ens/2018-2022-1440x721.zarr \
  --obs_path=gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr \
  --lsm_dataset=gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr \
  --output_dir=$OUTDIR/1440x721/probabilistic/ \
  --output_file_prefix=ens_vs_analysis_2020_ \
  --input_chunks=init_time=1,lead_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=probabilistic \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,10m_wind_speed,wind_speed \
  --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/ens/2018-2022-1440x721.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr \
  --output_dir=$OUTDIR/1440x721/probabilistic/ \
  --output_file_prefix=ens_vs_era_2020_ \
  --input_chunks=init_time=1,lead_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=probabilistic \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed \
  --use_beam=True
```

```bash
python evaluate.py -- \
  --forecast_path=gs://weatherbench2/datasets/ens/2018-2022-1440x721.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr \
  --output_dir=$OUTDIR/1440x721/probabilistic/ \
  --output_file_prefix=climatology_vs_era_2020_ \
  --input_chunks=init_time=1,lead_time=1 \
  --fanout=27 \
  --regions=all \
  --eval_configs=probabilistic \
  --evaluate_probabilistic_climatology=True \
  --probabilistic_climatology_start_year=1990 \
  --probabilistic_climatology_end_year=2019 \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed \
  --use_beam=True
```

## Spectra

```bash
python compute_zonal_energy_spectrum.py -- \
 --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr  \
 --output_path=$OUTDIR/highest_res/spectra/era_2020.zarr \
 --time_start=2020 \
 --time_stop=2020 \
 --base_variables=geopotential,specific_humidity,temperature,u_component_of_wind,v_component_of_wind,wind_speed,10m_u_component_of_wind,10m_v_component_of_wind,10m_wind_speed,2m_temperature,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr \
```

```bash
python compute_zonal_energy_spectrum.py -- \
 --input_path=gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr  \
 --output_path=$OUTDIR/highest_res/spectra/hres_2020.zarr \
 --time_start=2020 \
 --time_stop=2020 \
 --base_variables=geopotential,specific_humidity,temperature,u_component_of_wind,v_component_of_wind,wind_speed,10m_u_component_of_wind,10m_v_component_of_wind,10m_wind_speed,2m_temperature,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr \
```

```bash
python compute_zonal_energy_spectrum.py -- \
 --input_path=gs://weatherbench2/datasets/ens/2018-2022-1440x721_mean.zarr/  \
 --output_path=$OUTDIR/highest_res/spectra/ens_mean_2020.zarr \
 --time_start=2020 \
 --time_stop=2020 \
 --base_variables=geopotential,specific_humidity,temperature,u_component_of_wind,v_component_of_wind,wind_speed,10m_u_component_of_wind,10m_v_component_of_wind,10m_wind_speed,2m_temperature,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr \
```

```bash
python compute_zonal_energy_spectrum.py -- \
 --input_path=gs://weatherbench2/datasets/pangu/2020-1440x721_raw.zarr  \
 --output_path=$OUTDIR/highest_res/spectra/pangu_2020.zarr \
 --time_start=2020 \
 --time_stop=2020 \
 --base_variables=geopotential_500,specific_humidity_700,temperature_850,u_component_of_wind_850,v_component_of_wind_850,10m_u_component_of_wind,10m_v_component_of_wind,2m_temperature,mean_sea_level_pressure \
```

```bash
python compute_zonal_energy_spectrum.py -- \
 --input_path=gs://weatherbench2/datasets/keisler/2020-360x181.zarr  \
 --output_path=$OUTDIR/highest_res/spectra/keisler_2020.zarr \
 --time_start=2020 \
 --time_stop=2020 \
 --base_variables=geopotential,specific_humidity,temperature,u_component_of_wind,v_component_of_wind,wind_speed \
```

```bash
python compute_zonal_energy_spectrum.py -- \
 --input_path=gs://weatherbench2/datasets/era5-forecasts/2020-1440x721.zarr/  \
 --output_path=$OUTDIR/highest_res/spectra/era5-forecasts_2020.zarr \
 --time_start=2020 \
 --time_stop=2020 \
 --base_variables=geopotential,specific_humidity,temperature,u_component_of_wind,v_component_of_wind,wind_speed,10m_u_component_of_wind,10m_v_component_of_wind,10m_wind_speed,2m_temperature,mean_sea_level_pressure \
```