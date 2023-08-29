# Init vs Valid Time Conventions

The WeatherBench 2 evaluation code can work with two forecast time conventions: init-time and valid-time. For all official WB2 evaluation, the init-time convention is used. You can switch between the conventions using the `by_init` parameter in the `config.Data`.

## Init-time convention
Here, the `time` dimension of the forecast dataset refers to the initialization time of each forecast. To get the time at which each forecast step is valid, one has to add the `lead_time` (also called `prediction_timedelta`). This is the format used by ECMWF. To avoid confusion, the WB2 code internally renames the `time` dimension to `init_time` and a `valid_time` dimension is created from `init_time` + `lead_time`. 

Note that the time ranges given to the evaluation code (in the `Selection` instance) refer to the initialization time. This means that for a forecast with a lead time of 10 days, `time_slice = slice('2020-01-01', '2020-12-31')` will include forecasts that run up to `2021-01-10`. 


## Valid-time convention
In the valid-time convention, the `time` dimension on the forecast dataset refers to the time at which the forecast is valid. Internally, this is renamed to `valid_time`. To get to the initialization time, one has to subtract the lead time: `init_time = valid_time - lead_time`. 

In this case the `time_slice` argument is applied to the `valid_time`, so that `time_slice = slice('2020-01-01', '2020-12-31')` will only include forecasts that are valid in 2020. However, here some of the forecasts will have been initialized at the end of 2019.

When evaluating a full year of data, the differences in the time convention are minimal.

See `compute_init_to_valid_time.py` in the command line scripts to convert from init to valid time.

See also [this page](https://climpred.readthedocs.io/en/stable/alignment.html) from the climpred project. There, the init-time convention is referred to as `same_init`, while the valid-time convention is referred to as `same_verif`.

