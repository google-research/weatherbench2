# Submit a new model to WeatherBench 2

You have the latest and greatest new ML model? Awesome! Here are the steps required to evaluate it in the WeatherBench 2 framework. If there are any questions, please don't hesitate to ask them via a GitHub issue.

## Save forecasts in a standardized Zarr format

The WeatherBench evaluation pipeline expects forecasts and ground truth to be stored as Zarr files. See the Quickstart [LINK] for a description of the format. Saving all variables is too much. You can also select specific levels and variables. See the Pangu forecasts for an example: [LINK]

We recommend having single chunks in space (latitude and longitude) but small chunks in time to allow for efficient paralellization.

## Regrid

The WeatherBench 2 code can be used to evaluate any resolution. However, we only provide ground truth data for a range of resolutions (see Data Guide [LINK]). The official reesolution used in the paper and the website is 1.5 degrees. Note that absolute values of metrics computed at different resolution are not necessarily comparable (see paper.)

We are still working to open-source the regridding code. Until then, alternatives are TODO.

Note that following the WMO standards we are using conservative regridding.

## Run the evaluation script



## "Submit" the results

To have your model appear on the WeatherBench 2 website, you can either provide us with your forecasts (create a GitHub issue or contact us directly at weatherbench2@google.com). We will then evaluate the forecast. Or you evaluate the forecasts yourself and share your evaluation code and results with us. Then we will update the results on the website. 


