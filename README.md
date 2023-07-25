![logo](docs/source/_static/wb2-logo-wide.png)
# WeatherBench 2 - A benchmark for the next generation of data-driven global weather models

[![CI](https://github.com/google-research/weatherbench2/actions/workflows/ci-build.yml/badge.svg)](https://github.com/google-research/weatherbench2/actions/workflows/ci-build.yml)

WeatherBench 2 is a framework for evaluating and comparing data-driven and traditional numerical weather forecasting models. WeatherBench consists of:
- Publicly available, cloud-optimized ground truth and baseline datasets. For a complete list, see [this page](data-guide). 
- Open-source evaluation code. See this [quick-start](evaluation) to explore the basic functionality or the [API docs](api) for more detail. Since high-resolution forecast files can be large, the WeatherBench 2 code was written with scalability in mind. See the [command-line scripts](cli) based on [Xarray-Beam](https://xarray-beam.readthedocs.io/en/latest/) and [this guide](dataflow) for running the scripts on GCP using [DataFlow](https://cloud.google.com/dataflow).
- A [website](LINK) displaying up-to-date scores of many of the state-of-the-art data-driven and physical approaches.
- A [paper] describing the rationale behind the evaluation setup.

WeatherBench 2 is supposed to be an evolving tool for the entire community. For this reason, we welcome any feedback (ideally, submitted as [GitHub issues](https://github.com/google-research/weatherbench2/issues)) or contributions. If you would like you model to be part of WeatherBench, check out [this guide](submit).


## License

This is not an official Google product.

```
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```