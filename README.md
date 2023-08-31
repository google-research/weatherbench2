
![logo](docs/source/_static/wb2-logo-wide.png)

[![CI](https://github.com/google-research/weatherbench2/actions/workflows/ci-build.yml/badge.svg)](https://github.com/google-research/weatherbench2/actions/workflows/ci-build.yml)
[![Lint](https://github.com/google-research/weatherbench2/actions/workflows/lint.yml/badge.svg)](https://github.com/google-research/weatherbench2/actions/workflows/lint.yml)
[![Documentation Status](https://readthedocs.org/projects/weatherbench2/badge/?version=latest)](https://weatherbench2.readthedocs.io/en/latest/?badge=latest)
<a target="_blank" href="https://colab.research.google.com/github/google-research/weatherbench2/blob/main/docs/source/evaluation.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# WeatherBench 2 - A benchmark for the next generation of data-driven global weather models

[Paper](https://drive.google.com/file/d/1OqSR5H_h2y2HXEf4LVFO7ZLxgdafK2Va/view?usp=sharing)

## Why WeatherBench?

WeatherBench 2 is a framework for evaluating and comparing data-driven and traditional numerical weather forecasting models. WeatherBench consists of:
- Publicly available, cloud-optimized ground truth and baseline datasets. For a complete list, see [this page](https://weatherbench2.readthedocs.io/en/latest/data-guide.html). 
- Open-source evaluation code. See this [quick-start](https://weatherbench2.readthedocs.io/en/latest/evaluation.html) to explore the basic functionality or the [API docs](https://weatherbench2.readthedocs.io/en/latest/api.html) for more detail. Since high-resolution forecast files can be large, the WeatherBench 2 code was written with scalability in mind. See the [command-line scripts](https://weatherbench2.readthedocs.io/en/latest/command-line-scripts.html) based on [Xarray-Beam](https://xarray-beam.readthedocs.io/en/latest/) and [this guide](https://weatherbench2.readthedocs.io/en/latest/beam-in-the-cloud.html) for running the scripts on GCP using [DataFlow](https://cloud.google.com/dataflow).
- A [website](https://sites.research.google/weatherbench) displaying up-to-date scores of many of the state-of-the-art data-driven and physical approaches.
- A [paper](https://drive.google.com/file/d/1OqSR5H_h2y2HXEf4LVFO7ZLxgdafK2Va/view?usp=sharing) describing the rationale behind the evaluation setup.

WeatherBench 2 has been built as an evolving tool for the entire community. For this reason, we welcome any feedback (ideally, submitted as [GitHub issues](https://github.com/google-research/weatherbench2/issues)) or contributions. If you would like you model to be part of WeatherBench, check out [this guide](https://weatherbench2.readthedocs.io/en/latest/submit.html).




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