<!-- # WeatherBench 2: A benchmark for the next generation of data-driven weather forecasts -->

![image](_static/wb2-logo-wide.png)

## Why WeatherBench?

WeatherBench 2 is a framework for evaluating and comparing data-driven and traditional numerical weather forecasting models. WeatherBench consists of:
- Publicly available, cloud-optimized ground truth and baseline datasets. For a complete list, see [this page](data-guide). 
- Open-source evaluation code. See this [quick-start](evaluation) to explore the basic functionality or the [API docs](api) for more detail. Since high-resolution forecast files can be large, the WeatherBench 2 code was written with scalability in mind. See the [command-line scripts](cli) based on [Xarray-Beam](https://xarray-beam.readthedocs.io/en/latest/) and [this guide](dataflow) for running the scripts on GCP using [DataFlow](https://cloud.google.com/dataflow).
- A [website](https://sites.research.google/weatherbench) displaying up-to-date scores of many of the state-of-the-art data-driven and physical approaches.
- A [paper](https://arxiv.org/abs/2308.15560) describing the rationale behind the evaluation setup.

WeatherBench 2 has been built as an evolving tool for the entire community. For this reason, we welcome any feedback (ideally, submitted as [GitHub issues](https://github.com/google-research/weatherbench2/issues)) or contributions. If you would like your model to be part of WeatherBench, check out [this guide](submit).



## Installation

To get started using the WeatherBench 2 code, first clone the GitHub repository:

```
git clone git@github.com:google-research/weatherbench2.git
```

Then, install using pip

```
cd weatherbench2
pip install .
```

If you would like to actively develop the code, install using

```
pip install -e .
```

If you plan to launch jobs on GCP, also install the GCP requirements:

```
pip install .[gcp]
```

## Contents

```{toctree}
:maxdepth: 1
evaluation.ipynb
data-guide.ipynb
command-line-scripts.md
beam-in-the-cloud.md
official-evaluation.md
submit.md
init-vs-valid-time.md
api.md
```
