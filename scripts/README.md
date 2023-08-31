# Weatherbench 2 scripts

Here are core routines for producing Weatherbench 2 datasets and evaluations. To
see how to invoke a given pipeline (with examples), please run the commensurate
script with the `--help` flag. For example:

```shell
python3 evaluate.py --help
```

All of our pipelines make use of [Apache Beam](https://beam.apache.org/). In our
examples, we illustrate how to run these
on [Google Cloud Dataflow](https://cloud.google.com/dataflow/docs/guides/deploying-a-pipeline#python).
If Dataflow is not an available option, each pipeline can be run with the Apache
Spark runner, which is available on every major commercial cloud (and can be
self-hosted). For more on the Spark runner,
please [consult this documentation](https://beam.apache.org/documentation/runners/spark/).

For more information on how to run each Beam pipeline, please consult [this
documentation](https://weatherbench2.readthedocs.io/en/latest/beam-in-the-cloud.html).
