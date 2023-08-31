(dataflow)=
# Distributed computing using Beam on GCP

Since the input datasets can be large, distributed computing is key for efficient evaluation. Here is a guide for running the [command line scripts](cli) on [Google Cloud Dataflow](https://cloud.google.com/dataflow).

## Local execution

To run the scripts locally, chose the `DirectRunner` as your Beam runner. There, you will have an option for how many local workers to use, e.g.:

```bash
python evaluate.py \
  --forecast_path=gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_with_poles_conservative.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_with_poles_conservative.zarr \
  --output_dir=./ \
  --output_file_prefix=hres_vs_era_2020_ \
  --by_init=True \
  --input_chunks=init_time=1 \
  --eval_configs=deterministic \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential \
  --use_beam=True \
  --beam_runner=DirectRunner \
  -- \
  --direct_num_workers 2
```

For a full list of how to configure the direct runner, please review [this page](https://beam.apache.org/documentation/runners/direct/).

## Dataflow execution

Make sure to install the package including the GCP requirements: `pip install -e .[gcp]`

To run on Google Cloud Dataflow, use the `DataflowRunner`. Additionally, a few parameters need to be specified.

* `--runner`: The `PipelineRunner` to use. This field can be either `DirectRunner` or `DataflowRunner`.
  Default: `DirectRunner` (local mode)
* `--project`: The project ID for your Google Cloud Project. This is required if you want to run your pipeline using the
  Dataflow managed service (i.e. `DataflowRunner`).
* `--temp_location`: Cloud Storage path for temporary files. Must be a valid Cloud Storage URL, beginning with `gs://`.
* `--region`: Specifies a regional endpoint for deploying your Dataflow jobs. Default: `us-central1`.
* `--job_name`: The name of the Dataflow job being executed as it appears in Dataflow's jobs list and job details.
* `--setup_file`: To make sure all the required packages are installed on the workers, pass the `setup.py` file to GCP.

Example run:

```bash
python scripts/evaluate.py \
  --forecast_path=gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_with_poles_conservative.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_with_poles_conservative.zarr \
  --output_dir=gs://$BUCKET \
  --output_file_prefix=hres_vs_era_2020_ \
  --by_init=True \
  --input_chunks=init_time=10 \
  --eval_configs=deterministic \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential \
  --use_beam=True \
  --runner=DataflowRunner \
  -- \
  --project $PROJECT \
  --region $REGION \
  --temp_location gs://$BUCKET/tmp/
  --setup_file=./setup.py
```

For a full list of how to configure the Dataflow pipeline, please review
[this table](https://cloud.google.com/dataflow/docs/reference/pipeline-options).

### Monitoring

When running Dataflow, you
can [monitor jobs through UI](https://cloud.google.com/dataflow/docs/guides/using-monitoring-intf),
or [via Dataflow's CLI commands](https://cloud.google.com/dataflow/docs/guides/using-command-line-intf):

For example, to see all outstanding Dataflow jobs, simply run:

```shell
gcloud dataflow jobs list
```

To describe stats about a particular Dataflow job, run:

```shell
gcloud dataflow jobs describe $JOBID
```

In addition, Dataflow provides a series
of [Beta CLI commands](https://cloud.google.com/sdk/gcloud/reference/beta/dataflow).

These can be used to keep track of job metrics, like so:

```shell
JOBID=<enter job id here>
gcloud beta dataflow metrics list $JOBID --source=user
```

You can even [view logs via the beta commands](https://cloud.google.com/sdk/gcloud/reference/beta/dataflow/logs/list):

```shell
gcloud beta dataflow logs list $JOBID
```

## Spark execution

Users may want to run Weatherbench 2 scripts on other clouds beyond GCP. To best
accomplish this, we recommend using a managed or self-hosted Apache Spark
runner: [beam.apache.org/documentation/runners/spark/](https://beam.apache.org/documentation/runners/spark/).

Once you have a Spark cluster set up (specifically, running on the JVM), please
follow the above documentation to configure the `PortableRunner` on the cluster.

