from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import pandas as pd
import xarray as xr
import xarray_beam

from . import compute_quantiles


class ComputeQuantileTest(parameterized.TestCase):

  def _dict_to_str(self, a_dict):
    return ','.join(f'{k}={v}' for k, v in a_dict.items())

  @parameterized.named_parameters(
      dict(
          testcase_name='NoChunks',
      ),
      dict(
          testcase_name='SpecifyInputWorkingChunks1',
          input_chunks={'time': 2},
          working_chunks={'time': 1, 'timedelta': 1},
      ),
      dict(
          testcase_name='SpecifyInputWorkingChunks2',
          input_chunks={'time': 2},
          working_chunks={'time': 1, 'timedelta': -1},
      ),
      dict(
          testcase_name='SpecifyInputOutputAndWorkingChunks1',
          input_chunks={'time': 2},
          output_chunks={'time': 2, 'timedelta': 3},
          working_chunks={'time': -1},
      ),
      dict(
          testcase_name='SpecifyInputOutputAndWorkingChunks2',
          input_chunks={'timedelta': 2},
          output_chunks={'time': 2},
          working_chunks={'timedelta': 1},
      ),
      dict(
          testcase_name='SpecifyInputOutputChunks1',
          input_chunks={'timedelta': 2},
          output_chunks={'time': 2},
      ),
      dict(
          testcase_name='SpecifyInputOutputChunks2_NameSuffix',
          input_chunks={'timedelta': 2},
          output_chunks={'time': -1},
          name_suffix='_quantile',
      ),
  )
  def test_basic_dataset(
      self,
      input_chunks=None,
      output_chunks=None,
      working_chunks=None,
      name_suffix='',
  ):
    input_chunks = input_chunks or {}
    output_chunks = output_chunks or {}
    working_chunks = working_chunks or {}
    times = pd.DatetimeIndex(
        [
            '2023-01-01',
            '2023-01-02',
            '2023-01-03',
            '2023-01-04',
        ]
    )  # fmt: skip
    lats = np.arange(50)
    timedeltas = np.arange(6)

    precip = np.random.RandomState(
        802701 + len(input_chunks) + len(output_chunks) + len(working_chunks)
    ).rand(4, 50, 6)

    quantiles = [0.2, 0.8]

    input_ds = xr.Dataset(
        {
            'precip': xr.DataArray(
                precip,
                coords=[times, lats, timedeltas],
                dims=['time', 'lat', 'timedelta'],
            ),
            'should_drop': xr.DataArray(
                precip * 2,
                coords=[times, lats, timedeltas],
                dims=['time', 'lat', 'timedelta'],
            ),
        }
    )  # fmt: skip

    input_path = self.create_tempdir('source').full_path
    input_ds.chunk(input_chunks).to_zarr(input_path)

    # Get modified output
    output_path = self.create_tempdir('output').full_path
    with flagsaver.as_parsed(
        input_path=input_path,
        output_path=output_path,
        working_chunks=self._dict_to_str(working_chunks),
        output_chunks=self._dict_to_str(output_chunks),
        dim='lat',
        variables='precip',
        time_start='2023-01-01',
        time_stop='2023-01-03',
        quantiles=','.join(str(q) for q in quantiles),
        name_suffix=name_suffix,
        runner='DirectRunner',
    ):
      compute_quantiles.main([])
    output, actual_output_chunks = xarray_beam.open_zarr(output_path)

    # Output only has the "preserved dims" + quantile
    self.assertCountEqual(output.dims, ['time', 'timedelta', 'quantile'])

    expected_output_chunks = {'quantile': -1}
    for k in output.dims:
      if k in output_chunks and output_chunks[k] == -1:
        expected_output_chunks[k] = output.sizes[k]
      elif k in output_chunks:
        expected_output_chunks[k] = output_chunks[k]
      else:
        expected_output_chunks[k] = min(
            input_chunks.get(k, np.inf), output.sizes[k]
        )
    self.assertEqual(expected_output_chunks, actual_output_chunks)

    xr.testing.assert_equal(
        input_ds[['precip']]
        .isel(time=slice(3))
        .quantile(quantiles, dim='lat')
        .rename_vars({'precip': 'precip' + name_suffix}),
        output,
    )


if __name__ == '__main__':
  absltest.main()
