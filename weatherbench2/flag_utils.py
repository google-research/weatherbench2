# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""WeatherBench2 utilities for working with command line flags."""
import re
from typing import Any, Union

from absl import flags

DimValueType = Union[int, float, str]


def _chunks_string_is_valid(chunks_string: str) -> bool:
  return re.fullmatch(r'(\w+=-?\d+(,\w+=-?\d+)*)?', chunks_string) is not None


def _parse_chunks(chunks_string: str) -> dict[str, int]:
  """Parse a chunks string into a dict."""
  chunks = {}
  if chunks_string:
    for entry in chunks_string.split(','):
      key, value = entry.split('=')
      chunks[key] = int(value)
  return chunks


# TODO(shoyer): write ArgumentParser[dict[str, int]] when ArgumentParser becomes
# Generic in the next absl-py release.
class _ChunksParser(flags.ArgumentParser):
  """Parser for Xarray-Beam chunks flags."""

  syntactic_help: str = (
      'comma separate list of dim=size pairs, e.g., "time=10,longitude=100"'
  )

  def parse(self, argument: str) -> dict[str, int]:
    if not _chunks_string_is_valid(argument):
      raise ValueError(f'invalid chunks string: {argument}')
    return _parse_chunks(argument)

  def flag_type(self) -> str:
    """Returns a string representing the type of the flag."""
    return 'dict[str, int]'


class _DimValuePairSerializer(flags.ArgumentSerializer):
  """Serializer for dim=value pairs."""

  def serialize(self, value: dict[str, int]) -> str:
    return ','.join(f'{k}={v}' for k, v in value.items())


def DEFINE_chunks(  # pylint: disable=invalid-name
    name: str,
    default: str,
    help: str,  # pylint: disable=redefined-builtin
    **kwargs: Any,
):
  """Define a flag for defining Xarray-Beam chunks."""
  parser = _ChunksParser()
  serializer = _DimValuePairSerializer()
  return flags.DEFINE(
      parser, name, default, help, serializer=serializer, **kwargs
  )


# Key/value pairs of the form dimension=integer have the same requirements as
# chunks.
DEFINE_dim_integer_pairs = DEFINE_chunks


class _DimValuePairParser(flags.ArgumentParser):
  """Parser for dim=value pairs."""

  syntactic_help: str = (
      'comma separate list of dim=value pairs, e.g.,'
      '"time=0 days,longitude=100"'
  )

  def parse(self, argument: str) -> dict[str, DimValueType]:
    return _parse_dim_value_pairs(argument)

  def flag_type(self) -> str:
    """Returns a string representing the type of the flag."""
    return 'dict[str, int | float | str]'


def get_dim_value(value_string: str) -> DimValueType:
  """Tries returning int then float, fallback to string."""
  # If typing fails to catch a float being passed, then the first try/except
  # will just return it as an int.
  value_string = str(value_string)
  try:
    return int(value_string)
  except ValueError:
    pass
  try:
    return float(value_string)
  except ValueError:
    pass
  return value_string


def _parse_dim_value_pairs(
    dim_value_string: str,
) -> dict[str, DimValueType]:
  """Parse a chunks string into a dict."""
  pairs = {}
  if dim_value_string:
    for entry in dim_value_string.split(','):
      key, value = entry.split('=')
      pairs[key] = get_dim_value(value)
  return pairs


def DEFINE_dim_value_pairs(  # pylint: disable=invalid-name
    name: str,
    default: str,
    help: str,  # pylint: disable=redefined-builtin
    **kwargs: Any,
):
  """Flag for defining key=value pairs, string key, value a str/int/float."""
  parser = _DimValuePairParser()
  serializer = _DimValuePairSerializer()
  return flags.DEFINE(
      parser, name, default, help, serializer=serializer, **kwargs
  )
