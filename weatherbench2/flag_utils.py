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

from absl import flags


def _chunks_string_validator(chunks_string: str) -> bool:
  return re.fullmatch(r'(\w+=-?\d+(,\w+=-?\d+)*)?', chunks_string) is not None


def DEFINE_chunks(name, default, help):  # pylint: disable=invalid-name,redefined-builtin
  """Define a flag for defining Xarray-Beam chunks."""
  flag_values = flags.DEFINE_string(name, default, help=help)
  flags.register_validator(flag_values, _chunks_string_validator)
  return flag_values


def parse_chunks(chunks_string: str) -> dict[str, int]:
  """Parse a chunks string into a dict."""
  chunks = {}
  for entry in chunks_string.split(','):
    key, value = entry.split('=')
    chunks[key] = int(value)
  return chunks
