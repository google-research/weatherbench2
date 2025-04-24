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
"""Setup Weatherbench2."""
import setuptools

base_requires = [
    'apache_beam>=2.31.0',
    'cftime>=1.6.2',
    'jax[cpu]',
    'numpy>=2.1.3',
    'pandas>=2.2.3',
    'scipy',
    'scikit-learn',
    'xarray>=2024.11.0',
    'xarray-beam',
    'zarr',
]

docs_requires = [
    'myst-nb',
    'myst-parser',
    'sphinx',
    'sphinx_rtd_theme',
    # 'sphinx-book-theme',
    'scipy',
]
tests_requires = [
    'absl-py',
    'pytest',
    'pyink',
    # work around https://github.com/zarr-developers/zarr-python/issues/2963
    'numcodecs<0.16.0',
]

gcp_requires = [
    'apache_beam[gcp]>=2.31.0',
    'gcsfs',
]

setuptools.setup(
    name='weatherbench2',
    version='0.2.1',
    license='Apache 2.0',
    author='Google LLC',
    author_email='noreply@google.com',
    install_requires=base_requires,
    extras_require={
        'tests': tests_requires,
        'docs': docs_requires,
        'gcp': gcp_requires,
        'viz': ['matplotlib'],
    },
    url='https://github.com/google-research/weatherbench2',
    packages=setuptools.find_packages(exclude=['notebooks', 'scripts']),
    python_requires='>=3',
)
