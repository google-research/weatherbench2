# How to Contribute

We would love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google/conduct/).

## Contribution process

### Code Reviews

All submissions, including submissions by project members, require review. We 
use [GitHub pull requests](https://docs.github.com/articles/about-pull-requests)
for this purpose. If possible, consider reviewing [Google's Code Review Guide](https://google.github.io/eng-practices/review/).

### Development Setup

After cloning the project, we recommend creating a local Python3.11 environment
for development (try [Miniconda](https://docs.conda.io/en/latest/miniconda.html)).

To locally install the project for development, please run:
```shell
pip install -e ".[tests]"
```
> Note: The `-e` flag installs sources in "editable" mode. Except when one needs
> to add a new dependency, you should only have to do this once.

### Local Testing
To locally test changes to the `weatherbench2` package, run:

```shell
pytest weatherbench2/
```

To test all changes to all scripts, run:
```shell
for test in scripts/*_test.py; do pytest $test; done
```

Or, to test a single script, simply run:
```shell
pytest scripts/<script_test.py>
```

In addition, we require that all<sup>*</sup> code adhere to the [Google Style Python Guide](https://google.github.io/styleguide/pyguide.html).
To assist with this, we've configured the project with the `pyink` and `isort` 
formatters. To format your change before patch, please run:

```shell
pyink .
isort .
```

_<sup>*</sup>We do express some project-specific opinions._

