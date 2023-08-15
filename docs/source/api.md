(api)=
# API docs

```{eval-rst}
.. currentmodule:: weatherbench2
```

## Metrics
### Deterministic Metrics
```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    metrics.Metric
    metrics.RMSE
    metrics.WindVectorRMSE
    metrics.MSE
    metrics.SpatialMSE
    metrics.MAE
    metrics.SpatialMAE
    metrics.Bias
    metrics.SpatialBias
    metrics.ACC
```

### Probabilistic Metrics
```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    metrics.EnsembleMetric
    metrics.CRPS
    metrics.CRPSSpread
    metrics.CRPSSkill
    metrics.EnsembleStddev
    metrics.EnsembleMeanRMSE
    metrics.EnergyScore
    metrics.EnergyScoreSpread
    metrics.EnergyScoreSkill
```

## Config

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    config.Selection
    config.Paths
    config.Data
    config.Eval
    config.Viz
    config.PanelConfig
```

## Regions

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    regions.Region
    regions.SliceRegion
    regions.ExtraTropicalRegion
    regions.LandRegion
```

## Derived Variables

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    derived_variables.DerivedVariable
    derived_variables.WindSpeed
    derived_variables.PrecipitationAccumulation
    derived_variables.AggregatePrecipitationAccumulation
    derived_variables.ZonalEnergySpectrum
```

## Evaluation

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    evaluation.open_source_files
    evaluation.open_forecast_and_truth_datasets
    evaluation.evaluate_in_memory
    evaluation.evaluate_with_beam
```