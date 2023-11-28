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
    metrics.SEEPS
    metrics.SpatialSEEPS
```

### Probabilistic Metrics
```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    metrics.EnsembleMetric
    metrics.CRPS
    metrics.SpatialCRPS
    metrics.CRPSSpread
    metrics.SpatialCRPSSpread
    metrics.CRPSSkill
    metrics.SpatialCRPSSkill
    metrics.EnsembleStddev
    metrics.EnsembleVariance
    metrics.SpatialEnsembleVariance
    metrics.EnsembleMeanRMSE
    metrics.EnsembleMeanMSE
    metrics.SpatialEnsembleMeanMSE
    metrics.EnergyScore
    metrics.EnergyScoreSpread
    metrics.EnergyScoreSkill
    metrics.RankHistogram
    metrics.GaussianCRPS
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
    config.Panel
```

## Regions

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    regions.Region
    regions.SliceRegion
    regions.ExtraTropicalRegion
    regions.LandRegion
    regions.CombinedRegion
```

## Derived Variables

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    derived_variables.DerivedVariable
    derived_variables.WindSpeed
    derived_variables.WindDivergence
    derived_variables.WindVorticity
    derived_variables.VerticalVelocity
    derived_variables.EddyKineticEnergy
    derived_variables.GeostrophicWindSpeed
    derived_variables.AgeostrophicWindSpeed
    derived_variables.UComponentOfAgeostrophicWind
    derived_variables.VComponentOfAgeostrophicWind
    derived_variables.LapseRate
    derived_variables.TotalColumnWater
    derived_variables.IntegratedWaterTransport
    derived_variables.RelativeHumidity
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