# Interior Point Newton method

```@meta
CurrentModule = Manopt
```

```@docs
interior_point_Newton
```

## State

```@docs
InteriorPointNewtonState
```

## Subproblem functions

```@docs
CondensedKKTVectorField
CondensedKKTVectorFieldJacobian
KKTVectorField
KKTVectorFieldJacobian
KKTVectorFieldAdjointJacobian
KKTVectorFieldNormSq
KKTVectorFieldNormSqGradient
```

## Helpers

```@docs
InteriorPointCentralityCondition
Manopt.calculate_σ
```

## Additional stopping criteria

```@docs
StopWhenKKTResidualLess
```

## References

```@bibliography
Pages = ["interior_point_Newton.md"]
Canonical=false
```
