# [Vectorial objectives](@id vectorial_objectives)

```@meta
CurrentModule = Manopt
```

```@docs
Manopt.AbstractVectorFunction
Manopt.AbstractVectorGradientFunction
Manopt.VectorGradientFunction
Manopt.VectorHessianFunction
```


```@docs
Manopt.AbstractVectorialType
Manopt.CoefficientVectorialType
Manopt.ComponentVectorialType
Manopt.FunctionVectorialType
```

## Access functions

```@docs
Manopt.get_jacobian
Manopt.get_jacobian!
Manopt.get_adjoint_jacobian
Manopt.get_adjoint_jacobian!
Manopt.get_value
Manopt.get_value_function
Base.length(::VectorGradientFunction)
```

## Internal functions

```@docs
Manopt._to_iterable_indices
Manopt._change_basis!
Manopt.get_basis
Manopt.get_range
```
