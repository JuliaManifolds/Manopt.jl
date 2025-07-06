# Extension with Manifolds.jl

Loading `Manifolds.jl` introduces the following additional functions

```@docs
Manopt.max_stepsize(::FixedRankMatrices, ::Any)
Manopt.max_stepsize(::Hyperrectangle, ::Any)
Manopt.max_stepsize(::TangentBundle, ::Any)
mid_point
```

Internally, `Manopt.jl` provides the two additional functions to choose some
Euclidean space when needed as

```@docs
Manopt.Rn
Manopt.Rn_default
```

Together with [`JuMP.jl`](https://jump.dev/), one further extension introduces conversions between typed point and tangent vectors on a manifold to the representation in `JuMP`.