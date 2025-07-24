@doc raw"""
    FrankWolfeCost{P,T}

A structure to represent the oracle sub problem in the [`Frank_Wolfe_method`](@ref).
The cost function reads

```math
F(q) = ⟨X, \log_p q⟩
```

The values `p` and `X` are stored within this functor and should be references to the
iterate and gradient from within [`FrankWolfeState`](@ref).
"""
mutable struct FrankWolfeCost{P, T}
    p::P
    X::T
end
function (FWO::FrankWolfeCost)(M, q)
    return real(inner(M, FWO.p, FWO.X, log(M, FWO.p, q)))
end

@doc raw"""
    FrankWolfeGradient{P,T}

A structure to represent the gradient of the oracle sub problem in the [`Frank_Wolfe_method`](@ref),
that is for a given point `p` and a tangent vector `X` the function reads

```math
F(q) = ⟨X, \log_p q⟩
```

Its gradient can be computed easily using `adjoint_differential_log_argument`.

The values `p` and `X` are stored within this functor and should be references to the
iterate and gradient from within [`FrankWolfeState`](@ref).
"""
mutable struct FrankWolfeGradient{P, T}
    p::P
    X::T
end
function (FWG::FrankWolfeGradient)(M, Y, q)
    return adjoint_differential_log_argument!(M, Y, FWG.p, q, FWG.X)
end
function (FWG::FrankWolfeGradient)(M, q)
    return adjoint_differential_log_argument(M, FWG.p, q, FWG.X)
end
