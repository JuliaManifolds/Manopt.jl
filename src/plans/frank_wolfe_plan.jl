@doc raw"""
    FrankWolfeOptions{Type} <: Options

A struct to store the current state of the [`Frank_Wolfe_algorithm`](@ref)

It comes in two forms, depending on the realisation of the `subproblem`.

# Fields

* `p` – the current iterate, i.e. a point on the manifold
* `X` – the current gradient ``\operatorname{grad} f(p)``, i.e. a tangent vector to `p`.
* `subtask` – a type representing the subtask.
* `stop` – ([`StopAfterIteration`](@ref)`(200) | `[`StopWhenGradientNormLess`](@ref)`(1.0e-6)`) a [`StoppingCriterion`]
* `stepsize` _ ([`DecreasingStepsize`](@ref)`(; length=2.0, shift=2))
For the subtask, we need a method to solve

```math
    \operatorname*{argmin}_{q∈\mathcal M} ⟨X, \log_p q⟩,\qquad where X=\operatorname{grad} f(p)
```

where currently two variants are supported
1. `subtask(M, q, X, p)` is a mutating function, i.e. we have a closed form solution of the
  optimization problem given `M`, `X` and `p` which is computed in place of `q`, which even
  works correctly, if we pass the same memory to `p` and `q`.
2. `subtask::Tuple{<:Problem,<:Options}` specifies a plan to solve the sub task with a subsolver,
  i.e. the cost within `subtask[1]` is a [`FrankWolfeOracleCost``](@ref) using `p`and `X`
  internally, i.e. the cost is updated as soon as they are updated.
  Similarly for gradient based functions using the [`FrankWolfeOracleGradient`](@ref).

# Constructor
"""
mutable struct FrankWolfeOptions{
    S,
    P,
    T,
    TStep<:Stepsize,
    TStop<:StoppingCriterion,
    TM<:AbstractRetractionMethod,
    ITM<:AbstractInverseRetractionMethod,
}
    p::P
    X::T
    subtask::Sub
    stop::TStop
    stepsize::TStep
    retraction_method::TM,
    inverse_retraction_method::ITM,
    function FrankWolfeOptions(
        M::AbstractManifold,
        p::P,
        subtask::S;
        initial_vector::T=zero_vector(M, p),
        stopping_criterion::TStop=StopAfterIteration(200) |
                                  StopWhenGradientNormLess(1.0e-6),
        stepsize::TStep=DecreasingStepsize(; length=2.0, shift=2),
        retraction_method::TM=default_retraction_method(M),
        inverse_retraction_method::ITM=default_inverse_retraction_method(M),
    ) where {P,S,T,TStop<:StoppingCriterion,TStep<:Stepsize}
        return new{S,T,P,TStep,TStop,TM,ITM}(
            p,
            initial_vector,
            subtask,
            stopping_criterion,
            stepsize,
            retraction_method,
            inverse_retraction_method,
        )
    end
end

mutable struct FrankWolfeOracleCost{P,T}
    p::P
    X::T
end
function (FWO::FrankWolfeOracleCost)(M, q)
    return inner(M, FWO.p, FWO.X, log(M, FWO.p, q))
end

mutable struct FrankWolfeOracleGradient{P,T}
    p::P
    X::T
end
function (FWG::FrankWolfeOracleGradient)(M, Y, q)
    return adjoint_differential_log_argument!(M, Y, FWG.p, q, FWG.X)
end
function (FWG::FrankWolfeOracleGradient)(M, q)
    return adjoint_differential_log_argument(M, FWG.p, q, FWG.X)
end
