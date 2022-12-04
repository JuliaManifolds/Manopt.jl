@doc raw"""
    FrankWolfeOptions <: Options

A struct to store the current state of the [`Frank_Wolfe_method`](@ref)

It comes in two forms, depending on the realisation of the `subproblem`.

# Fields

* `p` – the current iterate, i.e. a point on the manifold
* `X` – the current gradient ``\operatorname{grad} F(p)``, i.e. a tangent vector to `p`.
* `evalulation` [`AllocatingEvaluation`](@ref) specify the  type if it is a function.
* `inverse_retraction_method` – (`default_inverse_retraction_method(M)`) an inverse retraction method to use within Frank Wolfe.
* `subtask` – a type representing the subtask (see below).
* `stop` – ([`StopAfterIteration`](@ref)`(200) | `[`StopWhenGradientNormLess`](@ref)`(1.0e-6)`) a [`StoppingCriterion`](@ref)
* `stepsize` - ([`DecreasingStepsize`](@ref)`(; length=2.0, shift=2)`) ``s_k`` which by default is set to ``s_k = \frac{2}{k+2}``.
* `retraction_method` – (`default_retraction_method(M)`) a retraction to use within Frank-Wolfe

For the subtask, we need a method to solve

```math
    \operatorname*{argmin}_{q∈\mathcal M} ⟨X, \log_p q⟩,\qquad \text{ where }X=\operatorname{grad} f(p)
```

where currently two variants are supported
1. `subtask(M, q, X, p)` is a mutating function, i.e. we have a closed form solution of the
   optimization problem given `M`, `X` and `p` which is computed in place of `q`, which even
   works correctly, if we pass the same memory to `p` and `q`.
2. `subtask::Tuple{<:Problem,<:Options}` specifies a plan to solve the subtask with a subsolver,
   i.e. the cost within `subtask[1]` is a [`FrankWolfeCost`](@ref) using references to `p`and `X`,
   that is to the current iterate and gradient internally.
   Similarly for gradient based functions using the [`FrankWolfeGradient`](@ref).

# Constructor

    FrankWolfeOptions(M, p, X, subtask)

where the remaining fields from above are keyword arguments with their defaults already given in brackets.
"""
mutable struct FrankWolfeOptions{
    S,
    P,
    T,
    TStep<:Stepsize,
    TStop<:StoppingCriterion,
    TM<:AbstractRetractionMethod,
    ITM<:AbstractInverseRetractionMethod,
} <: AbstractGradientOptions
    p::P
    X::T
    subtask::S
    stop::TStop
    stepsize::TStep
    retraction_method::TM
    inverse_retraction_method::ITM
    function FrankWolfeOptions(
        M::AbstractManifold,
        p::P,
        subtask::S;
        evaluation=AllocatingEvaluation(),
        initial_vector::T=zero_vector(M, p),
        stopping_criterion::TStop=StopAfterIteration(200) |
                                  StopWhenGradientNormLess(1.0e-6),
        stepsize::TStep=DecreasingStepsize(; length=2.0, shift=2),
        retraction_method::TM=default_retraction_method(M),
        inverse_retraction_method::ITM=default_inverse_retraction_method(M),
    ) where {
        P,
        S<:Function,
        T,
        TStop<:StoppingCriterion,
        TStep<:Stepsize,
        TM<:AbstractRetractionMethod,
        ITM<:AbstractInverseRetractionMethod,
    }
        return new{Tuple{S,typeof(evaluation)},P,T,TStep,TStop,TM,ITM}(
            p,
            initial_vector,
            (subtask, evaluation),
            stopping_criterion,
            stepsize,
            retraction_method,
            inverse_retraction_method,
        )
    end
    function FrankWolfeOptions(
        M::AbstractManifold,
        p::P,
        subtask::S;
        evaluation=AllocatingEvaluation(),
        initial_vector::T=zero_vector(M, p),
        stopping_criterion::TStop=StopAfterIteration(200) |
                                  StopWhenGradientNormLess(1.0e-6),
        stepsize::TStep=DecreasingStepsize(; length=2.0, shift=2),
        retraction_method::TM=default_retraction_method(M),
        inverse_retraction_method::ITM=default_inverse_retraction_method(M),
    ) where {
        P,
        S<:Tuple{<:Problem,<:Options},
        T,
        TStop<:StoppingCriterion,
        TStep<:Stepsize,
        TM<:AbstractRetractionMethod,
        ITM<:AbstractInverseRetractionMethod,
    }
        return new{S,P,T,TStep,TStop,TM,ITM}(
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
get_iterate(O::FrankWolfeOptions) = O.p
function set_iterate!(O::FrankWolfeOptions, p)
    O.p = p
    return O
end
get_gradient(O::FrankWolfeOptions) = O.X

@doc raw"""
    FrankWolfeCost{P,T}

A structure to represent the oracle sub problem in the [`Frank_Wolfe_method`](@ref).
The cost function reads

```math
F(q) = ⟨X, \log_p q⟩
```

The values `p`and `X` are stored within this functor and hsould be references to the
iterate and gradient from within [`FrankWolfeOptions`](@ref).
"""
mutable struct FrankWolfeCost{P,T}
    p::P
    X::T
end
function (FWO::FrankWolfeCost)(M, q)
    return inner(M, FWO.p, FWO.X, log(M, FWO.p, q))
end

@doc raw"""
    FrankWolfeGradient{P,T}

A structure to represent the gradeint of the oracle sub problem in the [`Frank_Wolfe_method`](@ref),
that is for a given point `p` and a tangent vector `X` we have

```math
F(q) = ⟨X, \log_p q⟩
```

Its gradient can be computed easily using [`adjoint_differential_log_argument`](@ref).

The values `p`and `X` are stored within this functor and hsould be references to the
iterate and gradient from within [`FrankWolfeOptions`](@ref).
"""
mutable struct FrankWolfeGradient{P,T}
    p::P
    X::T
end
function (FWG::FrankWolfeGradient)(M, Y, q)
    return adjoint_differential_log_argument!(M, Y, FWG.p, q, FWG.X)
end
function (FWG::FrankWolfeGradient)(M, q)
    return adjoint_differential_log_argument(M, FWG.p, q, FWG.X)
end
