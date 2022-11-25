@doc raw"""
    CostProblem{T, Manifold, TCost} <: Problem{T}

speficy a problem for solvers just based on cost functions, i.e.
gradient free ones.
# Fields

* `M`            – a manifold ``\mathcal M``
* `cost` – a function ``F: \mathcal M → ℝ`` to minimize

# Constructors

    CostProblem(M, cost; evaluation=AllocatingEvaluation())

Generate a problem. While this Problem does not have any allocating functions,
the type `T` can be set for consistency reasons with other problems.

# See also
[`NelderMead`](@ref)
"""
struct CostProblem{T,mT<:AbstractManifold,Tcost} <: Problem{T}
    M::mT
    cost::Tcost
end
function CostProblem(
    M::mT, cost::T; evaluation::AbstractEvaluationType=AllocatingEvaluation()
) where {mT<:AbstractManifold,T}
    return CostProblem{typeof(evaluation),mT,T}(M, cost)
end
@doc raw"""
    NelderMeadOptions <: Options

Describes all parameters and the state of a Nealer-Mead heuristic based
optimization algorithm.

# Fields

The naming of these parameters follows the [Wikipedia article](https://en.wikipedia.org/wiki/Nelder–Mead_method)
of the Euclidean case. The default is given in brackets, the required value range
after the description

* `population` – an `Array{`point`,1}` of ``n+1`` points ``x_i``, ``i=1,…,n+1``, where ``n`` is the
  dimension of the manifold.
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(2000)`) a [`StoppingCriterion`](@ref)
* `α` – (`1.`) reflection parameter (``α > 0``)
* `γ` – (`2.`) expansion parameter (``γ > 0``)
* `ρ` – (`1/2`) contraction parameter, ``0 < ρ ≤ \frac{1}{2}``,
* `σ` – (`1/2`) shrink coefficient, ``0 < σ ≤ 1``
* `x` – (`p[1]`) - a field to collect the current best value
* `retraction_method` – `ExponentialRetraction()` the rectraction to use, defaults to
  the exponential map
* `inverse_retraction_method` - `LogarithmicInverseRetraction` an `inverse_retraction(M,x,y)` to use.

# Constructors

    NelderMead(M[, population]; kwargs)

construct a Nelder-Mead Option with a default popultion (if not provided) of set of `dimension(M)+1` random points.

In the constructor all fields (besides the population) are keyword arguments.
"""
mutable struct NelderMeadOptions{
    T,
    S<:StoppingCriterion,
    Tα<:Real,
    Tγ<:Real,
    Tρ<:Real,
    Tσ<:Real,
    TR<:AbstractRetractionMethod,
    TI<:AbstractInverseRetractionMethod,
} <: Options
    population::Vector{T}
    stop::S
    α::Tα
    γ::Tγ
    ρ::Tρ
    σ::Tσ
    x::T
    costs::Vector{Float64}
    retraction_method::TR
    inverse_retraction_method::TI
    function NelderMeadOptions(
        M::AbstractManifold,
        population::Vector{T};
        stopping_criterion::StoppingCriterion=StopAfterIteration(2000),
        α=1.0,
        γ=2.0,
        ρ=1 / 2,
        σ=1 / 2,
        retraction_method::AbstractRetractionMethod=default_retraction_method(M),
        inverse_retraction_method::AbstractInverseRetractionMethod=default_inverse_retraction_method(
            M
        ),
    ) where {T}
        return new{
            T,
            typeof(stopping_criterion),
            typeof(α),
            typeof(γ),
            typeof(ρ),
            typeof(σ),
            typeof(retraction_method),
            typeof(inverse_retraction_method),
        }(
            population,
            stopping_criterion,
            α,
            γ,
            ρ,
            σ,
            population[1],
            [],
            retraction_method,
            inverse_retraction_method,
        )
    end
end
function NelderMeadOptions(M::AbstractManifold; kwargs...)
    p = [random_point(M) for i in 1:(manifold_dimension(M) + 1)]
    return NelderMeadOptions(M, p; kwargs...)
end
@deprecate NelderMeadOptions(
    population,
    retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
    inverse_retraction_method::AbstractInverseRetractionMethod=LogarithmicInverseRetraction(),
    kwargs...,
) NelderMeadOptions(
    DefaultManifold(2),
    population,
    retraction_method=retraction_method,
    inverse_retraction_method=inverse_retraction_method,
    kwargs...,
)
get_iterate(O::NelderMeadOptions) = O.x
function set_iterate!(O::NelderMeadOptions, p)
    O.x = p
    return O
end
