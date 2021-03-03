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
struct CostProblem{T,mT<:Manifold,Tcost} <: Problem{T}
    M::mT
    cost::Tcost
end
function CostProblem(
    M::mT, cost::T; evaluation::AbstractEvaluationType=AllocatingEvaluation()
) where {mT<:Manifold,T}
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

    NelderMead(M,stop, retr; α=1. , γ=2., ρ=1/2, σ=1/2)

construct a Nelder-Mead Option with a set of `dimension(M)+1` random points.

    NelderMead(p, stop retr; α=1. , γ=2., ρ=1/2, σ=1/2)

construct a Nelder-Mead Option with a set `p` of points
"""
mutable struct NelderMeadOptions{
    T,
    Tα<:Real,
    Tγ<:Real,
    Tρ<:Real,
    Tσ<:Real,
    TR<:AbstractRetractionMethod,
    TI<:AbstractInverseRetractionMethod,
} <: Options
    population::Vector{T}
    stop::StoppingCriterion
    α::Tα
    γ::Tγ
    ρ::Tρ
    σ::Tσ
    x::T
    costs::Vector{Float64}
    retraction_method::TR
    inverse_retraction_method::TI
    function NelderMeadOptions(
        M::MT;
        stop::StoppingCriterion=StopAfterIteration(2000),
        α=1.0,
        γ=2.0,
        ρ=1 / 2,
        σ=1 / 2,
        retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
        inverse_retraction_method::AbstractInverseRetractionMethod=LogarithmicInverseRetraction(),
    ) where {MT<:Manifold}
        p = [random_point(M) for i in 1:(manifold_dimension(M) + 1)]
        return new{
            eltype(p),
            typeof(α),
            typeof(γ),
            typeof(ρ),
            typeof(σ),
            typeof(retraction_method),
            typeof(inverse_retraction_method),
        }(
            p, stop, α, γ, ρ, σ, p[1], [], retraction_method, inverse_retraction_method
        )
    end
    function NelderMeadOptions(
        population::Vector{T},
        stop::StoppingCriterion=StopAfterIteration(2000);
        α=1.0,
        γ=2.0,
        ρ=1 / 2,
        σ=1 / 2,
        retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
        inverse_retraction_method::AbstractInverseRetractionMethod=LogarithmicInverseRetraction(),
    ) where {T}
        return new{
            T,
            typeof(α),
            typeof(γ),
            typeof(ρ),
            typeof(σ),
            typeof(retraction_method),
            typeof(inverse_retraction_method),
        }(
            population,
            stop,
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
get_solver_result(o::NelderMeadOptions) = o.x
