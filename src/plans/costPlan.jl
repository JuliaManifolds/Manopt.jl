#
# A very simple problem for all solvers that just require a cost
#

export CostProblem
export NelderMeadOptions

@doc doc"""
    CostProblem <: Problem

speficy a problem for solvers just based on cost functions, i.e.
gradient free ones.

# Fields

* `M`            – a manifold $\mathcal M$
* `costFunction` – a function $F\colon\mathcal M\to\mathbb R$ to minimize

# See also
[`NelderMead`](@ref)
"""
struct CostProblem{mT <: Manifold} <: Problem
    M::mT
    costFunction::Function
end

@doc doc"""
    NelderMeadOptions <: Options

Describes all parameters and the state of a Nealer-Mead heuristic based
optimization algorithm.

# Fields

The naming of these parameters follows the [Wikipedia article](https://en.wikipedia.org/wiki/Nelder–Mead_method)
of the Euclidean case. The default is given in brackets, the required value range
after the description

* `population` – an `Array{`[`MPoint`](@ref)`,1}` of $n+1$ points $x_i$, $i=1,\ldots,n+1$, where $n$ is the
  dimension of the manifold.
* `stoppingCriterion` – ([`stopAfterIteration`](@ref)`(2000)`) a [`StoppingCriterion`](@ref)
* `retraction` – (`exp`) the rectraction to use
* `α` – (`1.`) reflection parameter ($\alpha > 0$)
* `γ` – (`2.`) expansion parameter ($\gamma>0$)
* `ρ` – (`1/2`) contraction parameter, $0 < \rho \leq \frac{1}{2}$,
* `σ` – (`1/2`) shrink coefficient, $0 < \sigma \leq 1$
* `x` – (`p[1]`) - a field to collect the current best value

# Constructors

    NelderMead(M,stop, retr; α=1. , γ=2., ρ=1/2, σ=1/2)

construct a Nelder-Mead Option with a set of `dimension(M)+1` random points.

    NelderMead(p, stop retr; α=1. , γ=2., ρ=1/2, σ=1/2)

construct a Nelder-Mead Option with a set `p` of [`MPoint`](@ref)s
"""
mutable struct NelderMeadOptions{P <: MPoint} <: Options
    population::Array{P,1}
    stop::StoppingCriterion
    α::Real
    γ::Real
    ρ::Real
    σ::Real
    x::P
    costs::Array{Float64,1}
    NelderMeadOptions(M::mT,
        stop::StoppingCriterion = stopAfterIteration(2000);
        α = 1., γ = 2., ρ=1/2, σ = 1/2
    ) where {mT <: Manifold } = 
        new{typeof(randomMPoint(M))}(
            [randomMPoint(M) for i=1:(manifoldDimension(M)+1) ],
            stop, α, γ, ρ, σ, randomMPoint(M),[] )
    NelderMeadOptions(p::Array{P,1},
        stop::StoppingCriterion = stopAfterIteration(2000);
        α = 1., γ = 2., ρ=1/2, σ = 1/2
    ) where {P <: MPoint} = new{P}(p, stop, α, γ, ρ, σ, p[1],[] )
end 