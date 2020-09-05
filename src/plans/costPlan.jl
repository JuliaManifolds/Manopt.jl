@doc raw"""
    CostProblem <: Problem

speficy a problem for solvers just based on cost functions, i.e.
gradient free ones.

# Fields

* `M`            – a manifold $\mathcal M$
* `cost` – a function $F\colon\mathcal M\to\mathbb R$ to minimize

# See also
[`NelderMead`](@ref)
"""
struct CostProblem{mT <: Manifold, Tcost} <: Problem
    M::mT
    cost::Tcost
end

@doc raw"""
    NelderMeadOptions <: Options

Describes all parameters and the state of a Nealer-Mead heuristic based
optimization algorithm.

# Fields

The naming of these parameters follows the [Wikipedia article](https://en.wikipedia.org/wiki/Nelder–Mead_method)
of the Euclidean case. The default is given in brackets, the required value range
after the description

* `population` – an `Array{`point`,1}` of $n+1$ points $x_i$, $i=1,\ldots,n+1$, where $n$ is the
  dimension of the manifold.
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(2000)`) a [`StoppingCriterion`](@ref)
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

construct a Nelder-Mead Option with a set `p` of points
"""
mutable struct NelderMeadOptions{T,Tα<:Real,Tγ<:Real,Tρ<:Real,Tσ<:Real} <: Options
    population::Vector{T}
    stop::StoppingCriterion
    α::Tα
    γ::Tγ
    ρ::Tρ
    σ::Tσ
    x::T
    costs::Vector{Float64}
    function NelderMeadOptions(
        M::MT;
        stop::StoppingCriterion = StopAfterIteration(2000),
        α = 1.,
        γ = 2.,
        ρ=1/2,
        σ = 1/2
    ) where {MT <: Manifold }
        p = [random_point(M) for i=1:(manifold_dimension(M)+1) ]
        new{eltype(p),typeof(α),typeof(γ),typeof(ρ),typeof(σ)}(p, stop, α, γ, ρ, σ, p[1],[])
    end
    function NelderMeadOptions(population::Vector{T},
        stop::StoppingCriterion = StopAfterIteration(2000);
        α = 1., γ = 2., ρ=1/2, σ = 1/2
    ) where {T}
        return new{T,typeof(α),typeof(γ),typeof(ρ),typeof(σ)}(population, stop, α, γ, ρ, σ, population[1],[])
    end
end
get_solver_result(o::NelderMeadOptions) = o.x
