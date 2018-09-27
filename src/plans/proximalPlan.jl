#
#
# Proximal Point Problem and Options
#
#
@doc doc"""
    ProximalProblem <: Problem
specify a problem for solvers based on the evaluation of proximal map(s).

# Fields
* `M`            : a manifold $\mathcal M$
* `costFunction` : a function $F\colon\mathcal M\to\mathbb R$ to minimize
* `proximalMaps` : proximal maps $\operatorname{prox}_{\lambda\varphi}\colon\mathcal M\to\mathcal M$
  as functions (λ,x) -> y, i.e. the prox parameter λ also belongs to the signature of the proximal map.
# See also
[`cyclicProximalPoint`](@ref), [`getCost`](@ref),
[`getProximalMaps`](@ref),[`getProximalMap`](@ref),
"""
mutable struct ProximalProblem{mT <: Manifold} <: Problem
  M::mT
  costFunction::Function
  proximalMaps::Array{Function,N} where N
end
"""
    getCost(p,x)

evaluate the cost function `F` stored within a [`ProximalProblem`](@ref) at the [`MPoint`](@ref)` x`.
"""
function getCost(p::P,x::MP) where {P <: ProximalProblem{M} where M <: Manifold, MP <: MPoint}
  return p.costFunction(x)
end
@doc doc"""
    getProximalMaps(p,λ,x)
evaluate all proximal maps of `ProximalProblem p` at the point `x` of `p.M` and
some `λ`$>0$ which might be given as a vector the same length as the number of
proximal maps.
"""
getProximalMaps(p::P,λ,x::MP) where {P <: ProximalProblem{M} where M <: Manifold, MP<:MPoint} =
    p.proximalMaps.(λ,x);
@doc doc"""
    getProximalMap(p,λ,x,i)
evaluate the `i`th proximal map of `ProximalProblem p` at the point `x` of `p.M` with parameter `λ`$>0$.
"""
function getProximalMap(p::P,λ,x::MP,i) where {P <: ProximalProblem{M} where M <: Manifold, MP<:MPoint}
    if i>length(p.proximalMaps)
        ErrorException("the $(i)th entry does not exists, only $(len(p.proximalMaps)) available.")
    end
    return p.proximalMaps[i].(λ,x);
end
#
#
# Proximal based Options
#
#
"""
    CyclicProximalPointOptions <: Options
stores options for the [`cyclicProximalPoint`](@ref) algorithm. These are the

# Fields
* `x0` : an [`MPoint`](@ref) to start
* `stoppingCriterion` : a function `@(iter,x,xnew,λ_k)` based on the current
    `iter`, `x` and `xnew` as well as the current value of `λ`.
* `λ` : (@(iter) -> 1/iter) a function for the values of λ_k per iteration/cycle
* `orderType` : (`LinearEvalOrder()`) how to cycle through the proximal maps.
    Other values are `RandomEvalOrder()` that takes a new random order each
    iteration, and `FixedRandomEvalOrder()` that fixes a random cycle for all iterations.

# See also
[`cyclicProximalPoint`](@ref)
"""
mutable struct CyclicProximalPointOptions <: Options
    x0::P where {P <: MPoint}
    stoppingCriterion::Function
    λ::Function
    orderType::EvalOrder
    CyclicProximalPointOptions(x0::P where {P <: MPoint},sC::Function,λ::Function=(iter)-> 1.0/iter,o::EvalOrder=LinearEvalOrder()) = new(x0,sC,λ,o)
end
function evaluateStoppingCriterion(o::O,iter::I, x::P, xnew::P,λ) where {O<:CyclicProximalPointOptions, P <: MPoint, I<:Integer}
  o.stoppingCriterion(iter, x, xnew, λ)
end
