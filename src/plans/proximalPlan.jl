#
#
# Proximal Point Problem and Options
#
#
export ProximalProblem
export CyclicProximalPointOptions, DouglasRachfordOptions
export getCost, getProximalMap, getProximalMaps
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

evaluate the cost function `F` stored within a [`ProximalProblem`](@ref) at the [`MPoint`](@ref) `x`.
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
        ErrorException("the $(i)th entry does not exists, only $(length(p.proximalMaps)) available.")
    end
    return p.proximalMaps[i](λ,x);
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
* `evaluationOrder` : (`LinearEvalOrder()`) how to cycle through the proximal maps.
    Other values are `RandomEvalOrder()` that takes a new random order each
    iteration, and `FixedRandomEvalOrder()` that fixes a random cycle for all iterations.

# See also
[`cyclicProximalPoint`](@ref)
"""
mutable struct CyclicProximalPointOptions{P} <: Options where {P <: MPoint}
    x::P
    xOld::P
    stoppingCriterion::Function
    λ::Function
    orderType::EvalOrder
    order::Array{Int,1}
    CyclicProximalPointOptions{P}(x::P,sC::Function,λ::Function=(iter)-> 1.0/iter,o::EvalOrder=LinearEvalOrder()) where {P <: MPoint} = new(x,x,sC,λ,o,[])
end
CyclicProximalPointOptions(x::P,sC::Function,λ::Function=(iter)-> 1.0/iter,o::EvalOrder=LinearEvalOrder()) where {P <: MPoint} = CyclicProximalPointOptions{P}(x,sC,λ,o)
@doc doc"""
    DouglasRachfordOptions <: Options
Store all options required for the DouglasRachford algorithm,

# Fields
* `x0` - initial start point
* `λ` – (`(iter)->1.0`) function to provide the value for the proximal parameter
  during the calls
* `α` – (`(iter)->0.9`) relaxation of the step from old to new iterate, i.e.
  $x^{(k+1)} = g(α(k); x^{(k)}, t^{(k)})$, where $t^{(k)}$ is the result
  of the double reflection involved in the DR algorithm
* `R` – ([`reflection`](@ref)) method employed in the iteration to perform the reflection of `x` at
  the prox `p`.
"""
mutable struct DouglasRachfordOptions <: Options
    x::P where {P <: MPoint}
    xOld::P where {P <: MPoint}
    stoppingCriterion::Function
    λ::Function
    α::Function
    R::Function
    DouglasRachfordOptions(x::P where {P <: MPoint}, sC::Function, λ::Function=(iter)->1.0, α::Function=(iter)->0.9, R=reflection) = new(x,x,sC,λ,α,reflection)
end
debug(p::ProximalProblem, o::CyclicProximalPointOptions, ::Val{:λ}, iter::Int, out::IO=Base.stdout) where {P <: Problem, O <: Options} = print(out,"λ: ",o.λ(iter))