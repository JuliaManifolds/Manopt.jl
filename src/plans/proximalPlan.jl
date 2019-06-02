#
#
# Proximal Point Problem and Options
#
#
export ProximalProblem
export CyclicProximalPointOptions, DouglasRachfordOptions
export getCost, getProximalMap
export DebugProximalParameter
export RecordProximalParameter

@doc doc"""
    ProximalProblem <: Problem
specify a problem for solvers based on the evaluation of proximal map(s).

# Fields
* `M`            - a [`Manifold`](@ref) $\mathcal M$
* `costFunction` - a function $F\colon\mathcal M\to\mathbb R$ to
  minimize
* `proximalMaps` - proximal maps $\operatorname{prox}_{\lambda\varphi}\colon\mathcal M\to\mathcal M$
  as functions (λ,x) -> y, i.e. the prox parameter λ also belongs to the signature of the proximal map.
* `numberOfProxes` - (length(proximalMaps)) number of proxmal Maps,
  e.g. if one of the maps is a compined one such that the proximal Maps
  functions return more than one entry per function

# See also
[`cyclicProximalPoint`](@ref), [`getCost`](@ref), [`getProximalMap`](@ref)
"""
mutable struct ProximalProblem{mT <: Manifold} <: Problem
  M::mT
  costFunction::Function
  proximalMaps::Array{Function,N} where N
  numberOfProxes::Array{Int,1}
  ProximalProblem(M::mT, cF::Function, proxMaps::Array{Function,1}) where {mT <: Manifold}= new{mT}(M,cF,proxMaps,ones(length(proxMaps)))
  ProximalProblem(M::mT, cF::Function, proxMaps::Array{Function,1}, nOP::Array{Int,1}) where {mT <: Manifold} =
    length(nOP) != length(proxMaps) ? throw(ErrorException("The numberOfProxes ($(nOP)) has to be the same length as the number of Proxes ($(length(proxMaps)).")) :
    new{mT}(M,cF,proxMaps,nOP)
end
@doc doc"""
    getProximalMap(p,λ,x,i)

evaluate the `i`th proximal map of `ProximalProblem p` at the point `x` of `p.M` with parameter `λ`$>0$.
"""
function getProximalMap(p::P,λ,x::MP,i) where {P <: ProximalProblem{M} where M <: Manifold, MP<:MPoint}
    if i>length(p.proximalMaps)
        throw( ErrorException("the $(i)th entry does not exists, only $(length(p.proximalMaps)) available.") )
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
* `x0` – an [`MPoint`](@ref) to start
* `stoppingCriterion` – a function `@(iter,x,xnew,λ_k)` based on the current
    `iter`, `x` and `xnew` as well as the current value of `λ`.
* `λ` – (@(iter) -> 1/iter) a function for the values of λ_k per iteration/cycle
* `evaluationOrder` – ([`LinearEvalOrder`](@ref)`()`) how to cycle through the proximal maps.
    Other values are [`RandomEvalOrder`](@ref)`()` that takes a new random order each
    iteration, and [`FixedRandomEvalOrder`](@ref)`()` that fixes a random cycle for all iterations.

# See also
[`cyclicProximalPoint`](@ref)
"""
mutable struct CyclicProximalPointOptions{P} <: Options where {P <: MPoint}
    x::P
    stop::StoppingCriterion
    λ::Function
    orderType::EvalOrder
    order::Array{Int,1}
    CyclicProximalPointOptions{P}(x::P,s::StoppingCriterion, λ::Function=(iter)-> 1.0/iter,o::EvalOrder=LinearEvalOrder()) where {P <: MPoint} = new(x,s,λ,o,[])
end
CyclicProximalPointOptions(x::P,s::StoppingCriterion,λ::Function=(iter)-> 1.0/iter,o::EvalOrder=LinearEvalOrder()) where {P <: MPoint} = CyclicProximalPointOptions{P}(x,s,λ,o)
@doc doc"""
    DouglasRachfordOptions <: Options

Store all options required for the DouglasRachford algorithm,

# Fields
* `x` - the current iterate (result) For the parallel Douglas-Rachford, this is
  not a value from the [`Power`](@ref) manifold but the mean.
* `s` – the last result of the double reflection at the proxes relaxed by `α`.
* `λ` – (`(iter)->1.0`) function to provide the value for the proximal parameter
  during the calls
* `α` – (`(iter)->0.9`) relaxation of the step from old to new iterate, i.e.
  $x^{(k+1)} = g(α(k); x^{(k)}, t^{(k)})$, where $t^{(k)}$ is the result
  of the double reflection involved in the DR algorithm
* `R` – ([`reflection`](@ref)) method employed in the iteration to perform the reflection of `x` at
  the prox `p`.
* `stop` – ([`stopAfterIteration`](@ref)`(300)`) a [`StoppingCriterion`](@ref)
* `parallel` – (`false`) inducate whether we are running a pallel Douglas-Rachford
  or not.
"""
mutable struct DouglasRachfordOptions <: Options
    x::P where {P <: MPoint}
    s::P where {P <: MPoint}
    λ::Function
    α::Function
    R::Function
    stop::StoppingCriterion
    parallel::Bool
    DouglasRachfordOptions(x::P where {P <: MPoint}, λ::Function=(iter)->1.0,
        α::Function=(iter)->0.9, R=reflection,
        stop::StoppingCriterion = stopAfterIteration(300),
        parallel=false
    ) = new(x,x,λ,α,reflection,stop,parallel)
end
#
# Debug
#
# overwrite defaults, since we store the result in the mean field
#
# Debug the Cyclic Proximal point parameter
#
@doc doc"""
    DebugProximalParameter <: DebugAction

print the current iterates proximal point algorithm parameter given by
[`Options`](@ref)s `o.λ`.
"""
mutable struct DebugProximalParameter <: DebugAction
    print::Function
    prefix::String
    DebugProximalParameter(long::Bool=false,print::Function=print) = new(print, long ? "Proximal Map Parameter λ(i):" : "λ:" )
end
(d::DebugProximalParameter)(p::ProximalProblem,o::DouglasRachfordOptions,i::Int) = d.print((i>0) ? d.prefix*string(o.λ(i)) : "")
(d::DebugProximalParameter)(p::ProximalProblem,o::CyclicProximalPointOptions,i::Int) = d.print((i>0) ? d.prefix*string(o.λ(i)) : "")

#
# Record
@doc doc"""
    RecordProximalParameter <: RecordAction

recoed the current iterates proximal point algorithm parameter given by in
[`Options`](@ref)s `o.λ`.
"""
mutable struct RecordProximalParameter <: RecordAction
    recordedValues::Array{Float64,1}
    RecordProximalParameter() = new(Array{Float64,1}())
end
(r::RecordProximalParameter)(p::P,o::O,i::Int) where {P <: ProximalProblem, O <: CyclicProximalPointOptions} = recordOrReset!(r, o.λ(i), i)
(r::RecordProximalParameter)(p::P,o::O,i::Int) where {P <: ProximalProblem, O <: DouglasRachfordOptions} = recordOrReset!(r, o.λ(i), i)
