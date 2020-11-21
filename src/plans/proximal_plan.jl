#
#
# Proximal Point Problem and Options
#
#
export ProximalProblem
export CyclicProximalPointOptions, DouglasRachfordOptions
export get_cost, get_proximal_map
export DebugProximalParameter
export RecordProximalParameter

@doc raw"""
    ProximalProblem <: Problem
specify a problem for solvers based on the evaluation of proximal map(s).

# Fields
* `M` - a [Manifold](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold) $\mathcal M$
* `cost` - a function $F\colon\mathcal M\to\mathbb R$ to
  minimize
* `proxes` - proximal maps $\operatorname{prox}_{\lambda\varphi}\colon\mathcal M\to\mathcal M$
  as functions (λ,x) -> y, i.e. the prox parameter λ also belongs to the signature of the proximal map.
* `number_of_proxes` - (length(proxes)) number of proxmal Maps,
  e.g. if one of the maps is a combined one such that the proximal Maps
  functions return more than one entry per function

# See also
[`cyclic_proximal_point`](@ref), [`get_cost`](@ref), [`get_proximal_map`](@ref)
"""
mutable struct ProximalProblem{mT<:Manifold,TCost,TProxes<:Union{Tuple,AbstractVector}} <:
               Problem
    M::mT
    cost::TCost
    proxes::TProxes
    number_of_proxes::Vector{Int}
    function ProximalProblem(
        M::mT, cF, proxMaps::Union{Tuple,AbstractVector}
    ) where {mT<:Manifold}
        return new{mT,typeof(cF),typeof(proxMaps)}(M, cF, proxMaps, ones(length(proxMaps)))
    end
    function ProximalProblem(
        M::mT, cF, proxMaps::Union{Tuple,AbstractVector}, nOP::Vector{Int}
    ) where {mT<:Manifold}
        return if length(nOP) != length(proxMaps)
            throw(ErrorException("The number_of_proxes ($(nOP)) has to be the same length as the number of proxes ($(length(proxMaps))."))
        else
            new{mT,typeof(cF),typeof(proxMaps)}(M, cF, proxMaps, nOP)
        end
    end
end
@doc raw"""
    get_proximal_map(p,λ,x,i)

evaluate the `i`th proximal map of `ProximalProblem p` at the point `x` of `p.M` with parameter `λ`$>0$.
"""
function get_proximal_map(p::ProximalProblem, λ, x, i)
    if i > length(p.proxes)
        throw(ErrorException("the $(i)th entry does not exists, only $(length(p.proxes)) available."))
    end
    return p.proxes[i](λ, x)::typeof(x)
end
#
#
# Proximal based Options
#
#
"""
    CyclicProximalPointOptions <: Options

stores options for the [`cyclic_proximal_point`](@ref) algorithm. These are the

# Fields
* `x0` – an point to start
* `stopping_criterion` – a function `@(iter,x,xnew,λ_k)` based on the current
    `iter`, `x` and `xnew` as well as the current value of `λ`.
* `λ` – (@(iter) -> 1/iter) a function for the values of λ_k per iteration/cycle
* `evaluationOrder` – ([`LinearEvalOrder`](@ref)`()`) how to cycle through the proximal maps.
    Other values are [`RandomEvalOrder`](@ref)`()` that takes a new random order each
    iteration, and [`FixedRandomEvalOrder`](@ref)`()` that fixes a random cycle for all iterations.

# See also
[`cyclic_proximal_point`](@ref)
"""
mutable struct CyclicProximalPointOptions{
    TX,TStop<:StoppingCriterion,Tλ,TOrder<:EvalOrder
} <: Options
    x::TX
    stop::TStop
    λ::Tλ
    orderType::TOrder
    order::Vector{Int}
end
function CyclicProximalPointOptions(
    x, s::StoppingCriterion, λ=(iter) -> 1.0 / iter, o::EvalOrder=LinearEvalOrder()
)
    return CyclicProximalPointOptions{typeof(x),typeof(s),typeof(λ),typeof(o)}(
        x, s, λ, o, []
    )
end
@doc raw"""
    DouglasRachfordOptions <: Options

Store all options required for the DouglasRachford algorithm,

# Fields
* `x` - the current iterate (result) For the parallel Douglas-Rachford, this is
  not a value from the `PowerManifold` manifold but the mean.
* `s` – the last result of the double reflection at the proxes relaxed by `α`.
* `λ` – (`(iter)->1.0`) function to provide the value for the proximal parameter
  during the calls
* `α` – (`(iter)->0.9`) relaxation of the step from old to new iterate, i.e.
  $x^{(k+1)} = g(α(k); x^{(k)}, t^{(k)})$, where $t^{(k)}$ is the result
  of the double reflection involved in the DR algorithm
* `R` – ([`reflect`](@ref)) method employed in the iteration to perform the reflection of `x` at
  the prox `p`.
* `stop` – ([`StopAfterIteration`](@ref)`(300)`) a [`StoppingCriterion`](@ref)
* `parallel` – (`false`) inducate whether we are running a pallel Douglas-Rachford
  or not.
"""
mutable struct DouglasRachfordOptions{TX,Tλ,Tα,TR} <: Options
    x::TX
    s::TX
    λ::Tλ
    α::Tα
    R::TR
    stop::StoppingCriterion
    parallel::Bool
    function DouglasRachfordOptions(
        x,
        λ=(iter) -> 1.0,
        α=(iter) -> 0.9,
        R=reflect,
        stop::StoppingCriterion=StopAfterIteration(300),
        parallel=false,
    )
        return new{typeof(x),typeof(λ),typeof(α),typeof(R)}(x, x, λ, α, R, stop, parallel)
    end
end
#
# Debug
#
# overwrite defaults, since we store the result in the mean field
#
# Debug the Cyclic Proximal point parameter
#
@doc raw"""
    DebugProximalParameter <: DebugAction

print the current iterates proximal point algorithm parameter given by
[`Options`](@ref)s `o.λ`.
"""
mutable struct DebugProximalParameter <: DebugAction
    io::IO
    prefix::String
    function DebugProximalParameter(long::Bool=false, io::IO=stdout)
        return new(io, long ? "Proximal Map Parameter λ(i):" : "λ:")
    end
end
function (d::DebugProximalParameter)(::ProximalProblem, o::DouglasRachfordOptions, i::Int)
    print(d.io, (i > 0) ? d.prefix * string(o.λ(i)) : "")
    return nothing
end
function (d::DebugProximalParameter)(
    ::ProximalProblem, o::CyclicProximalPointOptions, i::Int
)
    print(d.io, (i > 0) ? d.prefix * string(o.λ(i)) : "")
    return nothing
end

#
# Record
@doc raw"""
    RecordProximalParameter <: RecordAction

recoed the current iterates proximal point algorithm parameter given by in
[`Options`](@ref)s `o.λ`.
"""
mutable struct RecordProximalParameter <: RecordAction
    recorded_values::Array{Float64,1}
    RecordProximalParameter() = new(Array{Float64,1}())
end
function (r::RecordProximalParameter)(
    ::ProximalProblem, o::CyclicProximalPointOptions, i::Int
)
    return record_or_reset!(r, o.λ(i), i)
end
function (r::RecordProximalParameter)(::ProximalProblem, o::DouglasRachfordOptions, i::Int)
    return record_or_reset!(r, o.λ(i), i)
end
