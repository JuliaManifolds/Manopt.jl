#
#
# Proximal Point Problem and Options
#
#
@doc raw"""
    ProximalProblem <: Problem
specify a problem for solvers based on the evaluation of proximal map(s).

# Fields
* `M` - a Riemannian manifold
* `cost` - a function ``F:\mathcal M→ℝ`` to
  minimize
* `proxes` - proximal maps ``\operatorname{prox}_{λ\varphi}:\mathcal M→\mathcal M``
  as functions (λ,x) -> y, i.e. the prox parameter λ also belongs to the signature of the proximal map.
* `number_of_proxes` - (length(proxes)) number of proximal Maps,
  e.g. if one of the maps is a combined one such that the proximal Maps
  functions return more than one entry per function

# See also
[`cyclic_proximal_point`](@ref), [`get_cost`](@ref), [`get_proximal_map`](@ref)
"""
mutable struct ProximalProblem{
    T,mT<:AbstractManifold,TCost,TProxes<:Union{Tuple,AbstractVector}
} <: Problem{T}
    M::mT
    cost::TCost
    proximal_maps!!::TProxes
    number_of_proxes::Vector{Int}
    function ProximalProblem(
        M::mT,
        cF,
        proxMaps::Union{Tuple,AbstractVector};
        evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    ) where {mT<:AbstractManifold}
        return new{typeof(evaluation),mT,typeof(cF),typeof(proxMaps)}(
            M, cF, proxMaps, ones(length(proxMaps))
        )
    end
    function ProximalProblem(
        M::mT,
        cF,
        proxMaps::Union{Tuple,AbstractVector},
        nOP::Vector{Int};
        evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    ) where {mT<:AbstractManifold}
        return if length(nOP) != length(proxMaps)
            throw(
                ErrorException(
                    "The number_of_proxes ($(nOP)) has to be the same length as the number of proxes ($(length(proxMaps)).",
                ),
            )
        else
            new{typeof(evaluation),mT,typeof(cF),typeof(proxMaps)}(M, cF, proxMaps, nOP)
        end
    end
end
function check_prox_number(n, i)
    (i > n) && throw(ErrorException("the $(i)th entry does not exists, only $n available."))
    return true
end
@doc raw"""
    get_proximal_map(p,λ,x,i)

evaluate the `i`th proximal map of `ProximalProblem p` at the point `x` of `p.M` with parameter ``λ>0``.
"""
function get_proximal_map(p::ProximalProblem{AllocatingEvaluation}, λ, x, i)
    check_prox_number(length(p.proximal_maps!!), i)
    return p.proximal_maps!![i](p.M, λ, x)
end
function get_proximal_map!(p::ProximalProblem{AllocatingEvaluation}, y, λ, x, i)
    check_prox_number(length(p.proximal_maps!!), i)
    return copyto!(p.M, y, p.proximal_maps!![i](p.M, λ, x))
end
function get_proximal_map(p::ProximalProblem{MutatingEvaluation}, λ, x, i)
    check_prox_number(length(p.proximal_maps!!), i)
    y = allocate_result(p.M, get_proximal_map, x)
    return p.proximal_maps!![i](p.M, y, λ, x)
end
function get_proximal_map!(p::ProximalProblem{MutatingEvaluation}, y, λ, x, i)
    check_prox_number(length(p.proximal_maps!!), i)
    return p.proximal_maps!![i](p.M, y, λ, x)
end
#
#
# Proximal based Options
#
#
@doc raw"""
    CyclicProximalPointOptions <: Options

stores options for the [`cyclic_proximal_point`](@ref) algorithm. These are the

# Fields
* `x` – the current iterate
* `stopping_criterion` – a [`StoppingCriterion`](@ref)
* `λ` – (@(iter) -> 1/iter) a function for the values of λ_k per iteration/cycle
* `oder_type` – (`:LinearOrder`) – whether
  to use a randomly permuted sequence (`:FixedRandomOrder`), a per
  cycle permuted sequence (`RandomOrder`) or the default linear one.

# Constructor
    CyclicProximalPointOptions(M, p)

Generate the options with the following keyword arguments

* `stopping_criterion` (`StopAfterIteration(2000)`) – a [`StoppingCriterion`](@ref).
* `λ` ( `(iter) -> 1.0 / iter`) a function to compute the ``λ_k, k ∈ \mathbb N``,
* `evaluation_order`(`:LinearOrder`) a Symbol indicating the order the proxes are applied.

# See also
[`cyclic_proximal_point`](@ref)
"""
mutable struct CyclicProximalPointOptions{TX,TStop<:StoppingCriterion,Tλ} <: Options
    x::TX
    stop::TStop
    λ::Tλ
    order_type::Symbol
    order::AbstractVector{Int}
end
@deprecate CyclicProximalPointOptions(
    x, s::StoppingCriterion, λ=(iter) -> 1.0 / iter, o::Symbol=:LinearOrder
) CyclicProximalPointOptions(
    DefaultManifold(2), x; stopping_criterion=s, λ=λ, evaluation_order=o
)

function CyclicProximalPointOptions(
    M::AbstractManifold,
    x::P;
    stopping_criterion::S=StopAfterIteration(2000),
    λ::F=(iter) -> 1.0 / iter,
    evaluation_order::Symbol=:LinearOrder,
) where {P,S,F}
    return CyclicProximalPointOptions{P,S,F}(x, stopping_criterion, λ, evaluation_order, [])
end
@doc raw"""
    DouglasRachfordOptions <: Options

Store all options required for the DouglasRachford algorithm,

# Fields
* `x` - the current iterate (result) For the parallel Douglas-Rachford, this is
  not a value from the `PowerManifold` manifold but the mean.
* `s` – the last result of the double reflection at the proxes relaxed by `α`.
* `λ` – function to provide the value for the proximal parameter during the calls
* `α` – relaxation of the step from old to new iterate, i.e.
  ``x^{(k+1)} = g(α(k); x^{(k)}, t^{(k)})``, where ``t^{(k)}`` is the result
  of the double reflection involved in the DR algorithm
* `R` – method employed in the iteration to perform the reflection of `x` at the prox `p`.
* `stop` – a [`StoppingCriterion`](@ref)
* `parallel` – indicate whether we are running a parallel Douglas-Rachford or not.

# Constructor

    DouglasRachfordOptions(M, p; kwargs...)

Generate the options for a Manifold `M` and an initial point `p`, where the following keyword arguments can be used

* `λ` – (`(iter)->1.0`) function to provide the value for the proximal parameter
  during the calls
* `α` – (`(iter)->0.9`) relaxation of the step from old to new iterate, i.e.
  ``x^{(k+1)} = g(α(k); x^{(k)}, t^{(k)})``, where ``t^{(k)}`` is the result
  of the double reflection involved in the DR algorithm
* `R` – ([`reflect`](@ref)) method employed in the iteration to perform the reflection of `x` at
  the prox `p`.
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(300)`) a [`StoppingCriterion`](@ref)
* `parallel` – (`false`) indicate whether we are running a parallel Douglas-Rachford
  or not.
"""
mutable struct DouglasRachfordOptions{TX,Tλ,Tα,TR,S} <: Options
    x::TX
    xtmp::TX
    s::TX
    stmp::TX
    λ::Tλ
    α::Tα
    R::TR
    stop::S
    parallel::Bool
    function DouglasRachfordOptions(
        ::AbstractManifold,
        x::P;
        λ::Fλ=(iter) -> 1.0,
        α::Fα=(iter) -> 0.9,
        R::FR=Manopt.reflect,
        stopping_criterion::S=StopAfterIteration(300),
        parallel=false,
    ) where {P,Fλ,Fα,FR,S<:StoppingCriterion}
        return new{P,Fλ,Fα,FR,S}(
            x, deepcopy(x), deepcopy(x), deepcopy(x), λ, α, R, stopping_criterion, parallel
        )
    end
    @deprecate DouglasRachfordOptions(
        x,
        λ=(iter) -> 1.0,
        α=(iter) -> 0.9,
        R=Manopt.reflect,
        stop::StoppingCriterion=StopAfterIteration(300),
        parallel=false,
    ) DouglasRachfordOptions(
        DefaultManifold(2), x; λ=λ, α=α, R=R, stopping_criterion=stop, parallel=parallel
    )
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
    format::String
    function DebugProximalParameter(;
        long::Bool=false,
        prefix=long ? "Proximal Map Parameter λ(i):" : "λ:",
        format="$prefix%s",
        io::IO=stdout,
    )
        return new(io, format)
    end
end
function (d::DebugProximalParameter)(::ProximalProblem, o::DouglasRachfordOptions, i::Int)
    (i > 0) && Printf.format(d.io, Printf.Format(d.format), o.λ(i))
    return nothing
end
function (d::DebugProximalParameter)(
    ::ProximalProblem, o::CyclicProximalPointOptions, i::Int
)
    (i > 0) && Printf.format(d.io, Printf.Format(d.format), o.λ(i))
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
