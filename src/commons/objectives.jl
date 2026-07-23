#
#
# ---
@doc """
    ManifoldCostObjective{F} <: AbstractManifoldCostObjective{F}

specify an [`AbstractManifoldObjective`](@ref) that does only have information about
the cost function ``f:  $(_math(:Manifold)) → ℝ`` implemented as a function `(M, p) -> c`
to compute the cost value `c` at `p` on the manifold `M`.

* `cost`: a function ``f: $(_math(:Manifold)) → ℝ`` to minimize

# Constructors

    ManifoldCostObjective(f::F)

Generate a problem. While this Problem does not have any allocating functions,

## See also
[`NelderMead`](@ref), [`particle_swarm`](@ref)
"""
struct ManifoldCostObjective{F} <: AbstractManifoldCostObjective{F}
    cost::F
end
function show(io::IO, mco::ManifoldCostObjective{F}) where {F}
    return print(io, "ManifoldCostObjective(mco.cost)")
end
function status_summary(::ManifoldCostObjective{F}; context::Symbol = :default) where {F}
    return "A cost function on a Riemannian manifold `f = (M,p) -> ℝ`."
end

#
#
# ---
@doc """
    ManifoldFirstOrderObjective{E<:AbstractEvaluationType, F} <: AbstractManifoldFirstOrderObjective{E, F}

specify an objective containing a cost and its gradient or differential,
where the [`AbstractEvaluationType`](@ref) `E` indicates the type of evaluation for a gradient.

# Fields

* `functions::F`: a function or a tuple of functions containing the cost and first order information.

Currently the following cases are covered, sorted by their popularity

1. a single function `fg`, i.e. a function, represents a combined
    function `(M, X, p) -> (c, X)` that computes the cost `c=cost(M,p)` and gradient `X=grad_f(M, X, p)`;
2. a single function `fdf`, i.e. a function, represents a combined function
    `(M, d, p) -> (c, d)` that computes the cost `c=cost(M,p)` and differential `d=diff_f(M, d, p)`;
3. pairs of single functions `(f, g)`, `(f, df)` of a cost function `f` and either its
    gradient `g` or its differential `d`, respectively
4. The function `(fg, d)` and `(fdf, g)`  from 1 and 2, respectively joined by
    the other missing third information, the differential for the first or the gradient for the second
5. a tuple `(f, g, d)` of three functions, computing cost, `f`, gradient `g`,
    and `differential `d` separately
6. a `(f, gd)` of a cost function and a combined function `(X, d) = gd(M, (X,d), p)`
    to compute gradient and differential together

For all cases where a gradient and/or a differential is present are considered to work in-place,
see [`AllocatingManifoldFunction`](@ref) for alternatives.

The cases of a common `fg` function for cost and gradient and the tuple `(f,g)` are the most common one.
They can also be addressed by their alternate constructors
[`ManifoldCostGradientObjective`](@ref)`(fg)` and [`ManifoldGradientObjective`](@ref)`(f,g)`, respectively.

# Constructors
    ManifoldFirstOrderObjective(; kwargs...)

## Keyword arguments

* `cost = nothing` the cost function `c = f(M,p)`
* `differential = nothing` the differential `d = df(M, p, X)`
* `gradient=nothing` the gradient function `g(M, p)` or in-place `g!(M, X, p)`
* `costgradient = nothing` the combined cost and gradient function `fg(M,p)` or in-place `fg!(M, X, p))`
* `costdifferential = nothing` the combined cost and differential function  `fdf(M, p, X)`

Where:
 * At least one of `cost`, `costgradient` or `costdifferential` must be provided.
 * Either `gradient`, `costgradient`, `differential` or `costdifferential` must be provided.
 * If more than one function provides the same thing (e.g. cost), it is assumed that all
   such functions return the same value. Optimization algorithms will attempt to make the
   most efficient use of provided functions fitting for the access required.

# Used with
[`gradient_descent`](@ref), [`conjugate_gradient_descent`](@ref), [`quasi_Newton`](@ref)
"""
struct ManifoldFirstOrderObjective{F <: NamedTuple} <: AbstractManifoldFirstOrderObjective{F, F}
    functions::F
end
# TODO: Test here how to maybe handle the old evaluation= kwarg to now automatically “wrap”
# allocating variants.
function ManifoldFirstOrderObjective(;
        cost = nothing, differential = nothing, gradient = nothing,
        costgradient = nothing, costdifferential = nothing,
    )
    no_cost = isnothing(cost)
    no_diff = isnothing(differential)
    no_grad = isnothing(gradient)
    ncg = isnothing(costgradient)
    ncd = isnothing(costdifferential)

    if no_cost && ncg && ncd
        throw(
            ArgumentError(
                "Either cost, costgradient or costdifferential keyword argument needs to be provided",
            ),
        )
    end
    if no_grad && ncg && no_diff && ncd
        throw(
            ArgumentError(
                "Either gradient, costgradient, differential or costdifferential keyword argument needs to be provided",
            ),
        )
    end
    nt = (;)
    if !no_cost
        nt = merge(nt, (; cost = cost))
    end
    if !no_grad
        nt = merge(nt, (; gradient = gradient))
    end
    if !no_diff
        nt = merge(nt, (; differential = differential))
    end
    if !ncg
        nt = merge(nt, (; costgradient = costgradient))
    end
    if !ncd
        nt = merge(nt, (; costdifferential = costdifferential))
    end
    return ManifoldFirstOrderObjective{typeof(nt)}(nt)
end

const ManifoldGradientObjective{F, G} = ManifoldFirstOrderObjective{
    <:Union{
        NamedTuple{Tuple{:cost, :gradient}, Tuple{F, G}},
        NamedTuple{Tuple{:cost, :gradient, :differential}, Tuple{F, G, D where {D}}},
    },
}
@doc """
    ManifoldGradientObjective(cost, gradient; kwargs...)

Generate an objective with a function `cost` and its `gradient`.
The gradient is assumed to work in-place

    * as a function `(M, X, p) -> X` that work in place of `X`, an [`InplaceEvaluation`](@ref)

Internally this is stored in a [`ManifoldFirstOrderObjective`](@ref). The `kwargs...`
are also passed to this representation, which allows to add a special function
to evaluate the `differential`.

# Used with
[`gradient_descent`](@ref), [`conjugate_gradient_descent`](@ref), [`quasi_Newton`](@ref)
"""
function ManifoldGradientObjective(cost, grad; kwargs...)
    return ManifoldFirstOrderObjective(; cost = cost, gradient = grad, kwargs...)
end

const ManifoldCostGradientObjective{FG} = ManifoldFirstOrderObjective{
    <:Union{
        NamedTuple{Tuple{:costgradient}, Tuple{FG}},
        NamedTuple{Tuple{:costgradient, :differential}, Tuple{FG, D where {D}}},
    },
}
@doc """
    ManifoldCostGradientObjective(costgrad; evaluation::E=AllocatingEvaluation(), kwargs...)

create an objective containing one function to perform a combined computation of cost and its gradient

Depending on the [`AbstractEvaluationType`](@ref) `E` the gradient can have to forms

* as a function `(M, p) -> (c, X)` that allocates memory for the gradient `X`, an [`AllocatingEvaluation`](@ref)
* as a function `(M, X, p) -> (c, X)` that work in place of `X`, an [`InplaceEvaluation`](@ref)

Internally this is stored in a [`ManifoldFirstOrderObjective`](@ref). The `kwargs...`
are also passed to this representation, which allows to add a special function
to evaluate the `differential`.

# Used with
[`gradient_descent`](@ref), [`conjugate_gradient_descent`](@ref), [`quasi_Newton`](@ref)
"""
function ManifoldCostGradientObjective(cost_grad; kwargs...)
    return ManifoldFirstOrderObjective(; costgradient = cost_grad, kwargs...)
end

# accessors
function get_cost(
        M::AbstractManifold, mfo::ManifoldFirstOrderObjective, p
    )
    haskey(mfo.functions, :cost) && (return mfo.functions[:cost](M, p))
    X = zero_vector(M, p)
    if haskey(mfo.functions, :costdifferential)
        return mfo.functions[:costdifferential](M, X, p, X)[1]
    end
    haskey(mfo.functions, :costgradient) && (return mfo.functions[:costgradient](M, X, p)[1])
    return error("$mfo does not seem to provide a cost")
end

#TODO: Since Y is a keyword, maybe a better name is gradient_cache? and add the evaluated bool here as well
function get_cost_and_differential(
        M::AbstractManifold, mfo::ManifoldFirstOrderObjective, p, X; Y = nothing,
    )
    if haskey(mfo.functions, :costdifferential)
        return mfo.functions[:costdifferential](M, p, X)
    elseif haskey(mfo.functions, :cost) && haskey(mfo.functions, :differential)
        return (mfo.functions[:cost](M, p), mfo.functions[:differential](M, p, X))
    elseif haskey(mfo.functions, :costgradient)
        _Y = isnothing(Y) ? zero_vector(M, p) : Y
        cost, grad = mfo.functions[:costgradient](M, _Y, p)
        return (cost, real(inner(M, p, X, grad)))
    elseif haskey(mfo.functions, :cost) && haskey(mfo.functions, :gradient)
        cost = mfo.functions[:cost](M, p)
        _Y = isnothing(Y) ? zero_vector(M, p) : Y
        grad = mfo.functions[:gradient](M, _Y, p)
        return (cost, real(inner(M, p, X, grad)))
    end
    return error("$mfo does not provide a cost and a differential")
end
function get_cost_and_gradient!(
        M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective, p
    )
    haskey(mfo.functions, :costgradient) && (return mfo.functions[:costgradient](M, X, p))
    if haskey(mfo.functions, :cost) && haskey(mfo.functions, :gradient)
        return mfo.functions[:cost](M, p), mfo.functions[:gradient](M, X, p)
    end
    Y = zero_vector(M, p)
    if haskey(mfo.functions, :costdifferential) && haskey(mfo.functions, :gradient)
        return (
            mfo.functions[:costdifferential](M, p, Y)[1], mfo.functions[:gradient](M, X, p),
        )
    end
    return error("$mfo seems to either have no access to a cost or a gradient")
end

function get_cost_function(
        mfo::ManifoldFirstOrderObjective, recursive::Bool = false
    )
    if haskey(mfo.functions, :cost)
        return mfo.functions[:cost]
    else
        return (M, p) -> get_cost(M, mfo, p)
    end
end

function get_differential(
        M::AbstractManifold, mfo::ManifoldFirstOrderObjective, p, X;
        gradient = nothing, evaluated::Bool = false, kwargs...,
    )
    # If we have a differential – evaluate that
    haskey(mfo.functions, :differential) && (return mfo.functions[:differential](M, p, X))
    haskey(mfo.functions, :costdifferential) &&
        (return mfo.functions[:costdifferential](M, p, X)[2])
    # default: inner with gradient
    # (a) we have gradient but it is not evaluated -> eval
    (!evaluated && !isnothing(gradient)) && (get_gradient!(M, gradient, mfo, p))
    # if grad is nothing -> allocated gradient
    isnothing(gradient) && (gradient = get_gradient(M, mfo, p))
    # -> we have a gradient!
    return real(inner(M, p, gradient, X))
end

function get_differential_function(
        mfo::ManifoldFirstOrderObjective, recursive::Bool = false
    )
    if haskey(mfo.functions, :differential)
        return mfo.functions[:differential]
    else
        return (M, p, X; kwargs...) -> get_differential(M, mfo, p, X, kwargs...)
    end
end
function get_gradient!(
        M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{<:NamedTuple}, p,
    )
    haskey(mfo.functions, :gradient) && (return mfo.functions[:gradient](M, X, p))
    haskey(mfo.functions, :costgradient) && (return mfo.functions[:costgradient](M, X, p)[2])
    return error("$mfo does not seem to provide a gradient")
end

function get_gradient_function(
        mfo::ManifoldFirstOrderObjective, recursive = false
    )
    haskey(mfo.functions, :gradient) && (return mfo.functions[:gradient])
    return (M, X, p) -> get_gradient!(M, X, mfo, p)
end

function status_summary(mfo::ManifoldFirstOrderObjective; context::Symbol = :default)
    _is_inline(context) && (return repr(mfo))
    return "A first order objective with $(length(mfo.functions)) provided functions.\n\n" * join([ "* $k:$(_MANOPT_INDENT) $(v)" for (k, v) in zip(keys(mfo.functions), mfo.functions) ], "\n")
end
function Base.show(io::IO, mfo::ManifoldFirstOrderObjective)
    print(io, "ManifoldFirstOrderObjective(; ")
    print(io, join([ "$k = $v" for (k, v) in zip(keys(mfo.functions), mfo.functions)], ", "))
    print(io, ", ")
    return print(io, ")")
end


@doc """
    ManifoldProximalMapObjective{TC, TP, V <: Vector{<:Integer}} <: AbstractManifoldCostObjective{E, TC}

specify a problem for solvers based on the evaluation of proximal maps,
which represents proximal maps ``$(_tex(:prox))_{λf_i}`` for summands ``f = f_1 + f_2+ … + f_N`` of the cost function ``f``.

# Fields

* `cost`: a function ``f:$(_math(:Manifold))→ℝ`` to
  minimize
* `proxes`: proximal maps ``$(_tex(:prox))_{λf_i}:$(_math(:Manifold)) → $(_math(:Manifold))``
  as functions `(M, λ, p) -> q` or in-place `(M, q, λ, p)`.
* `number_of_proxes`: number of proximal maps per function,
  to specify when one of the maps is a combined one such that the proximal maps
  functions return more than one entry per function, you have to adapt this value.
  if not specified, it is set to one prox per function.

# Constructor

    ManifoldProximalMapObjective( f, proxes_f::Union{Tuple,AbstractVector}, number_of_proxes=onex(length(proxes)) )

Generate a proximal problem with a tuple or vector of functions, where by default every function computes a single prox
of one component of ``f``.

    ManifoldProximalMapObjective(f, prox_f)

Generate a proximal objective for ``f`` and its proxial map ``$(_tex(:prox))_{λf}``

# See also

[`cyclic_proximal_point`](@ref), [`get_cost`](@ref), [`get_proximal_map`](@ref)
"""
mutable struct ManifoldProximalMapObjective{TC, TP, V} <: AbstractManifoldCostObjective{TC}
    cost::TC
    proximal_maps!!::TP
    number_of_proxes::V
    function ManifoldProximalMapObjective(f, proxes_f::Union{Tuple, AbstractVector})
        np = ones(length(proxes_f))
        return new{typeof(f), typeof(proxes_f), typeof(np)}(
            f, proxes_f, np
        )
    end
    function ManifoldProximalMapObjective(
            f::F, proxes_f::Union{Tuple, AbstractVector}, nOP::Vector{<:Integer}
        ) where {F}
        return if length(nOP) != length(proxes_f)
            throw(
                ErrorException(
                    "The number_of_proxes ($(nOP)) has to be the same length as the number of proxes ($(length(proxes_f)).",
                ),
            )
        else
            new{F, typeof(proxes_f), typeof(nOP)}(f, proxes_f, nOP)
        end
    end
    function ManifoldProximalMapObjective(f::F, prox_f::PF) where {F, PF}
        i = 1
        return new{F, PF, typeof(i)}(f, prox_f, i)
    end
end
function _check_prox_number(pf::Union{Tuple, Vector}, i)
    n = length(pf)
    (i > n) && throw(ErrorException("the $(i)th entry does not exists, only $n available."))
    return true
end

@doc """
    q = get_proximal_map(M::AbstractManifold, mpo::ManifoldProximalMapObjective, λ, p)
    get_proximal_map!(M::AbstractManifold, q, mpo::ManifoldProximalMapObjective, λ, p)
    q = get_proximal_map(M::AbstractManifold, mpo::ManifoldProximalMapObjective, λ, p, i)
    get_proximal_map!(M::AbstractManifold, q, mpo::ManifoldProximalMapObjective, λ, p, i)

evaluate the (`i`th) proximal map of the [`ManifoldProximalMapObjective`](@ref)` mpo` at
the point `p` of `M` with parameter ``λ>0``.
"""
get_proximal_map(::AbstractManifold, ::ManifoldProximalMapObjective, ::Any...)

function get_proximal_map(
        M::AbstractManifold,
        mpo::ManifoldProximalMapObjective{InplaceEvaluation, F, <:Union{<:Tuple, <:Vector}},
        λ, p, i,
    ) where {F}
    _check_prox_number(mpo.proximal_maps!!, i)
    q = allocate_result(M, get_proximal_map, p)
    mpo.proximal_maps!![i](M, q, λ, p)
    return q
end
function get_proximal_map!(
        M::AbstractManifold, q, mpo::ManifoldProximalMapObjective{F, <:Union{<:Tuple, <:Vector}},
        λ, p, i,
    ) where {F}
    _check_prox_number(mpo.proximal_maps!!, i)
    mpo.proximal_maps!![i](M, q, λ, p)
    return q
end

function get_proximal_map(
        M::AbstractManifold, mpo::ManifoldProximalMapObjective, λ, p
    )
    q = allocate_result(M, get_proximal_map, p)
    mpo.proximal_maps!!(M, q, λ, p)
    return q
end
function get_proximal_map!(
        M::AbstractManifold, q, mpo::ManifoldProximalMapObjective, λ, p
    )
    return mpo.proximal_maps!!(M, q, λ, p)
end
function status_summary(mpo::ManifoldProximalMapObjective; context::Symbol = :default)
    (context === :short) && (return repr(mpo))
    return "A proximal map objective for a cost with $(mpo.number_of_proxes) proximal maps"
end
function Base.show(io::IO, mpo::ManifoldProximalMapObjective)
    print(io, "ManifoldProximalMapObjective(", mpo.cost, ", ", mpo.proximal_maps!!, ", ")
    print(io, mpo.number_of_proxes)
    return print(io, ")")
end


@doc """
    ManifoldProximalGradientObjective{TC, TG, TGG, TP} <: AbstractManifoldObjective{TC,TGG}

Model an objective of the form

```math
f(p) = g(p) + h(p), $(_tex(:qquad)) p ∈ $(_math(:Manifold)),
```

where ``g: $(_math(:Manifold)) → $(_tex(:eR))`` is a differentiable function
and ``h: → $(_tex(:eR))`` is a (possibly) lower semicontinous, and proper function.

This objective provides the total cost ``f``, its smooth component ``g``,
as well as ``$(_tex(:grad)) g`` and ``$(_tex(:prox))_{λ h}``.

# Fields

* `cost`: the overall cost ``f = g + h``
* `cost_smooth`: the smooth cost component ``g``
* `gradient_g!!`: the gradient ``$(_tex(:grad)) g``
* `proximal_map_h!!`: the proximal map ``$(_tex(:prox))_{λ h}``

# Constructor
    ManifoldProximalGradientObjective(f, g, grad_g, prox_h)

Generate the proximal gradient objective given the total cost ``f = g + h``, smooth cost ``g``, the gradient of the smooth component ``$(_tex(:grad)) g``, and the proximal map of the nonsmooth component ``$(_tex(:prox))_{λ h}``.
"""
struct ManifoldProximalGradientObjective{TC, TG, TGG, TP} <: AbstractManifoldCostObjective{TC}
    cost::TC # f = g + h
    cost_smooth::TG # smooth part
    gradient_g!!::TGG
    proximal_map_h!!::TP
    function ManifoldProximalGradientObjective(
            f::TC, g::TG, grad_g::TGG, prox_h::TP
        ) where {TC, TG, TGG, TP}
        return new{TC, TG, TGG, TP}(f, g, grad_g, prox_h)
    end
end

"""
    get_gradient(M::AbstractManifold, mgo::ManifoldProximalGradientObjective, p)
    get_gradient!(M::AbstractManifold, X, mgo::ManifoldProximalGradientObjective, p)

Evaluate the gradient of the smooth part of a [`ManifoldProximalGradientObjective`](@ref) `mgo` at `p`.
"""
get_gradient(::AbstractManifold, ::ManifoldProximalGradientObjective, p)

function get_gradient!(M::AbstractManifold, X, mpgo::ManifoldProximalGradientObjective, p)
    return mpgo.gradient_g!!(M, X, p)
end

function Base.show(io::IO, mpgo::ManifoldProximalGradientObjective{E}) where {E}
    print(io, "ManifoldProximalGradientObjective(", mpgo.cost, ", ", mpgo.cost_smooth, ", ")
    print(io, mpgo.gradient_g!!, ", ", mpgo.proximal_map_h!!)
    return print(io, ")")
end

function status_summary(mpgo::ManifoldProximalGradientObjective; context::Symbol = :default)
    (context === :short) && return repr(mpgo)
    s = "A proximal gradient objective `f = g + h`, where `g` is smooth and `h` is possibly nonsmooth."
    (context === :inline) && (return s)
    return """
    $s

    # Components
    * `f`:          $(mpgo.cost)
    * `g`:          $(mpgo.cost_smooth)
    * `gradient_g`: $(mpgo.gradient_g!!)
    * `prox_h`:     $(mpgo.proximal_map_h!!)"""
end
"""
    get_cost_smooth(M::AbstractManifold, objective, p)

Helper function to extract the smooth part `g` of a proximal gradient objective at the point `p`.
"""
function get_cost_smooth(
        M::AbstractManifold, objective::ManifoldProximalGradientObjective, p
    )
    return objective.cost_smooth(M, p)
end

@doc """
    q = get_proximal_map(M::AbstractManifold, mpo::ManifoldProximalGradientObjective, λ, p)
    get_proximal_map!(M::AbstractManifold, q, mpo::ManifoldProximalGradientObjective, λ, p)

Evaluate proximal map of the nonsmooth component ``h`` of the [`ManifoldProximalGradientObjective`](@ref)` mpo`
at the point `p` on `M` with parameter ``λ>0``.
"""
get_proximal_map(M::AbstractManifold, mpgo::ManifoldProximalGradientObjective, λ, p)

function get_proximal_map!(
        M::AbstractManifold, q, mpgo::ManifoldProximalGradientObjective, λ, p
    )
    return mpgo.proximal_map_h!!(M, q, λ, p)
end

@doc """
    ManifoldSubgradientObjective{T<:AbstractEvaluationType,C,S} <:AbstractManifoldCostObjective{T, C}

A structure to store information about a objective for a subgradient based optimization problem

# Fields

* `cost`:        the function ``f`` to be minimized
* `subgradient`: a function returning a subgradient ``∂f`` of ``f``

# Constructor

    ManifoldSubgradientObjective(f, ∂f)

Generate the [`ManifoldSubgradientObjective`](@ref) for a subgradient objective, consisting
of a (cost) function `f(M, p)` and a function `∂f(M, p)` that returns a not necessarily
deterministic element from the subdifferential at `p` on a manifold `M`.
"""
struct ManifoldSubgradientObjective{C, S} <: AbstractManifoldCostObjective{C}
    cost::C
    subgradient!!::S
    function ManifoldSubgradientObjective(cost::C, subgrad::S) where {C, S}
        return new{C, S}(cost, subgrad)
    end
end

"""
    X = get_subgradient(M;;AbstractManifold, sgo::ManifoldSubgradientObjective, p)
    get_subgradient!(M;;AbstractManifold, X, sgo::ManifoldSubgradientObjective, p)

Evaluate the (sub)gradient of a [`ManifoldSubgradientObjective`](@ref) `sgo`
at the point `p`.

The evaluation is done in place of `X` for the `!`-variant.
The result might not be deterministic, _one_ element of the subdifferential is returned.
"""
function get_subgradient(M::AbstractManifold, sgo::ManifoldSubgradientObjective, p)
    X = zero_vector(M, p)
    return sgo.subgradient!!(M, X, p)
end
function get_subgradient!(
        M::AbstractManifold, X, sgo::ManifoldSubgradientObjective, p
    )
    return sgo.subgradient!!(M, X, p)
end

@doc """
    get_subgradient_function(objective::ManifoldSubgradientObjective, recursive=false)

return the function to evaluate (just) the gradient ``$(_tex(:grad)) f(p)``
and is of the form `(M, X, p) -> X` to work in-place of `X`,
where either the gradient function using the decorator or without the decorator is used.

By default `recursive` is set to `false`, since usually to just pass the gradient function
somewhere, one still wants for example the cached one or the one that still counts calls.
"""
function get_subgradient_function(objective::ManifoldSubgradientObjective, recursive = false)
    return objective.subgradient!!
end

function Base.show(io::IO, objective::ManifoldSubgradientObjective)
    return print(io, "ManifoldSubgradientObjective(", objective.cost, ", ", objective.subgradient!!, ")")
end

function status_summary(objective::ManifoldSubgradientObjective; context::Symbol = :default)
    (context === :short) && return repr(objective)
    s = "A subgradient objective "
    (context === :inline) && (return s)
    return """
    $s

    ## Components
    * `f`:  $(objective.cost)
    * `∂f`: $(objective.subgradient!!)"""
end
