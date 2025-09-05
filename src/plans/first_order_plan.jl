
@doc """
    AbstractManifoldFirstOrderObjective{E<:AbstractEvaluationType, FGD} <: AbstractManifoldCostObjective{E, FGD}

An abstract type for all objectives that provide
* a cost
* first order information, so either a (full) gradient or a differential, where
`E` is a [`AbstractEvaluationType`](@ref) for the gradient function.
"""
abstract type AbstractManifoldFirstOrderObjective{E<:AbstractEvaluationType,FGD} <:
              AbstractManifoldCostObjective{E,FGD} end

@doc """
    ManifoldFirstOrderObjective{E<:AbstractEvaluationType, F} <: AbstractManifoldFirstOrderObjective{E, F}

specify an objective containing a cost and its gradient or differential,
where the [`AbstractEvaluationType`](@ref) `E` indicates the type of evaluation for a gradient.

# Fields

* `functions::F`: a function or a tuple of functions containing the cost and first order information.

Currently the following cases are covered, sorted by their popularity

1. a single function `fg`, i.e. a function or a functor, represents a combined
    function `(M, p) -> (c, X)` that computes the cost `c=cost(M,p)` and gradient `X=grad_f(M,p)`;
2. a single function `fdf`, i.e. a function or a functor, represents a combined function
    `(M, p) -> (c, d)` that computes the cost `c=cost(M,p)` and differential `d=diff_f(M,p)`;
3. pairs of single functions `(f, g)`, `(f, df)` of a cost function `f` and either its
    gradient `g` or its differential `d`, respectively
4. The function `(fg, d)` and `(fdf, g)`  from 1 and 2, respectively joined by
    the other missing third information, the differential for the first or the gradient for the second
5. a tuple `(f, g, d)` of three functions, computing cost, `f`, gradient `g`,
    and `differential `d` separately
6. a `(f, gd)` of a cost function and a combined function `(X, d) = gd(M, p, X)`
    to compute gradient and differential together
7. a single function `(c, X, d) = fgd(M, p,X)`

For all cases where a gradient is present, also an in-place variant is possible, where the
signature has the result `Y` in second place.

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
$(_var(:Keyword, :evaluation))

Where:
 * At least one of `cost`, `costgradient` or `costdifferential` must be provided.
 * Either `gradient`, `costgradient`, `differential` or `costdifferential` must be provided.
 * If more than one function provides the same thing (e.g. cost), it is assumed that all
   such functions return the same value. Optimization algorithms will attempt to make the
   most efficient use of provided functions.

# Used with
[`gradient_descent`](@ref), [`conjugate_gradient_descent`](@ref), [`quasi_Newton`](@ref)
"""
struct ManifoldFirstOrderObjective{E<:AbstractEvaluationType,F<:NamedTuple} <:
       AbstractManifoldFirstOrderObjective{E,F}
    functions::F
end

# A Monster constructor
function ManifoldFirstOrderObjective(;
    cost=nothing,
    differential=nothing,
    gradient=nothing,
    costgradient=nothing,
    costdifferential=nothing,
    evaluation::E=AllocatingEvaluation(),
) where {E<:AbstractEvaluationType}
    nc = isnothing(cost)
    nd = isnothing(differential)
    ng = isnothing(gradient)
    ncg = isnothing(costgradient)
    ncd = isnothing(costdifferential)

    if nc && ncg && ncd
        throw(
            ArgumentError(
                "Either cost, costgradient or costdifferential keyword argument needs to be provided",
            ),
        )
    end
    if ng && ncg && nd && ncd
        throw(
            ArgumentError(
                "Either gradient, costgradient, differential or costdifferential keyword argument needs to be provided",
            ),
        )
    end

    nt = (;)
    if !nc
        nt = merge(nt, (; cost=cost))
    end
    if !ng
        nt = merge(nt, (; gradient=gradient))
    end
    if !nd
        nt = merge(nt, (; differential=differential))
    end
    if !ncg
        nt = merge(nt, (; costgradient=costgradient))
    end
    if !ncd
        nt = merge(nt, (; costdifferential=costdifferential))
    end

    return ManifoldFirstOrderObjective{E,typeof(nt)}(nt)
end

const ManifoldGradientObjective{E,F,G} = ManifoldFirstOrderObjective{
    E,
    <:Union{
        NamedTuple{Tuple{:cost,:gradient},Tuple{F,G}},
        NamedTuple{Tuple{:cost,:gradient,:differential},Tuple{F,G,D where D}},
    },
}
@doc """
    ManifoldGradientObjective(cost, gradient; evaluation::E=AllocatingEvaluation() kwargs...)

Generate an objective with a function `cost` and its `gradient`.
Depending on the [`AbstractEvaluationType`](@ref) `E` the gradient can have to forms

* as a function `(M, p) -> X` that allocates memory for `X`, an [`AllocatingEvaluation`](@ref)
* as a function `(M, X, p) -> X` that work in place of `X`, an [`InplaceEvaluation`](@ref)

Internally this is stored in a [`ManifoldFirstOrderObjective`](@ref). The `kwargs...`
are also passed to this representation, which allows to add a special function
to evaluate the `differential`.

# Used with
[`gradient_descent`](@ref), [`conjugate_gradient_descent`](@ref), [`quasi_Newton`](@ref)
"""
function ManifoldGradientObjective(cost, grad; kwargs...)
    return ManifoldFirstOrderObjective(; cost=cost, gradient=grad, kwargs...)
end

const ManifoldCostGradientObjective{E,FG} = ManifoldFirstOrderObjective{
    E,
    <:Union{
        NamedTuple{Tuple{:costgradient},Tuple{FG}},
        NamedTuple{Tuple{:costgradient,:differential},Tuple{FG,D where D}},
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
    return ManifoldFirstOrderObjective(; costgradient=cost_grad, kwargs...)
end

#
# get_cost
function get_cost(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{AllocatingEvaluation}, p
)
    haskey(mfo.functions, :cost) && (return mfo.functions[:cost](M, p))
    if haskey(mfo.functions, :costdifferential)
        X = zero_vector(M, p)
        return mfo.functions[:costdifferential](M, p, X)[1]
    end
    haskey(mfo.functions, :costgradient) && (return mfo.functions[:costgradient](M, p)[1])

    return error("$mfo does not seem to provide a cost")
end
function get_cost(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{InplaceEvaluation}, p
)
    haskey(mfo.functions, :cost) && (return mfo.functions[:cost](M, p))
    X = zero_vector(M, p)
    haskey(mfo.functions, :costgradient) && return mfo.functions[:costgradient](M, X, p)[1]
    if haskey(mfo.functions, :costdifferential)
        return mfo.functions[:costdifferential](M, p, X)[1]
    end
    return error("$mfo does not seem to provide a cost")
end

# get_cost_and_differential

function get_cost_and_differential(
    M::AbstractManifold,
    mfo::ManifoldFirstOrderObjective{AllocatingEvaluation},
    p,
    X;
    kwargs...,
)
    if haskey(mfo.functions, :costdifferential)
        return mfo.functions[:costdifferential](M, p, X)
    elseif haskey(mfo.functions, :cost) && haskey(mfo.functions, :differential)
        return (mfo.functions[:cost](M, p), mfo.functions[:differential](M, p, X))
    elseif haskey(mfo.functions, :costgradient)
        cost, grad = mfo.functions[:costgradient](M, p)
        return (cost, real(inner(M, p, X, grad)))
    elseif haskey(mfo.functions, :cost) && haskey(mfo.functions, :gradient)
        cost = mfo.functions[:cost](M, p)
        grad = mfo.functions[:gradient](M, p)
        return (cost, real(inner(M, p, X, grad)))
    end
    return error("$mfo does not provide a cost and a differential")
end
function get_cost_and_differential(
    M::AbstractManifold,
    mfo::ManifoldFirstOrderObjective{InplaceEvaluation},
    p,
    X;
    Y=nothing,
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

# On problems -> “unpack”
function get_cost_and_differential(amp::AbstractManoptProblem, p, X; kwargs...)
    return get_cost_and_differential(get_manifold(amp), get_objective(amp), p, X; kwargs...)
end

function get_cost_and_differential(
    M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p, X; kwargs...
)
    return get_cost_and_differential(M, get_objective(admo, false), p, X; kwargs...)
end

# general: Generate a separate cost
function get_cost_function(
    mfo::ManifoldFirstOrderObjective{AllocatingEvaluation}, recursive::Bool=false
)
    if haskey(mfo.functions, :cost)
        return mfo.functions[:cost]
    else
        return (M, p) -> get_cost(M, mfo, p)
    end
end
function get_cost_function(
    mfo::ManifoldFirstOrderObjective{InplaceEvaluation}, recursive::Bool=false
)
    if haskey(mfo.functions, :cost)
        return mfo.functions[:cost]
    else
        return (M, p) -> get_cost(M, mfo, p)
    end
end

# Differential - passthrough
function get_differential(
    M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p, X; kwargs...
)
    return get_differential(M, get_objective(admo, false), p, X; kwargs...)
end
# On problems -> “unpack”
function get_differential(amp::AbstractManoptProblem, p, X; kwargs...)
    return get_differential(get_manifold(amp), get_objective(amp), p, X; kwargs...)
end
"""
     get_differential(amp::AbstractManoptProblem, p, X; kwargs...)
     get_differential(M::AbstractManifold, amfo:AbstractManifoldFirstOrderObjective, p, X; kwargs...)
     get_differential(M::AbstractManifold, amfo:AbstractDecoratedManifoldObjective, p, X; kwargs...)

Evaluate the differential ``Df(p)[X]`` of the function ``f`` represented by
the [`AbstractManifoldFirstOrderObjective`](@ref).
For [`AbstractManoptProblem`](@ref) the inner manifold and objectives are used,
similarly, any objective decorator would “pass though” to its inner objective.
By default this falls back to ``Df(p)[X] = ⟨$(_tex(:grad))f(p), X⟩

# Keyword arguments
* `gradient=nothing` – pass a tangent vector to be used internally as interims memory,
  e.g. in the default variant to evaluate the gradient in-place in.
* `evaluated=false` – indicate whether `gradient` is just memory (`false`, default) or
  already contains the evaluated gradient (`true`).
"""
function get_differential(
    M::AbstractManifold,
    amfo::AbstractManifoldFirstOrderObjective,
    p,
    X;
    gradient=nothing,
    evaluated::Bool=false,
)
    isnothing(gradient) && (return real(inner(M, p, get_gradient(M, amfo, p), X)))
    # if it is not nothing call in-place
    (!evaluated) && (get_gradient!(M, gradient, amfo, p))
    return real(inner(M, p, gradient, X))
end
function get_differential(
    M::AbstractManifold,
    mfo::ManifoldFirstOrderObjective,
    p,
    X;
    gradient=nothing,
    evaluated::Bool=false,
    kwargs...,
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
# Differential function - pass-through
function get_differential_function(
    admo::AbstractDecoratedManifoldObjective, recursive=false
)
    return get_differential_function(get_objective(admo, recursive))
end

@doc """
     get_differential_function(admo::AbstractManifoldFirstOrderObjective, recursive::Bool=false)

Return the function to evaluate (just) the differential ``Df(p)[X]``.
For a decorated objective, the `recursive` positional parameter determines whether to
directly call this function on the next decorator or whether to get the “most inner” objective.
"""
get_differential_function(::AbstractManifoldFirstOrderObjective; recursive::Bool=false)

function get_differential_function(
    mfo::ManifoldFirstOrderObjective{<:AbstractEvaluationType}, recursive::Bool=false
)
    if haskey(mfo.functions, :differential)
        return mfo.functions[:differential]
    else
        return (M, p, X; kwargs...) -> get_differential(M, mfo, p, X, kwargs...)
    end
end

# Decorator case
function get_gradient(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p)
    return get_gradient(M, get_objective(admo, false), p)
end
function get_gradient!(M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p)
    return get_gradient!(M, X, get_objective(admo, false), p)
end

# (a) alloc
function get_gradient(
    M::AbstractManifold,
    mfo::ManifoldFirstOrderObjective{AllocatingEvaluation,<:NamedTuple},
    p,
)
    haskey(mfo.functions, :gradient) && (return mfo.functions[:gradient](M, p))
    haskey(mfo.functions, :costgradient) && (return mfo.functions[:costgradient](M, p)[2])
    return error("$mfo does not seem to provide a gradient")
end
function get_gradient!(
    M::AbstractManifold,
    X,
    mfo::ManifoldFirstOrderObjective{AllocatingEvaluation,<:NamedTuple},
    p,
)
    haskey(mfo.functions, :gradient) &&
        (return copyto!(M, X, p, mfo.functions[:gradient](M, p)))
    haskey(mfo.functions, :costgradient) &&
        (return copyto!(M, X, p, mfo.functions[:costgradient](M, p)[2]))
    return error("$mfo does not seem to provide a gradient")
end
# (b) inplace
function get_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{InplaceEvaluation,<:NamedTuple}, p
)
    X = zero_vector(M, p)
    return get_gradient!(M, X, mfo, p)
end
function get_gradient!(
    M::AbstractManifold,
    X,
    mfo::ManifoldFirstOrderObjective{InplaceEvaluation,<:NamedTuple},
    p,
)
    haskey(mfo.functions, :gradient) && (return mfo.functions[:gradient](M, X, p))
    haskey(mfo.functions, :costgradient) &&
        (return mfo.functions[:costgradient](M, X, p)[2])
    return error("$mfo does not seem to provide a gradient")
end

@doc """
    get_gradient_function(amgo::AbstractManifoldFirstOrderObjective, recursive=false)

return the function to evaluate (just) the gradient ``$(_tex(:grad)) f(p)``,
where either the gradient function using the decorator or without the decorator is used.

By default `recursive` is set to `false`, since usually to just pass the gradient function
somewhere, one still wants for example the cached one or the one that still counts calls.

Depending on the [`AbstractEvaluationType`](@ref) `E` this is a function

* `(M, p) -> X` for the [`AllocatingEvaluation`](@ref) case
* `(M, X, p) -> X` for the [`InplaceEvaluation`](@ref) working in-place of `X`.
"""
get_gradient_function(::AbstractManifoldFirstOrderObjective; recursive=false)

function get_gradient_function(admo::AbstractDecoratedManifoldObjective, recursive=false)
    return get_gradient_function(get_objective(admo, recursive))
end
function get_gradient_function(
    mfo::ManifoldFirstOrderObjective{AllocatingEvaluation,<:NamedTuple}, recursive=false
)
    haskey(mfo.functions, :gradient) && (return mfo.functions[:gradient])
    return (M, p) -> get_gradient(M, mfo, p)
end
function get_gradient_function(
    mfo::ManifoldFirstOrderObjective{InplaceEvaluation}, recursive=false
)
    haskey(mfo.functions, :gradient) && (return mfo.functions[:gradient])
    return (M, X, p) -> get_gradient!(M, X, mfo, p)
end

#
#  Access cost and gradient – a bit of cases
# -----------------------------
# 0: General decorators
function get_cost_and_gradient(
    M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p
)
    return get_cost_and_gradient(M, get_objective(admo, false), p)
end
function get_cost_and_gradient!(
    M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p
)
    return get_cost_and_gradient!(M, X, get_objective(admo, false), p)
end

# Case 1, FG (a) alloc
function get_cost_and_gradient(
    M::AbstractManifold,
    mfo::ManifoldFirstOrderObjective{AllocatingEvaluation,<:NamedTuple},
    p,
)
    haskey(mfo.functions, :costgradient) && (return mfo.functions[:costgradient](M, p))
    if haskey(mfo.functions, :cost) && haskey(mfo.functions, :gradient)
        return (mfo.functions[:cost](M, p), mfo.functions[:gradient](M, p))
    end
    if haskey(mfo.functions, :costdifferential) && haskey(mfo.functions, :gradient)
        Y = zero_vector(M, p)
        return (
            mfo.functions[:costdifferential](M, p, Y)[1], mfo.functions[:gradient](M, p)
        )
    end
    return error("$mfo seems to either have no access to a cost or a gradient")
end
function get_cost_and_gradient!(
    M::AbstractManifold,
    X,
    mfo::ManifoldFirstOrderObjective{AllocatingEvaluation,<:NamedTuple},
    p,
)
    if haskey(mfo.functions, :costgradient)
        c, Y = mfo.functions[:costgradient](M, p)
        copyto!(M, X, p, Y)
        return c, X
    end
    if haskey(mfo.functions, :cost) && haskey(mfo.functions, :gradient)
        copyto!(M, X, p, mfo.functions[:gradient](M, p))
        return mfo.functions[:cost](M, p), X
    end
    Y = zero_vector(M, p)
    if haskey(mfo.functions, :costdifferential) && haskey(mfo.functions, :gradient)
        copyto!(M, X, p, mfo.functions[:gradient](M, p))
        return (mfo.functions[:costdifferential](M, p, Y)[1], X)
    end

    return error("$mfo seems to either have no access to a cost or a gradient")
end
function get_cost_and_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{InplaceEvaluation}, p
)
    X = zero_vector(M, p)
    return get_cost_and_gradient!(M, X, mfo, p)
end
function get_cost_and_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{InplaceEvaluation}, p
)
    haskey(mfo.functions, :costgradient) && (return mfo.functions[:costgradient](M, X, p))
    if haskey(mfo.functions, :cost) && haskey(mfo.functions, :gradient)
        return mfo.functions[:cost](M, p), mfo.functions[:gradient](M, X, p)
    end
    Y = zero_vector(M, p)
    if haskey(mfo.functions, :costdifferential) && haskey(mfo.functions, :gradient)
        return (
            mfo.functions[:costdifferential](M, p, Y)[1], mfo.functions[:gradient](M, X, p)
        )
    end

    return error("$mfo seems to either have no access to a cost or a gradient")
end
function show(io::IO, ::ManifoldFirstOrderObjective{E,FG}) where {E,FG}
    return print(io, "ManifoldFirstOrderObjective{$E, $FG}")
end

#
#  Access gradient
# -----------------------------

@doc """
    get_gradient(amp::AbstractManoptProblem, p)
    get_gradient!(amp::AbstractManoptProblem, X, p)

evaluate the gradient of an [`AbstractManoptProblem`](@ref) `amp` at the point `p`.

The evaluation is done in place of `X` for the `!`-variant.
"""
function get_gradient(mp::AbstractManoptProblem, p)
    return get_gradient(get_manifold(mp), get_objective(mp), p)
end
function get_gradient!(mp::AbstractManoptProblem, X, p)
    return get_gradient!(get_manifold(mp), X, get_objective(mp), p)
end

"""
    X = get_subgradient(M::AbstractManifold, sgo::AbstractManifoldFirstOrderObjective, p)
    get_subgradient!(M::AbstractManifold, X, sgo::AbstractManifoldFirstOrderObjective, p)

Evaluate the subgradient, which for the case of a objective having a gradient, means evaluating the
gradient itself.

While in general, the result might not be deterministic, for this case it is.
"""
function get_subgradient(M::AbstractManifold, agmo::AbstractManifoldFirstOrderObjective, p)
    return get_gradient(M, agmo, p)
end
function get_subgradient!(
    M::AbstractManifold, X, agmo::AbstractManifoldFirstOrderObjective, p
)
    return get_gradient!(M, X, agmo, p)
end

@doc """
    get_gradient(agst::AbstractGradientSolverState)

return the gradient stored within gradient options.
THe default returns `agst.X`.
"""
get_gradient(agst::AbstractGradientSolverState) = agst.X

@doc """
    set_gradient!(agst::AbstractGradientSolverState, M, p, X)

set the (current) gradient stored within an [`AbstractGradientSolverState`](@ref) to `X`.
The default function modifies `s.X`.
"""
function set_gradient!(agst::AbstractGradientSolverState, M, p, X)
    copyto!(M, agst.X, p, X)
    return agst
end

@doc """
    get_iterate(agst::AbstractGradientSolverState)

return the iterate stored within gradient options.
THe default returns `agst.p`.
"""
get_iterate(agst::AbstractGradientSolverState) = agst.p

@doc """
    set_iterate!(agst::AbstractGradientSolverState, M, p)

set the (current) iterate stored within an [`AbstractGradientSolverState`](@ref) to `p`.
The default function modifies `s.p`.
"""
function set_iterate!(agst::AbstractGradientSolverState, M, p)
    copyto!(M, agst.p, p)
    return agst
end

"""
    DirectionUpdateRule

A general functor, that handles direction update rules. It's fields are usually
only a [`StoreStateAction`](@ref) by default initialized to the fields required
for the specific coefficient, but can also be replaced by a (common, global)
individual one that provides these values.
"""
abstract type DirectionUpdateRule end

"""
    IdentityUpdateRule <: DirectionUpdateRule

The default gradient direction update is the identity, usually it just evaluates the gradient.

You can also use `Gradient()` to create the corresponding factory, though this only delays
this parameter-free instantiation to later.
"""
struct IdentityUpdateRule <: DirectionUpdateRule end
Gradient() = ManifoldDefaultsFactory(Manopt.IdentityUpdateRule; requires_manifold=false)

"""
    MomentumGradientRule <: DirectionUpdateRule

Store the necessary information to compute the [`MomentumGradient`](@ref)
direction update.

# Fields

$(_var(:Field, :p, "p_old"))
* `momentum::Real`: factor for the momentum
* `direction`: internal [`DirectionUpdateRule`](@ref) to determine directions
  to add the momentum to.
$(_var(:Field, :vector_transport_method))
$(_var(:Field, :X, "X_old"))

# Constructors


    MomentumGradientRule(M::AbstractManifold; kwargs...)

Initialize a momentum gradient rule to `s`, where `p` and `X` are memory for interim values.

## Keyword arguments

$(_var(:Keyword, :p))
* `s=`[`IdentityUpdateRule`](@ref)`()`
* `momentum=0.2`
$(_var(:Keyword, :vector_transport_method))
$(_var(:Keyword, :X))


# See also
[`MomentumGradient`](@ref)
"""
mutable struct MomentumGradientRule{
    P,T,D<:DirectionUpdateRule,R<:Real,VTM<:AbstractVectorTransportMethod
} <: DirectionUpdateRule
    momentum::R
    p_old::P
    direction::D
    vector_transport_method::VTM
    X_old::T
end
function MomentumGradientRule(
    M::AbstractManifold;
    p::P=rand(M),
    direction::Union{<:DirectionUpdateRule,ManifoldDefaultsFactory}=Gradient(),
    vector_transport_method::VTM=default_vector_transport_method(M, typeof(p)),
    X::Q=zero_vector(M, p),
    momentum::F=0.2,
) where {P,Q,F<:Real,VTM<:AbstractVectorTransportMethod}
    dir = _produce_type(direction, M)
    return MomentumGradientRule{P,Q,typeof(dir),F,VTM}(
        momentum, p, dir, vector_transport_method, X
    )
end
function (mg::MomentumGradientRule)(
    mp::AbstractManoptProblem, s::AbstractGradientSolverState, k
)
    M = get_manifold(mp)
    p = get_iterate(s)
    step, dir = mg.direction(mp, s, k) #get inner direction and step size
    mg.X_old =
        mg.momentum *
        vector_transport_to(M, mg.p_old, mg.X_old, p, mg.vector_transport_method) -
        step .* dir
    copyto!(M, mg.p_old, p)
    return step, -mg.X_old
end

"""
    MomentumGradient()

Append a momentum to a gradient processor, where the last direction and last iterate are
stored and the new is composed as ``η_i = m*η_{i-1}' - s d_i``,
where ``sd_i`` is the current (inner) direction and ``η_{i-1}'`` is the vector transported
last direction multiplied by momentum ``m``.

# Input

* `M` (optional)

# Keyword arguments

$(_var(:Keyword, :p))
* `direction=`[`IdentityUpdateRule`](@ref) preprocess the actual gradient before adding momentum
$(_var(:Keyword, :X))
* `momentum=0.2` amount of momentum to use
$(_var(:Keyword, :vector_transport_method))

$(_note(:ManifoldDefaultFactory, "MomentumGradientRule"))
"""
function MomentumGradient(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.MomentumGradientRule, args...; kwargs...)
end

"""
    AverageGradientRule <: DirectionUpdateRule

Add an average of gradients to a gradient processor. A set of previous directions (from the
inner processor) and the last iterate are stored. The average is taken after vector transporting
them to the current iterates tangent space.


# Fields

* `gradients`:               the last `n` gradient/direction updates
* `last_iterate`:            last iterate (needed to transport the gradients)
* `direction`:               internal [`DirectionUpdateRule`](@ref) to determine directions to apply the averaging to
$(_var(:Keyword, :vector_transport_method))

# Constructors

    AverageGradientRule(
        M::AbstractManifold;
        p::P=rand(M);
        n::Int=10
        direction::Union{<:DirectionUpdateRule,ManifoldDefaultsFactory}=IdentityUpdateRule(),
        gradients = fill(zero_vector(p.M, o.x),n),
        last_iterate = deepcopy(x0),
        vector_transport_method = default_vector_transport_method(M, typeof(p))
    )

Add average to a gradient problem, where

* `n`:                       determines the size of averaging
* `direction`:               is the internal [`DirectionUpdateRule`](@ref) to determine the gradients to store
* `gradients`:               can be pre-filled with some history
* `last_iterate`:            stores the last iterate
$(_var(:Keyword, :vector_transport_method))
"""
mutable struct AverageGradientRule{
    P,T,D<:DirectionUpdateRule,VTM<:AbstractVectorTransportMethod
} <: DirectionUpdateRule
    gradients::AbstractVector{T}
    last_iterate::P
    direction::D
    vector_transport_method::VTM
end
function AverageGradientRule(
    M::AbstractManifold;
    p::P=rand(M),
    n::Int=10,
    direction::Union{<:DirectionUpdateRule,ManifoldDefaultsFactory}=Gradient(),
    gradients=[zero_vector(M, p) for _ in 1:n],
    vector_transport_method::VTM=default_vector_transport_method(M, typeof(p)),
) where {P,VTM}
    dir = _produce_type(direction, M)
    return AverageGradientRule{P,eltype(gradients),typeof(dir),VTM}(
        gradients, copy(M, p), dir, vector_transport_method
    )
end
function (a::AverageGradientRule)(
    mp::AbstractManoptProblem, s::AbstractGradientSolverState, k
)
    # remove oldest/last
    pop!(a.gradients)
    M = get_manifold(mp)
    p = get_iterate(s)
    _, d = a.direction(mp, s, k) #get inner gradient and step
    for g in a.gradients
        vector_transport_to!(M, g, a.last_iterate, g, p, a.vector_transport_method)
    end
    pushfirst!(a.gradients, copy(M, p, d))
    copyto!(M, a.last_iterate, p)
    return 1.0, 1 / length(a.gradients) .* sum(a.gradients)
end

"""
    AverageGradient(; kwargs...)
    AverageGradient(M::AbstractManifold; kwargs...)

Add an average of gradients to a gradient processor. A set of previous directions (from the
inner processor) and the last iterate are stored, average is taken after vector transporting
them to the current iterates tangent space.

# Input

$(_var(:Argument, :M; type=true)) (optional)

# Keyword arguments

$(_var(:Keyword, :p; add=:as_Initial))
* `direction=`[`IdentityUpdateRule`](@ref) preprocess the actual gradient before adding momentum
* `gradients=[zero_vector(M, p) for _ in 1:n]` how to initialise the internal storage
* `n=10` number of gradient evaluations to take the mean over
$(_var(:Keyword, :X))
$(_var(:Keyword, :vector_transport_method))

$(_note(:ManifoldDefaultFactory, "AverageGradientRule"))
"""
function AverageGradient(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.AverageGradientRule, args...; kwargs...)
end

@doc """
    NesterovRule <: DirectionUpdateRule

Compute a Nesterov inspired direction update rule.
See [`Nesterov`](@ref) for details

# Fields

* `γ::Real`, `μ::Real`: coefficients from the last iterate
* `v::P`:      an interim point to compute the next gradient evaluation point `y_k`
* `shrinkage`: a function `k -> ...` to compute the shrinkage ``β_k`` per iterate `k``.
$(_var(:Keyword, :inverse_retraction_method))

# Constructor

    NesterovRule(M::AbstractManifold; kwargs...)

## Keyword arguments

$(_var(:Keyword, :p; add=:as_Initial))
* `γ=0.001``
* `μ=0.9``
* `shrinkage = k -> 0.8`
$(_var(:Keyword, :inverse_retraction_method))

# See also

[`Nesterov`](@ref)
"""
mutable struct NesterovRule{P,R<:Real} <: DirectionUpdateRule
    γ::R
    μ::R
    v::P
    shrinkage::Function
    inverse_retraction_method::AbstractInverseRetractionMethod
end
function NesterovRule(
    M::AbstractManifold;
    p::P=rand(M),
    γ::T=0.001,
    μ::T=0.9,
    shrinkage::Function=i -> 0.8,
    inverse_retraction_method::AbstractInverseRetractionMethod=default_inverse_retraction_method(
        M, typeof(p)
    ),
) where {P,T}
    p_ = _ensure_mutating_variable(p)
    return NesterovRule{typeof(p_),T}(
        γ, μ, copy(M, p_), shrinkage, inverse_retraction_method
    )
end
function (n::NesterovRule)(mp::AbstractManoptProblem, s::AbstractGradientSolverState, k)
    M = get_manifold(mp)
    h = get_stepsize(mp, s, k)
    p = get_iterate(s)
    α = (h * (n.γ - n.μ) + sqrt(h^2 * (n.γ - n.μ)^2 + 4 * h * n.γ)) / 2
    γbar = (1 - α) * n.γ + α * n.μ
    y = retract(
        M,
        p,
        ((α * n.γ) / (n.γ + α * n.μ)) *
        inverse_retract(M, p, n.v, n.inverse_retraction_method),
    )
    gradf_yk = get_gradient(mp, y)
    xn = retract(M, y, -h * gradf_yk)
    d =
        (((1 - α) * n.γ) / γbar) * inverse_retract(M, y, n.v, n.inverse_retraction_method) -
        (α / γbar) * gradf_yk
    n.v = retract(M, y, d, s.retraction_method)
    n.γ = 1 / (1 + n.shrinkage(k)) * γbar
    return h, (-1 / h) * inverse_retract(M, p, xn, n.inverse_retraction_method) # outer update
end

@doc """
    Nesterov(; kwargs...)
    Nesterov(M::AbstractManifold; kwargs...)

Assume ``f`` is ``L``-Lipschitz and ``μ``-strongly convex. Given

* a step size ``h_k<$(_tex(:frac, "1", "L"))`` (from the [`GradientDescentState`](@ref)
* a `shrinkage` parameter ``β_k``
* and a current iterate ``p_k``
* as well as the interim values ``γ_k`` and ``v_k`` from the previous iterate.

This compute a Nesterov type update using the following steps, see [ZhangSra:2018](@cite)

1. Compute the positive root ``α_k∈(0,1)`` of ``α^2 = h_k$(_tex(:bigl))((1-α_k)γ_k+α_k μ$(_tex(:bigr)))``.
2. Set ``$(_tex(:bar, "γ"))_k+1 = (1-α_k)γ_k + α_kμ``
3. ``y_k = $(_tex(:retr))_{p_k}\\Bigl(\\frac{α_kγ_k}{γ_k + α_kμ}$(_tex(:retr))^{-1}_{p_k}v_k \\Bigr)``
4. ``x_{k+1} = $(_tex(:retr))_{y_k}(-h_k $(_tex(:grad))f(y_k))``
5. ``v_{k+1} = $(_tex(:retr))_{y_k}\\Bigl(\\frac{(1-α_k)γ_k}{$(_tex(:bar, "γ"))_k}$(_tex(:retr))_{y_k}^{-1}(v_k) - \\frac{α_k}{$(_tex(:bar, "γ"))_{k+1}}$(_tex(:grad))f(y_k) \\Bigr)``
6. ``γ_{k+1} = \\frac{1}{1+β_k}$(_tex(:bar, "γ"))_{k+1}``

Then the direction from ``p_k`` to ``p_k+1`` by ``d = $(_tex(:invretr))_{p_k}p_{k+1}`` is returned.

# Input

$(_var(:Argument, :M; type=true)) (optional)

# Keyword arguments

$(_var(:Keyword, :p; add=:as_Initial))
* `γ=0.001`
* `μ=0.9`
* `shrinkage = k -> 0.8`
$(_var(:Keyword, :inverse_retraction_method))

$(_note(:ManifoldDefaultFactory, "NesterovRule"))
"""
function Nesterov(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.NesterovRule, args...; kwargs...)
end

"""
    PreconditionedDirectionRule{E<:AbstractEvaluationType} <: DirectionUpdateRule

Add a preconditioning as gradient processor, see [`PreconditionedDirection`](@ref)
for more mathematical background.

# Fields

* `direction`:      internal [`DirectionUpdateRule`](@ref) to determine directions to apply the preconditioning to
* `preconditioner`: the preconditioner function

# Constructors

    PreconditionedDirectionRule(
        M::AbstractManifold,
        preconditioner;
        direction::Union{<:DirectionUpdateRule,ManifoldDefaultsFactory}=IdentityUpdateRule(),
        evaluation::AbstractEvaluationType=AllocatingEvaluation()
    )

Add preconditioning to a gradient problem.

# Input

$(_var(:Argument, :M; type=true))
* `preconditioner`:   preconditioner function, either as a `(M, p, X)` -> Y` allocating or `(M, Y, p, X) -> Y` mutating function

# Keyword arguments

$(_var(:Keyword, :evaluation))
* `direction=`[`IdentityUpdateRule`](@ref) internal [`DirectionUpdateRule`](@ref) to determine the gradients to store or a [`ManifoldDefaultsFactory`](@ref) generating one
"""
mutable struct PreconditionedDirectionRule{
    E<:AbstractEvaluationType,D<:DirectionUpdateRule,F
} <: DirectionUpdateRule
    preconditioner::F
    direction::D
end
function PreconditionedDirectionRule(
    M::AbstractManifold,
    preconditioner::F;
    direction::Union{<:DirectionUpdateRule,ManifoldDefaultsFactory}=Gradient(),
    evaluation::E=AllocatingEvaluation(),
) where {E<:AbstractEvaluationType,F}
    dir = _produce_type(direction, M)
    return PreconditionedDirectionRule{E,typeof(dir),F}(preconditioner, dir)
end
function (pg::PreconditionedDirectionRule{AllocatingEvaluation})(
    mp::AbstractManoptProblem, s::AbstractGradientSolverState, k
)
    M = get_manifold(mp)
    p = get_iterate(s)
    # get inner direction and step size
    step, dir = pg.direction(mp, s, k)
    # precondition and set as gradient
    set_gradient!(s, M, p, pg.preconditioner(M, p, dir))
    return step, get_gradient(s)
end
function (pg::PreconditionedDirectionRule{InplaceEvaluation})(
    mp::AbstractManoptProblem, s::AbstractGradientSolverState, k
)
    M = get_manifold(mp)
    p = get_iterate(s)
    step, dir = pg.direction(mp, s, k) # get inner direction and step size
    pg.preconditioner(M, dir, p, dir)
    return step, dir
end

"""
    PreconditionedDirection(preconditioner; kwargs...)
    PreconditionedDirection(M::AbstractManifold, preconditioner; kwargs...)

Add a preconditioner to a gradient processor following the [motivation for optimization](https://en.wikipedia.org/wiki/Preconditioner#Preconditioning_in_optimization),
as a linear invertible map ``P: $(_math(:TpM)) → $(_math(:TpM))`` that usually should be

* symmetric: ``⟨X, P(Y)⟩ = ⟨P(X), Y⟩``
* positive definite ``⟨X, P(X)⟩ > 0`` for ``X`` not the zero-vector

The gradient is then preconditioned as ``P(X)``, where ``X`` is either the
gradient of the objective or the result of a previous (internally stored) gradient processor.

For example if you provide as the preconditioner the inverse of the Hessian ``$(_tex(:Hess))^{-1} f``,
you turn a gradient descent into a Newton method.

# Arguments

$(_var(:Argument, :M; type=true)) (optional)
* `preconditioner`:   preconditioner function, either as a `(M, p, X) -> Y` allocating or `(M, Y, p, X) -> Y` mutating function

# Keyword arguments

* `direction=`[`IdentityUpdateRule`](@ref) internal [`DirectionUpdateRule`](@ref) to determine the gradients to store or a [`ManifoldDefaultsFactory`](@ref) generating one
$(_var(:Keyword, :evaluation))

$(_note(:ManifoldDefaultFactory, "PreconditionedDirectionRule"))
"""
function PreconditionedDirection(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.PreconditionedDirectionRule, args...; kwargs...)
end

"""
    AbstractRestartCondition

A general struct, that indicates then to restart.
It is used within the [`ConjugateGradientDescentState`](@ref).

It is implemented to work as a functor `(problem, state, iteration) -> true|false`
and what is done in the restart case (`true`) is decided by the single solver.
"""
abstract type AbstractRestartCondition end

@doc """
    DebugGradient <: DebugAction

debug for the gradient evaluated at the current iterate

# Constructors
    DebugGradient(; long=false, prefix= , format= "\$prefix%s", io=stdout)

display the short (`false`) or long (`true`) default text for the gradient,
or set the `prefix` manually. Alternatively the complete format can be set.
"""
mutable struct DebugGradient <: DebugAction
    io::IO
    format::String
    function DebugGradient(;
        long::Bool=false,
        prefix=long ? "Gradient: " : "grad f(p):",
        format="$prefix%s",
        io::IO=stdout,
    )
        return new(io, format)
    end
end
function (d::DebugGradient)(::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    (k < 1) && return nothing
    Printf.format(d.io, Printf.Format(d.format), get_gradient(s))
    return nothing
end
function show(io::IO, dg::DebugGradient)
    return print(io, "DebugGradient(; format=\"$(dg.format)\")")
end
status_summary(dg::DebugGradient) = "(:Gradient, \"$(dg.format)\")"

@doc """
    DebugGradientNorm <: DebugAction

debug for gradient evaluated at the current iterate.

# Constructors
    DebugGradientNorm([long=false,p=print])

display the short (`false`) or long (`true`) default text for the gradient norm.

    DebugGradientNorm(prefix[, p=print])

display the a `prefix` in front of the gradient norm.
"""
mutable struct DebugGradientNorm <: DebugAction
    io::IO
    format::String
    function DebugGradientNorm(;
        long::Bool=false,
        prefix=long ? "Norm of the Gradient: " : "|grad f(p)|:",
        format="$prefix%s",
        io::IO=stdout,
    )
        return new(io, format)
    end
end
function (d::DebugGradientNorm)(
    mp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
)
    (k < 1) && return nothing
    Printf.format(
        d.io,
        Printf.Format(d.format),
        norm(get_manifold(mp), get_iterate(s), get_gradient(s)),
    )
    return nothing
end
function show(io::IO, dgn::DebugGradientNorm)
    return print(io, "DebugGradientNorm(; format=\"$(dgn.format)\")")
end
status_summary(dgn::DebugGradientNorm) = "(:GradientNorm, \"$(dgn.format)\")"

@doc """
    DebugStepsize <: DebugAction

debug for the current step size.

# Constructors
    DebugStepsize(;long=false,prefix="step size:", format="\$prefix%s", io=stdout)

display the a `prefix` in front of the step size.
"""
mutable struct DebugStepsize <: DebugAction
    io::IO
    format::String
    function DebugStepsize(;
        long::Bool=false,
        io::IO=stdout,
        prefix=long ? "step size:" : "s:",
        format="$prefix%s",
    )
        return new(io, format)
    end
end
function (d::DebugStepsize)(
    p::P, s::O, k::Int
) where {P<:AbstractManoptProblem,O<:AbstractGradientSolverState}
    (k < 1) && return nothing
    Printf.format(d.io, Printf.Format(d.format), get_last_stepsize(p, s, k))
    return nothing
end
function show(io::IO, ds::DebugStepsize)
    return print(io, "DebugStepsize(; format=\"$(ds.format)\")")
end
status_summary(ds::DebugStepsize) = "(:Stepsize, \"$(ds.format)\")"
#
# Records
#
@doc """
    RecordGradient <: RecordAction

record the gradient evaluated at the current iterate

# Constructors
    RecordGradient(ξ)

initialize the [`RecordAction`](@ref) to the corresponding type of the tangent vector.
"""
mutable struct RecordGradient{T} <: RecordAction
    recorded_values::Array{T,1}
    RecordGradient{T}() where {T} = new(Array{T,1}())
end
RecordGradient(::T) where {T} = RecordGradient{T}()
function (r::RecordGradient{T})(
    ::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
) where {T}
    return record_or_reset!(r, get_gradient(s), k)
end
show(io::IO, ::RecordGradient{T}) where {T} = print(io, "RecordGradient{$T}()")

@doc """
    RecordGradientNorm <: RecordAction

record the norm of the current gradient
"""
mutable struct RecordGradientNorm <: RecordAction
    recorded_values::Array{Float64,1}
    RecordGradientNorm() = new(Array{Float64,1}())
end
function (r::RecordGradientNorm)(
    mp::AbstractManoptProblem, ast::AbstractManoptSolverState, k::Int
)
    M = get_manifold(mp)
    return record_or_reset!(r, norm(M, get_iterate(ast), get_gradient(ast)), k)
end
show(io::IO, ::RecordGradientNorm) = print(io, "RecordGradientNorm()")

@doc """
    RecordStepsize <: RecordAction

record the step size
"""
mutable struct RecordStepsize <: RecordAction
    recorded_values::Array{Float64,1}
    RecordStepsize() = new(Array{Float64,1}())
end
function (r::RecordStepsize)(p::AbstractManoptProblem, s::AbstractGradientSolverState, k)
    return record_or_reset!(r, get_last_stepsize(p, s, k), k)
end
