@doc """
    AbstractManifoldFirstOrderObjective{E<:AbstractEvaluationType, FGD} <: AbstractManifoldCostObjective{E, FGD}

An abstract type for all objectives that provide
* a cost
* first order information, so either a (full) gradient or a differential, where
`E` is a [`AbstractEvaluationType`](@ref) for the gradient function.
"""
abstract type AbstractManifoldFirstOrderObjective{E <: AbstractEvaluationType, FGD} <:
AbstractManifoldCostObjective{E, FGD} end

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
$(_kwargs(:evaluation))

Where:
 * At least one of `cost`, `costgradient` or `costdifferential` must be provided.
 * Either `gradient`, `costgradient`, `differential` or `costdifferential` must be provided.
 * If more than one function provides the same thing (e.g. cost), it is assumed that all
   such functions return the same value. Optimization algorithms will attempt to make the
   most efficient use of provided functions.

# Used with
[`gradient_descent`](@ref), [`conjugate_gradient_descent`](@ref), [`quasi_Newton`](@ref)
"""
struct ManifoldFirstOrderObjective{E <: AbstractEvaluationType, F <: NamedTuple} <:
    AbstractManifoldFirstOrderObjective{E, F}
    functions::F
end

# A Monster constructor
function ManifoldFirstOrderObjective(;
        cost = nothing,
        differential = nothing,
        gradient = nothing,
        costgradient = nothing,
        costdifferential = nothing,
        evaluation::E = AllocatingEvaluation(),
    ) where {E <: AbstractEvaluationType}
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

    return ManifoldFirstOrderObjective{E, typeof(nt)}(nt)
end

const ManifoldGradientObjective{E, F, G} = ManifoldFirstOrderObjective{
    E,
    <:Union{
        NamedTuple{Tuple{:cost, :gradient}, Tuple{F, G}},
        NamedTuple{Tuple{:cost, :gradient, :differential}, Tuple{F, G, D where {D}}},
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
    return ManifoldFirstOrderObjective(; cost = cost, gradient = grad, kwargs...)
end

const ManifoldCostGradientObjective{E, FG} = ManifoldFirstOrderObjective{
    E,
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
        Y = nothing,
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
        mfo::ManifoldFirstOrderObjective{AllocatingEvaluation}, recursive::Bool = false
    )
    if haskey(mfo.functions, :cost)
        return mfo.functions[:cost]
    else
        return (M, p) -> get_cost(M, mfo, p)
    end
end
function get_cost_function(
        mfo::ManifoldFirstOrderObjective{InplaceEvaluation}, recursive::Bool = false
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
By default this falls back to ``Df(p)[X] = ⟨$(_tex(:grad))f(p), X⟩``.

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
        gradient = nothing,
        evaluated::Bool = false,
    )
    isnothing(gradient) && (return real(inner(M, p, get_gradient(M, amfo, p), X)))
    # if it is not nothing call in-place
    (!evaluated) && (get_gradient!(M, gradient, amfo, p))
    return real(inner(M, p, gradient, X))
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
# Differential function - pass-through
function get_differential_function(
        admo::AbstractDecoratedManifoldObjective, recursive = false
    )
    return get_differential_function(get_objective(admo, recursive))
end

@doc """
     get_differential_function(admo::AbstractManifoldFirstOrderObjective, recursive::Bool=false)

Return the function to evaluate (just) the differential ``Df(p)[X]``.
For a decorated objective, the `recursive` positional parameter determines whether to
directly call this function on the next decorator or whether to get the “most inner” objective.
"""
get_differential_function(::AbstractManifoldFirstOrderObjective; recursive::Bool = false)

function get_differential_function(
        mfo::ManifoldFirstOrderObjective{<:AbstractEvaluationType}, recursive::Bool = false
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

function get_gradient!(
        M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{<:NamedTuple}, p,
    )
    haskey(mfo.functions, :gradient) && (return mfo.functions[:gradient](M, X, p))
    haskey(mfo.functions, :costgradient) && (return mfo.functions[:costgradient](M, X, p)[2])
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
get_gradient_function(::AbstractManifoldFirstOrderObjective; recursive = false)

function get_gradient_function(admo::AbstractDecoratedManifoldObjective, recursive = false)
    return get_gradient_function(get_objective(admo, recursive))
end
function get_gradient_function(
        mfo::ManifoldFirstOrderObjective, recursive = false
    )
    haskey(mfo.functions, :gradient) && (return mfo.functions[:gradient])
    return (M, X, p) -> get_gradient!(M, X, mfo, p)
end

#
#  Access cost and gradient – a bit of cases
# -----------------------------
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

function get_cost_and_gradient(
        M::AbstractManifold, mfo::AbstractManifoldFirstOrderObjective, p
    )
    X = zero_vector(M, p)
    return get_cost_and_gradient!(M, X, mfo, p)
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
    X = zero_vector!(M, p)
    return get_sub_gradient!(M, X, agmo, p)
end
function get_subgradient!(
        M::AbstractManifold, X, agmo::AbstractManifoldFirstOrderObjective, p
    )
    return get_gradient!(M, X, agmo, p)
end

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
    recorded_values::Array{T, 1}
    RecordGradient{T}() where {T} = new(Array{T, 1}())
end
RecordGradient(::T) where {T} = RecordGradient{T}()
function (r::RecordGradient{T})(
        ::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
    ) where {T}
    return record_or_reset!(r, get_gradient(s), k)
end
show(io::IO, ::RecordGradient{T}) where {T} = print(io, "RecordGradient($T)")
function status_summary(rg::RecordGradient; context::Symbol = :default)
    (context === :short) && return ":Gradient"
    return "A RecordAction to record the current gradient"
end
@doc """
    RecordGradientNorm{R<:Real} <: RecordAction

record the norm of the current gradient

## Constructor
    RecordGradientNorm(r::Type{<:Real}=Float64)
"""
mutable struct RecordGradientNorm{R <: Real} <: RecordAction
    recorded_values::Array{R, 1}
    RecordGradientNorm(r::Type{<:Real} = Float64) = new{r}(Array{r, 1}())
end
function (r::RecordGradientNorm)(
        mp::AbstractManoptProblem, ast::AbstractManoptSolverState, k::Int
    )
    M = get_manifold(mp)
    return record_or_reset!(r, norm(M, get_iterate(ast), get_gradient(ast)), k)
end
show(io::IO, ::RecordGradientNorm) = print(io, "RecordGradientNorm()")
function status_summary(rg::RecordGradientNorm; context::Symbol = :default)
    (context === :short) && return ":GradientNorm"
    return "A RecordAction to record the current gradient norm"
end

@doc """
    RecordStepsize <: RecordAction

record the step size.

## Constructor
    RecordStepsise(r::Type{<:Real}=Float64)
"""
mutable struct RecordStepsize{R <: Real} <: RecordAction
    recorded_values::Array{R, 1}
    RecordStepsize(r::Type{<:Real} = Float64) = new{r}(Array{r, 1}())
end
function (r::RecordStepsize)(p::AbstractManoptProblem, s::AbstractGradientSolverState, k)
    return record_or_reset!(r, get_last_stepsize(p, s, k), k)
end
show(io::IO, ::RecordStepsize{R}) where {R} = print(io, "RecordStepsize($R)")
function status_summary(rg::RecordStepsize{R}; context::Symbol = :default) where {R}
    (context === :short) && return ":Stepsize"
    return "A RecordAction to record the current stepsize (of type $R)"
end
