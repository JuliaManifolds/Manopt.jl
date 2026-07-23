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
