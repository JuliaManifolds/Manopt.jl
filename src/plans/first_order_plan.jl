# TODO: Also implement get_cost/grad/diff on the function level or the tuple of functions,
#       that simplifies the dispatches again.

@doc """
    AbstractManifoldFirstOrderInformationFunction <: Function

An abstract type to represent the first order derivative information of a function.
"""
abstract type AbstractFirstOrderFunction <: Function end

@doc """
    CostDifferentialFunction{DD} <: AbstractFirstOrderFunction

A wrapper for a function representing a cost ``f`` and its differential ``Df: $(_math(:TM)) → ℝ``,
or in other words is is a map ``Df(p)[X] ∈ ℝ``, in a combined fashion as a function `(M, p, X) -> (c, d)`.

Since both return real values, this function would always work as an [`AllocatingEvaluation`](@ref).

# Fields

* `cost_diff::CD`: a function or functor for the gradient

# Constructor

    CostDifferentialFunction(cost_diff::FD)

Create a combined cost and differential function `cost_diff`.
"""
struct CostDifferentialFunction{FD} <: AbstractFirstOrderFunction
    cost_diff::FD
end

_maybe_unwrap_function(f) = f
_maybe_unwrap_function(cdf::CostDifferentialFunction) = cdf.cost_diff

@doc """
    CostGradientDifferentialFunction{FGD} <: AbstractFirstOrderFunction

A wrapper for a function representing a cost, its Riemannian gradient ``$(_tex(:grad))f: $(_math(:M)) → $(_math(:TM))``,
as well as its differential ``Df: $(_math(:TM)) → ℝ``, or in other words is is a map ``Df(p)[X] ∈ ℝ``.

It might have two forms

* as a function `(M, p, X) -> (c, Y, d)` that allocates memory for the gradient `Y` (also [`AllocatingEvaluation`](@ref))
* as a function `(M, Y, p, X) -> (c, Y, d)` that work in place of `Y` (also [`InplaceEvaluation`](@ref))

Note that both the cost and differential are real valued results, so both always work in an allocating sense.

# Fields

* `cost_grad_diff!!::FGD`: a function or functor for the cost, gradient, and differential

# Constructor

    CostGradientDifferentialFunction(cost_grad_diff!!::FGD)

Create a combined cost, grad, and diff function `cost_grad_diff!!`.
"""
struct CostGradientDifferentialFunction{FGD} <: AbstractFirstOrderFunction
    cost_grad_diff!!::FGD
end

_maybe_unwrap_function(cgdf::CostGradientDifferentialFunction) = cgdf.cost_grad_diff!!

@doc """
    CostGradientFunction{FG} <: AbstractFirstOrderFunction

A wrapper for a function representing the joint computation of the cost ``f`` and
its Riemannian gradient ``$(_tex(:grad))f: $(_math(:M)) → $(_math(:TM))``.

It might have two forms

* as a function `(M, p) -> (c, X)` that allocates memory for the gradient `X` (also [`AllocatingEvaluation`](@ref))
* as a function `(M, X, p) -> (c, X)` that work in place of `X` (also [`InplaceEvaluation`](@ref))

# Fields

* `cost_grad!!::FG`: a function or functor for the gradient

# Constructor

    CostGradientFunction(costgrad!!::FG)

Create a combined costgrad function `costgrad!!`.
"""
struct CostGradientFunction{FG} <: AbstractFirstOrderFunction
    cost_grad!!::FG
end

_maybe_unwrap_function(cgf::CostGradientFunction) = cgf.cost_grad!!

@doc """
    DifferentialFunction{D} <: AbstractFirstOrderFunction

A wrapper for a set of two function representing the gradient a function representing the differential ``Df: $(_math(:TM)) → ℝ``,
or in other words is is a map ``Df(p)[X] ∈ ℝ``.
Compared to a usual function or functor, this type is meant to distinguish
the differential from the gradient of a function, since both have similar
signatures that would only be distinguishable if we require types points and vectors.

# Fields
* `diff_f!!::D`: a function or functor for the gradient

# Constructor

    DifferentialFunction(diff_f::D)

Create a differential function `diff_f`
"""
struct DifferentialFunction{D} <: AbstractFirstOrderFunction
    diff::D
end

_maybe_unwrap_function(df::DifferentialFunction) = df.diff

@doc """
    GradientDifferentialFunction{CGD} <: AbstractFirstOrderFunction

A wrapper for a function representing a Riemannian gradient ``$(_tex(:grad))f: $(_math(:M)) → $(_math(:TM))``,
as well as its differential ``Df: $(_math(:TM)) → ℝ``, or in other words is is a map ``Df(p)[X] ∈ ℝ``.

It might have two forms

* as a function `(M, p, X) -> (Y, d)` that allocates memory for the gradient `Y` (also [`AllocatingEvaluation`](@ref))
* as a function `(M, Y, p, X) -> (Y, d)` that work in place of `Y` (also [`InplaceEvaluation`](@ref))

Note that differential is real valued, so always works in an allocating sense.

# Fields

* `grad_diff!!::GD`: a function or functor for the cost, gradient, and differential

# Constructor

    GradientDifferentialFunction(grad_diff!!::GD)

Create a combined cost, grad, and diff function `grad_diff!!`.
"""
struct GradientDifferentialFunction{GD} <: AbstractFirstOrderFunction
    grad_diff!!::GD
end

_maybe_unwrap_function(gdf::GradientDifferentialFunction) = gdf.grad_diff!!

@doc """
    GradientFunction{G} <: AbstractFirstOrderFunction

A wrapper for a function representing the Riemannian gradient ``$(_tex(:grad))f: $(_math(:M)) → $(_math(:TM))`` of a function.

# Fields
* `grad_f!!::G`: a function or functor for the gradient

# Constructor

    GradientFunction(grad_f)

Create a gradient function, where `grad_f`.
"""
struct GradientFunction{G} <: AbstractFirstOrderFunction
    grad!!::G
end

_maybe_unwrap_function(gf::GradientFunction) = gf.grad!!

@doc raw"""
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

1. a single type `FG`, i.e. a function or a functor, represents a combined function
  `(M, p) -> (c, X)` that computes the cost `c=cost(M,p)` and gradient `X=grad_f(M,p)`;
  this can wrapped in a [`CostGradientFunction`](@ref) optionally.
2. a `Tuple{F, G}` of two functions or functors represents a separate cost and gradient function;
  they can be wrapped in a [`CostFunction`](@ref) and a [`GradientFunction`](@ref), respectively.
3. a `Tuple{FG, D<:DifferentialFunction}` to provide the differential as a separate function,
  where the first is the same as in case 1 and can be wrapped in a [`CostGradientFunction`](@ref).
4. a `Tuple{F, G, D}` providing `cost::F`, `grad_f::G` and `diff_f::D` seperately; all three
  can be wrapped in their types, [`CostFunction`](@ref), [`GradientFunction`](@ref), and [`DifferentialFunction`](@ref), respectively
5. a `Tuple{F, GD <: GradientDifferentialFunction}` representing a `cost::F` and a common function
  `(X, d) = grad_diff(M, p, X)` indicated by the [`GradientDifferentialFunction`](@ref)
6. a [`CostDifferential`](@ref) representing a single function to compute `(c,d) = cost_diff(M, p, X)`.
7. a Tuple{FD<:CostDifferential, G} representing a function like in 6. and a separate gradient function,
  where the gradient function can be wrapped in a [`GradientFunction`](@ref).
8. a `cost` and a [`DifferentialFunction`]()
9. a [`CostGradientDifferential`](@ref) representing a single function to compute `(c,Y,d) = cost_grad_diff(M, p, X)`

For all cases where a gradient is present, also an in-place variant is possible, where the
signature has the result `Y` in second place.

The first two cases are the most common one. They can also be addressed by their constants
[`ManifoldCostGradientObjective`](@ref) and [`ManifoldGradientObjective`](@ref), respectively.

# Constructors
    ManifoldFirstOrderObjective(cost_grad; kwargs,...)
    ManifoldFirstOrderObjective(cost_differential::CostDifferentialFunction, grad=nothing; kwargs,...)
    ManifoldFirstOrderObjective(cost_grad_differential::CostGradientDifferentialFunction; kwargs,...)
    ManifoldFirstOrderObjective(cost, grad; kwargs,...)
    ManifoldFirstOrderObjective(cost; differential=nothing)

## Keyword arguments

* `differential = nothing` provide a separate function for the differential.
$(_var(:Keyword, :evaluation))

For the last signature, a differential has to be provided

# Used with
[`gradient_descent`](@ref), [`conjugate_gradient_descent`](@ref), [`quasi_Newton`](@ref)
"""
struct ManifoldFirstOrderObjective{E<:AbstractEvaluationType,F} <:
       AbstractManifoldFirstOrderObjective{E,F}
    functions::F
end

# a small wrapping helper
_mfo_wrap_diff(diff) = DifferentialFunction(diff)
_mfo_wrap_diff(diff::DifferentialFunction) = diff
_mfo_wrap_cost(cost) = CostFunction(cost)
_mfo_wrap_cost(cost::CostFunction) = cost
# Case 1: CostGrad
# Case 3: CostGrad and diff
function ManifoldFirstOrderObjective(
    cost_grad::FG; evaluation::E=AllocatingEvaluation(), differential=nothing
) where {FG<:CostGradientFunction,E<:AbstractEvaluationType}
    if !isnothing(differential)
        # 2-tuple, we have to indicate the second is a diff, so that the first is clearly
        # a costgrad
        cost_grad_diff = (costgrad, _mfo_wrap_diff(differential))
        return ManifoldFirstOrderObjective{E,typeof{cost_grad_diff}}(cost_grad_diff)
    end
    return ManifoldFirstOrderObjective{E,FG}(cost_grad)
end

# Case 2: cost and grad
# Case 4: cost and grad and diff
function ManifoldFirstOrderObjective(
    cost, grad; evaluation::E=AllocatingEvaluation(), differential=nothing
) where {E<:AbstractEvaluationType}
    if !isnothing(differential)
        cost_grad_diff = Tuple(cost, grad, differential)
        return ManifoldFirstOrderObjective{E,typeof{cost_grad_diff}}(cost_grad_diff)
    end
    cost_grad = (cost, grad)
    return ManifoldFirstOrderObjective{E,typeof(cost_grad)}(cost_grad)
end
# Case 5: cost and grad_diff
function ManifoldFirstOrderObjective(
    cost, grad_diff::GD; evaluation::E=AllocatingEvaluation()
) where {GD<:GradientDifferentialFunction,E<:AbstractEvaluationType}
    cost_grad_diff = (cost, grad_diff)
    return ManifoldFirstOrderObjective{E,typeof{cost_grad_diff}}(cost_grad_diff)
end
# Case 6: cost_diff
function ManifoldFirstOrderObjective(
    cost_diff::FD; evaluation::E=AllocatingEvaluation()
) where {FD<:CostDifferentialFunction,E<:AbstractEvaluationType}
    return ManifoldFirstOrderObjective{E,FD}(cost_diff)
end
# Case 7: cost_diff and extra grad
function ManifoldFirstOrderObjective(
    cost_diff::FD, grad; evaluation::E=AllocatingEvaluation()
) where {FD<:CostDifferentialFunction,E<:AbstractEvaluationType}
    cost_diff = (cost_diff, grad)
    return ManifoldFirstOrderObjective{E,typeof{cost_grad_diff}}(cost_diff)
end
# You can not provide diff twice – resolve an ambiguity
function ManifoldFirstOrderObjective(
    cost_diff::FD, grad_diff::GD; kwargs...
) where {FD<:CostDifferentialFunction,GD<:GradientDifferentialFunction}
    throw(
        DomainError(
            "You can not provide the differential together with a cost ($FD) and a gradient $(GD), please only provide a differential once.",
        ),
    )
end
# Case 8: cost_grad_diff in one function
function ManifoldFirstOrderObjective(
    cost_grad_diff::FGD; evaluation::E=AllocatingEvaluation()
) where {FGD<:CostGradientDifferentialFunction,E<:AbstractEvaluationType}
    return ManifoldFirstOrderObjective{E,FGD}(cost_grad_diff)
end
# Case 9: cost and diff (as keyword)
function ManifoldFirstOrderObjective(cost; differential=nothing)
    isnothing(differential) && thrown(
        DomainError(
            "For a first order objective some first order information as to be provide, here only a cost was provided, neither a gradient (positional) nor a differential",
        ),
    )
    # Make sure we store this typed, to avoid ambiguities
    cost_diff = Tuple(_mfo_wrap_cost(cost), _mfo_wrap_diff(differential))
    return ManifoldFirstOrderObjective{AllocatingEvaluation,typeof{cost_diff}}(cost_diff)
end
# For ease of use and to be nonbreaking in type names
const ManifoldGradientObjective{E,F,G} = ManifoldFirstOrderObjective{E,Tuple{F,G}}
@doc """
     ManifoldGradientObjective(cost, gradient; kwargs...)

TODO: Doocument old / comfort constructor
"""
function ManifoldGradientObjective(cost, grad; kwargs...)
    # Now case 2
    return ManifoldFirstOrderObjective(cost, grad; kwargs...)
end

const ManifoldCostGradientObjective{E,FG} = ManifoldFirstOrderObjective{E,FG}
@doc """
     ManifoldGradientObjective(cost_grad; kwargs...)

TODO: Doocument old / comfort constructor
"""
function ManifolCostGradientObjective(cost_grad; kwargs...)
    # Now case 1
    return ManifoldFirstOrderObjective(cost_grad; kwargs...)
end

#
# get_cost
# for all 9 cases of ManifoldFirstOrderObjective
# Case 1&3: FG, allocating
function get_cost(M::AbstractManifold, mfo::ManifoldFirstOrderObjective, p)
    (c, _) = _maybe_unwrap_function(mfo.functions)(M, p)
    return c
end
# Case 1&3: FG, Inplace
function get_cost(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E}, p
) where {E<:InplaceEvaluation}
    X = zero_vector(M, p)
    (c, _) = _maybe_unwrap_function(mfo.functions)(M, X, p)
    return c
end
# Case 2: F, G, Case 4: F, G, D, Case 5: F, GD, Case 9: F, D (cost alone first)
function get_cost(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{F,G}}, p
) where {E<:AllocatingEvaluation,F,G}
    return _maybe_unwrap_function(mfo.functions[1])(M, p)
end
# Case 2: F, G, Case 4: F, G, D, Case 5: F, GD, Case 9: F, D (cost alone first)
function get_cost(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{F,G}}, p
) where {E<:InplaceEvaluation,F,G}
    return _maybe_unwrap_function(mfo.functions[1])(M, p)
end
# Case 6: FD, dispatch on both eval cases to resolve an ambiguity.
function get_cost(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,<:CostDifferentialFunction}, p
) where {E<:AllocatingEvaluation}
    (c, _) = mfo.functions(M, p, zero_vector(M, p))
    return c
end
function get_cost(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,<:CostDifferentialFunction}, p
) where {E<:InplaceEvaluation}
    (c, _) = mfo.functions(M, p, zero_vector(M, p))
    return c
end
# Case 7: FD, G, twice to avoid ambiguities
function get_cost(
    M::AbstractManifold,
    mfo::ManifoldFirstOrderObjective{E,Tuple{<:CostDifferentialFunction,G}},
    p,
) where {E<:AllocatingEvaluation,G}
    (c, _) = mfo.functions[1](M, p, zero_vector(M, p))
    return c
end
function get_cost(
    M::AbstractManifold,
    mfo::ManifoldFirstOrderObjective{E,Tuple{<:CostDifferentialFunction,G}},
    p,
) where {E<:InplaceEvaluation,G}
    (c, _) = mfo.functions[1](M, p, zero_vector(M, p))
    return c
end

# Case 8: FGD, Alloc
function get_cost(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,FGD}, p
) where {E<:AllocatingEvaluation,FGD<:CostGradientDifferentialFunction}
    X = zero_vector(M, p)
    (c, _, _) = mfo.functions(M, p, X)
    return c
end
# Case 8: FGD, Inplace
function get_cost(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,FGD}, p
) where {E<:InplaceEvaluation,FGD<:CostGradientDifferentialFunction}
    X = zero_vector(M, p)
    (c, _, _) = mfo.functions(M, X, p, X)
    return c
end

# general: Generate a separate cost
function get_cost_function(mfo::ManifoldFirstOrderObjective, recursive=false)
    return (M, p) -> get_cost(M, mfo, p)
end
# general: return internal cost, case 4
function get_cost_function(
    mfo::ManifoldFirstOrderObjective{E,Tuple{F,G,D}}, recursive=false
) where {E<:AbstractEvaluationType,F,G,D}
    return _maybe_unwrap_function(mfo.functions[1])
end

#
# get_differential, for the cases where we have one (3,4,5,6,7,8,9) evaluate that
# for the other two (1,2) fall back to the inner product of the gradient
# 0 decorator cases
function get_differential(
    M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p, X
)
    return get_differential(M, get_objective(admo, false), p, X)
end

# Case 1: FG
function get_differential end
"""
    get_differential(M, amfo, p, X)

TODO
"""
get_differential(M::AbstractManifold, amfo::AbstractManifoldFirstOrderObjective, p, X)
# Case 1: FG (single FG), eval gradient and to an inner
function get_differential(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,FG}, p, X
) where {E<:AbstractEvaluationType,FG}
    return real(inner(M, p, get_gradient(M, mfo, p), X))
end
# Case 2: F, G
function get_differential(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{F,G}}, p, X
) where {E<:AbstractEvaluationType,F,G}
    return real(inner(M, p, get_gradient(M, mfo, p), X))
end
# Case 3: FG, D & Case 9: F, D (D second & typed)
function get_differential(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{F,D}}, p, X
) where {E<:AbstractEvaluationType,F<:CostFunction,D<:DifferentialFunction}
    return mfo.functions[2].diff(M, p, X)
end
# Case 4: F, G, D (D third – typed or not)
function get_differential(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{F,G,D}}, p, X
) where {E<:AbstractEvaluationType,F,G,D}
    return _maybe_unwrap_function(mfo.functions[3])(M, p, X)
end
# Case 5: F, GD (second typed), Allocating
function get_differential(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{F,GD}}, p, X
) where {E<:AllocatingEvaluation,F,GD<:GradientDifferentialFunction}
    (d, _) = mfo.functions[2](M, p, X)
    return d
end
# Case 5: F, GD (second typed), Inplace
function get_differential(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{F,GD}}, p, X
) where {E<:InplaceEvaluation,F,GD<:GradientDifferentialFunction}
    Y = zero_vector(M, p)
    (d, _) = mfo.funcrions[2](M, Y, p, X)
    return d
end
# Case 6: FD (typed)
function get_differential(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,FD}, p, X
) where {E<:AbstractEvaluationType,FD<:CostDifferentialFunction}
    (_, d) = mfo.functions(M, p, X)
    return d
end
# Case 7: FD, G (1 typed)
function get_differential(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{FD,G}}, p, X
) where {E<:AbstractEvaluationType,FD<:CostDifferentialFunction,G}
    (_, d) = mfo.functions[1].cost_diff(M, p, X)
    return d
end
# Case 8: FDG (typed), Allocating
function get_differential(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,FDG}, p, X
) where {E<:AllocatingEvaluation,FDG<:CostGradientDifferentialFunction}
    (_, _, d) = mfo.functions(M, p, X)
    return d
end
# Case 8: FDG (typed), Inplace
function get_differential(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,FDG}, p, X
) where {E<:InplaceEvaluation,FDG<:CostGradientDifferentialFunction}
    Y = zero_vector(M, p)
    (_, _, d) = mfo.functions(M, Y, p, X)
    return d
end

function get_differential_function end
@doc """
    get_differential_function(admo::AbstractManifoldFirstOrderObjective, recursive=false)

return the function to evaluate (just) the differential ``Df(p)[X]``.
For a decorated objective, the `recursive` positional parameter determines whether to
directly call this function on the next decorator or whether to get the “most inner” objective.
"""
get_differential_function(::AbstractManifoldFirstOrderObjective; recursive=false)

function get_differential_function(
    admo::AbstractDecoratedManifoldObjective, recursive=false
)
    return get_differential_function(get_objective(admo, recursive))
end
function get_differential_function(mfo::ManifoldFirstOrderObjective, recursive=false)
    return (M, p, X) -> get_differential(M, mfo, p, X)
end

#
#
# Gradient access – a bit of cases!
#
# 0 decorator cases
function get_gradient(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p)
    return get_gradient(M, get_objective(admo, false), p)
end
function get_gradient!(M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p)
    return get_gradient!(M, X, get_objective(admo, false), p)
end

# Case 1, FG (a) alloc
function get_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,FG}, p
) where {E<:AllocatingEvaluation,FG}
    (c, X) = _maybe_unwrap_function(mfo.functions)(M, p)
    return X
end
function get_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,FG}, p
) where {E<:AllocatingEvaluation,FG}
    (_, Y) = _maybe_unwrap_function(mfo.functions)(M, p)
    copyto!(M, X, p, Y)
    return X
end
# Case 1, FG (b) in-place
function get_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,FG}, p
) where {E<:InplaceEvaluation,FG}
    X = zero_vector(M, p)
    _maybe_unwrap_function(mfo.functions)(M, X, p)
    return X
end
function get_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,FG}, p
) where {E<:InplaceEvaluation,FG}
    _maybe_unwrap_function(mfo.functions)(M, X, p)
    return X
end
# case 2: F, G & Case 7: FD, G (gradient second)
# 2&7(a) alloc
function get_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{F,G}}, p
) where {E<:AllocatingEvaluation,F,G}
    return _maybe_unwrap_function(mfo.functions[2])(M, p)
end
function get_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,Tuple{F,G}}, p
) where {E<:AllocatingEvaluation,F,G}
    return copyto!(M, X, p, _maybe_unwrap_function(mfo.functions[2])(M, p))
end
# 2&7(b) in-place
function get_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{F,G}}, p
) where {E<:InplaceEvaluation,F,G}
    X = zero_vector(M, p)
    return _maybe_unwrap_function(mfo.functions[2])(M, X, p)
end
function get_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,Tuple{F,G}}, p
) where {E<:InplaceEvaluation,F,G}
    return _maybe_unwrap_function(mfo.functions[2])(M, X, p)
end
# Case 3: FG, D
# 3(a) alloc
function get_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{FG,D}}, p
) where {E<:AllocatingEvaluation,FG,D<:DifferentialFunction}
    (_, X) = _maybe_unwrap_function(mfo.functions[1])(M, p)
    return X
end
function get_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,Tuple{FG,D}}, p
) where {E<:AllocatingEvaluation,FG,D<:DifferentialFunction}
    (_, Y) = _maybe_unwrap_function(mfo.functions[1])(M, p)
    copyto!(M, X, p, Y)
    return X
end
# 3(b), in-place
function get_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{FG,D}}, p
) where {E<:InplaceEvaluation,FG,D<:DifferentialFunction}
    X = zero_vector(M, p)
    _maybe_unwrap_function(mfo.functions[1])(M, X, p)
    return X
end
function get_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,Tuple{FG,D}}, p
) where {E<:InplaceEvaluation,FG,D<:DifferentialFunction}
    _maybe_unwrap_function(mfo.functions[1])(M, X, p)
    return X
end
# Case 4: F, G, D
# 4(a) alloc
function get_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{F,G,D}}, p
) where {E<:AllocatingEvaluation,F,G,D}
    return _maybe_unwrap_function(mfo.functions[2])(M, p)
end
function get_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,Tuple{F,G,D}}, p
) where {E<:AllocatingEvaluation,F,G,D}
    return copyto!(M, X, p, _maybe_unwrap_function(mfo.functions[2])(M, p))
end
# 4(b), in-place
function get_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{F,G,D}}, p
) where {E<:InplaceEvaluation,F,G,D}
    X = zero_vector(M, p)
    return _maybe_unwrap_function(mfo.functions[2])(M, X, p)
end
function get_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,Tuple{F,G,D}}, p
) where {E<:InplaceEvaluation,F,G,D}
    return _maybe_unwrap_function(mfo.functions[2])(M, X, p)
end
# Case 5: F, GD (2 cases)
# 5(a) alloc
function get_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{F,GD}}, p
) where {E<:AllocatingEvaluation,F,GD<:GradientDifferentialFunction}
    X = zero_vector(M, p)
    (Y, _) = _maybe_unwrap_function(mfo.functions[2])(M, p, X)
    return Y
end
function get_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,Tuple{F,GD}}, p
) where {E<:AllocatingEvaluation,F,GD<:GradientDifferentialFunction}
    (Y, _) = _maybe_unwrap_function(mfo.functions[2])(M, p, X)
    return copyto!(M, X, p, Y)
end
# 5(b), in-place
function get_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{F,GD}}, p
) where {E<:InplaceEvaluation,F,GD<:GradientDifferentialFunction}
    X = zero_vector(M, p)
    _maybe_unwrap_function(mfo.functions[2])(M, X, p, X)
    return X
end
function get_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,Tuple{F,GD}}, p
) where {E<:InplaceEvaluation,F,GD<:GradientDifferentialFunction}
    _maybe_unwrap_function(mfo.functions[2])(M, X, p, X)
    return X
end
# Case 6 does not provide a gradient
# Case 8: FGD
# 8(a) alloc
function get_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,FGD}, p
) where {E<:AllocatingEvaluation,FGD<:CostGradientDifferentialFunction}
    X = zero_vector(M, p)
    (_, Y, _) = _maybe_unwrap_function(mfo.functions)(M, p, X)
    return Y
end
function get_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,FGD}, p
) where {E<:AllocatingEvaluation,FGD<:CostGradientDifferentialFunction}
    (_, Y, _) = _maybe_unwrap_function(mfo.functions)(M, p, X)
    return copyto!(M, X, p, Y)
end
# 8(b), in-place
function get_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,FGD}, p
) where {E<:InplaceEvaluation,FGD<:CostGradientDifferentialFunction}
    X = zero_vector(M, p)
    _maybe_unwrap_function(mfo.functions)(M, X, p, X)
    return X
end
function get_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,FGD}, p
) where {E<:InplaceEvaluation,FGD<:CostGradientDifferentialFunction}
    _maybe_unwrap_function(mfo.functions)(M, X, p, X)
    return X
end
# Case 9 does not provide a gradient

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
    cgo::ManifoldFirstOrderObjective{AllocatingEvaluation}, recursive=false
)
    return (M, p) -> get_gradient(M, cgo, p)
end
function get_gradient_function(
    cgo::ManifoldFirstOrderObjective{InplaceEvaluation}, recursive=false
)
    return (M, X, p) -> get_gradient!(M, X, cgo, p)
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
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,FG}, p
) where {E<:AllocatingEvaluation,FG}
    return _maybe_unwrap_function(mfo.functions)(M, p)
end
function get_cost_and_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,FG}, p
) where {E<:AllocatingEvaluation,FG}
    (c, Y) = _maybe_unwrap_function(mfo.functions)(M, p)
    copyto!(M, X, p, Y)
    return (c, X)
end
# Case 1, FG (b) in-place
function get_cost_and_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,FG}, p
) where {E<:InplaceEvaluation,FG}
    X = zero_vector(M, p)
    return _maybe_unwrap_function(mfo.functions)(M, X, p)
end
function get_cost_and_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,FG}, p
) where {E<:InplaceEvaluation,FG}
    return _maybe_unwrap_function(mfo.functions)(M, X, p)
end
# case 2: F, G & Case 7: FD, G (gradient second)
# 2&7(a) alloc
function get_cost_and_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{F,G}}, p
) where {E<:AllocatingEvaluation,F,G}
    c = _maybe_unwrap_function(mfo.functions[1])(M, p)
    X = _maybe_unwrap_function(mfo.functions[2])(M, p)
    return (c, X)
end
function get_cost_and_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,Tuple{F,G}}, p
) where {E<:AllocatingEvaluation,F,G}
    c = _maybe_unwrap_function(mfo.functions[1])(M, p)
    copyto!(M, X, p, _maybe_unwrap_function(mfo.functions[2])(M, p))
    return (c, X)
end
# 2&7(b) in-place
function get_cost_and_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{F,G}}, p
) where {E<:InplaceEvaluation,F,G}
    c = _maybe_unwrap_function(mfo.functions[1])(M, p)
    X = zero_vector(M, p)
    _maybe_unwrap_function(mfo.functions[2])(M, X, p)
    return (c, X)
end
function get_cost_and_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,Tuple{F,G}}, p
) where {E<:InplaceEvaluation,F,G}
    return _maybe_unwrap_function(mfo.functions[2])(M, X, p)
end
# Case 3: FG, D
# 3(a) alloc
function get_cost_and_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{FG,D}}, p
) where {E<:AllocatingEvaluation,FG,D<:DifferentialFunction}
    return _maybe_unwrap_function(mfo.functions[1])(M, p)
end
function get_cost_and_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,Tuple{FG,D}}, p
) where {E<:AllocatingEvaluation,FG,D<:DifferentialFunction}
    (c, Y) = _maybe_unwrap_function(mfo.functions[1])(M, p)
    copyto!(M, X, p, Y)
    return (c, X)
end
# 3(b), in-place
function get_cost_and_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{FG,D}}, p
) where {E<:InplaceEvaluation,FG,D<:DifferentialFunction}
    X = zero_vector(M, p)
    return _maybe_unwrap_function(mfo.functions[1])(M, X, p)
end
function get_cost_and_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,Tuple{FG,D}}, p
) where {E<:InplaceEvaluation,FG,D<:DifferentialFunction}
    return _maybe_unwrap_function(mfo.functions[1])(M, X, p)
end
# Case 4: F, G, D
# 4(a) alloc
function get_cost_and_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{F,G,D}}, p
) where {E<:AllocatingEvaluation,F,G,D}
    c = _maybe_unwrap_function(mfo.functions[1])(M, p)
    X = _maybe_unwrap_function(mfo.functions[2])(M, p)
    return nothing
end
function get_cost_and_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,Tuple{F,G,D}}, p
) where {E<:AllocatingEvaluation,F,G,D}
    c = _maybe_unwrap_function(mfo.functions[1])(M, p)
    copyto!(M, X, p, _maybe_unwrap_function(mfo.functions[2])(M, p))
    return (c, X)
end
# 4(b), in-place
function get_cost_and_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{F,G,D}}, p
) where {E<:InplaceEvaluation,F,G,D}
    c = _maybe_unwrap_function(mfo.functions[1])(M, p)
    X = zero_vector(M, p)
    _maybe_unwrap_function(mfo.functions[2])(M, X, p)
    return (c, X)
end
function get_cost_and_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,Tuple{F,G,D}}, p
) where {E<:InplaceEvaluation,F,G,D}
    c = _maybe_unwrap_function(mfo.functions[1])(M, p)
    _maybe_unwrap_function(mfo.functions[2])(M, X, p)
    return (c, X)
end
# Case 5: F, GD (2 cases)
# 5(a) alloc
function get_cost_and_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{F,GD}}, p
) where {E<:AllocatingEvaluation,F,GD<:GradientDifferentialFunction}
    c = _maybe_unwrap_function(mfo.functions[1])(M, p)
    X = zero_vector(M, p)
    (Y, _) = _maybe_unwrap_function(mfo.functions[2])(M, p, X)
    return (c, Y)
end
function get_cost_and_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,Tuple{F,GD}}, p
) where {E<:AllocatingEvaluation,F,GD<:GradientDifferentialFunction}
    c = _maybe_unwrap_function(mfo.functions[1])(M, p)
    (Y, _) = _maybe_unwrap_function(mfo.functions[2])(M, p, X)
    copyto!(M, X, p, Y)
    return (c, X)
end
# 5(b), in-place
function get_cost_and_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,Tuple{F,GD}}, p
) where {E<:InplaceEvaluation,F,GD<:GradientDifferentialFunction}
    c = _maybe_unwrap_function(mfo.functions[1])(M, p)
    X = zero_vector(M, p)
    _maybe_unwrap_function(mfo.functions[2])(M, X, p, X)
    return (c, X)
end
function get_cost_and_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,Tuple{F,GD}}, p
) where {E<:InplaceEvaluation,F,GD<:GradientDifferentialFunction}
    c = _maybe_unwrap_function(mfo.functions[1])(M, p)
    _maybe_unwrap_function(mfo.functions[2])(M, X, p, X)
    return (c, X)
end
# Case 6 does not provide a gradient
# Case 8: FGD
# 8(a) alloc
function get_cost_and_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,FGD}, p
) where {E<:AllocatingEvaluation,FGD<:CostGradientDifferentialFunction}
    X = zero_vector(M, p)
    (c, Y, _) = _maybe_unwrap_function(mfo.functions)(M, p, X)
    return (c, Y)
end
function get_cost_and_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,FGD}, p
) where {E<:AllocatingEvaluation,FGD<:CostGradientDifferentialFunction}
    (c, Y, _) = _maybe_unwrap_function(mfo.functions)(M, p, X)
    copyto!(M, X, p, Y)
    return (c, Y)
end
# 8(b), in-place
function get_cost_and_gradient(
    M::AbstractManifold, mfo::ManifoldFirstOrderObjective{E,FGD}, p
) where {E<:InplaceEvaluation,FGD<:CostGradientDifferentialFunction}
    X = zero_vector(M, p)
    (c, _, _) = _maybe_unwrap_function(mfo.functions)(M, X, p, X)
    return (c, X)
end
function get_cost_and_gradient!(
    M::AbstractManifold, X, mfo::ManifoldFirstOrderObjective{E,FGD}, p
) where {E<:InplaceEvaluation,FGD<:CostGradientDifferentialFunction}
    (c, _, _) = _maybe_unwrap_function(mfo.functions)(M, X, p, X)
    return (c, X)
end
# Case 9 does not provide a gradient

#
#  Access gradient
# -----------------------------

@doc raw"""
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

@doc raw"""
    get_gradient(agst::AbstractGradientSolverState)

return the gradient stored within gradient options.
THe default returns `agst.X`.
"""
get_gradient(agst::AbstractGradientSolverState) = agst.X

@doc raw"""
    set_gradient!(agst::AbstractGradientSolverState, M, p, X)

set the (current) gradient stored within an [`AbstractGradientSolverState`](@ref) to `X`.
The default function modifies `s.X`.
"""
function set_gradient!(agst::AbstractGradientSolverState, M, p, X)
    copyto!(M, agst.X, p, X)
    return agst
end

@doc raw"""
    get_iterate(agst::AbstractGradientSolverState)

return the iterate stored within gradient options.
THe default returns `agst.p`.
"""
get_iterate(agst::AbstractGradientSolverState) = agst.p

@doc raw"""
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

@doc raw"""
    DebugGradient <: DebugAction

debug for the gradient evaluated at the current iterate

# Constructors
    DebugGradient(; long=false, prefix= , format= "$prefix%s", io=stdout)

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

@doc raw"""
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

@doc raw"""
    DebugStepsize <: DebugAction

debug for the current step size.

# Constructors
    DebugStepsize(;long=false,prefix="step size:", format="$prefix%s", io=stdout)

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
@doc raw"""
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

@doc raw"""
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

@doc raw"""
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
