@doc """
    AbstractManifoldFirstOrderInformationFunction <: Function

An abstract type to represent the first order derivative information of a function.
"""
abstract type AbstractFirstOrderFunction <: Function end

@doc """
    GradientFunction{G} <: AbstractFirstOrderFunction

A wrapper for a function representing the Riemannian gradient ``$(_tex(:grad))f: $(_math(:M)) → $(_math(:TM))`` of a function.

# Fields
* `grad_f!!::G`: a function or functor for the gradient

# Constructor

    GradientFunction(grad_f)
    GradientFunction(grad_f::GradientFunction)

Create a gradient function, where `grad_f`.
This is a “single-wrapper” in the sense that the second constructor avoids to “wrap twice”.
"""
struct GradientFunction{G} <: AbstractFirstOrderFunction
    grad_f!!::G
end
# avoid double wrap
GradientFunction(grad_f::GradientFunction) = GradientFunction(grad_f.grad_f!!)

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
    DifferentialFunction(diff_f::DifferentialFunction)

Create a differential function `diff_f`
This is a “single-wrapper” in the sense that the second constructor avoids to “wrap twice”.
"""
struct DifferentialFunction{D} <: AbstractFirstOrderFunction
    diff_f!!::D
end
DifferentialFunction(diff_f::DifferentialFunction) = DifferentialFunction(diff_f.diff_f!!)

@doc """
    CostGradientFunction{CG} <: AbstractFirstOrderFunction

A wrapper for a function representing the joint computation of the cost ``f`` and
its Riemannian gradient ``$(_tex(:grad))f: $(_math(:M)) → $(_math(:TM))``.

It might have two forms

* as a function `(M, p) -> (c, X)` that allocates memory for the gradient `X` (also [`AllocatingEvaluation`](@ref))
* as a function `(M, X, p) -> (c, X)` that work in place of `X` (also [`InplaceEvaluation`](@ref))

# Fields

* `cost_grad!!::CG`: a function or functor for the gradient

# Constructor

    CostGradientFunction(costgrad!!::CG)
    CostGradientFunction(costgrad!!::CostGradientFunction)

Create a combined costgrad function `costgrad!!`.
This is a “single-wrapper” in the sense that the second constructor avoids to “wrap twice”.
"""
struct CostGradientFunction{CG} <: AbstractFirstOrderFunction
    cost_grad!!::CG
end
function CostGradientFunction(cost_grad::CostGradientFunction)
    return CostGradientFunction(cost_grad.cost_grad!!)
end

@doc """
    CostDifferentialFunction{CD} <: AbstractFirstOrderFunction

A wrapper for a function representing a cost ``f`` and its differential ``Df: $(_math(:TM)) → ℝ``,
or in other words is is a map ``Df(p)[X] ∈ ℝ``, in a combined fashion as a function `(M, p, X) -> (c, d)`.

Since both return real values, this function would always work as an [`AllocatingEvaluation`](@ref).

# Fields

* `cost_diff::CD`: a function or functor for the gradient

# Constructor

    CostDifferentialFunction(costdiff::CD)
    CostDifferenitalFunction(cost_diff::CostDifferentialFunction)

Create a combined cost and differential function `costdiff`
This is a “single-wrapper” in the sense that the second constructor avoids to “wrap twice”.
"""
struct CostDifferentialFunction{CD} <: AbstractFirstOrderFunction
    cost_diff::CD
end
function CostDifferentialFunction(cost_diff::CostDifferentialFunction)
    return CostDifferentialFunction(cost_diff.cost_diff)
end

@doc """
    CostGradientDifferentialFunction{CGD} <: AbstractFirstOrderFunction

A wrapper for a function representing a cost, its Riemannian gradient ``$(_tex(:grad))f: $(_math(:M)) → $(_math(:TM))``,
as well as its differential ``Df: $(_math(:TM)) → ℝ``, or in other words is is a map ``Df(p)[X] ∈ ℝ``.

It might have two forms

* as a function `(M, p, X) -> (c, Y, d)` that allocates memory for the gradient `Y` (also [`AllocatingEvaluation`](@ref))
* as a function `(M, Y, p, X) -> (c, Y, d)` that work in place of `Y` (also [`InplaceEvaluation`](@ref))

Note that both the cost and differential are real valued results, so both always work in an allocating sense.

# Fields

* `cost_grad_diff!!::CGD`: a function or functor for the cost, gradient, and differential

# Constructor

    CostGradientDifferentialFunction(cost_graddiff!!::CGD)

Create a combined cost, grad, and diff function `cost_graddiff!!`.
"""
struct CostGradientDifferentialFunction{CGD} <: AbstractFirstOrderFunction
    cost_grad_diff!!::CGD
end

@doc raw"""
    AbstractManifoldFirstOrderObjective{E<:AbstractEvaluationType, TC, TG} <: AbstractManifoldCostObjective{E, TC}

An abstract type for all objectives that provide a (full) gradient, where
`T` is a [`AbstractEvaluationType`](@ref) for the gradient function.
"""
abstract type AbstractManifoldFirstOrderObjective{E<:AbstractEvaluationType,TC,TG} <:
              AbstractManifoldCostObjective{E,TC} end

@doc """
    ManifoldFirstOrderObjective{T<:AbstractEvaluationType} <: AbstractManifoldFirstOrderObjective{T}

specify an objective containing a cost and its gradient or differential.

# Fields

* `cost`:          a function ``f: $(_math(:M)) → ℝ``
* `first_order!!`: the first order information of the cost function ``f``.
  If it is a function, it represents a gradient ``$(_tex(:grad))f: $(_math(:M)) → $(_math(:TM))``,
  which equivalently is a [`GradientFunction`](@ref). It can also represent a [`DifferentialFunction`](@ref),
  or wrap both simultaneously as a tuple of these.

Depending on the [`AbstractEvaluationType`](@ref) `T` the gradient can have to forms

* as a function `(M, p) -> X` that allocates memory for `X`, an [`AllocatingEvaluation`](@ref)
* as a function `(M, X, p) -> X` that work in place of `X`, an [`InplaceEvaluation`](@ref)

Since the differential always returns a real number, the evaluation type is not used for it.

# Constructors
    ManifoldFirstOrderObjective(cost, gradient=nothing; kwargs,...)

## Keyword arguments

* `differential = nothing` provide a separate function for the differential
$(_var(:Keyword, :evaluation; add=:GradientExample))

Either the gradient or the differential have to be different from `nothing`.

# Used with
[`gradient_descent`](@ref), [`conjugate_gradient_descent`](@ref), [`quasi_Newton`](@ref)
"""
struct ManifoldFirstOrderObjective{E<:AbstractEvaluationType,C,G} <:
       AbstractManifoldFirstOrderObjective{E,C,G}
    cost::C
    first_order!!::G
end

function ManifoldFirstOrderObjective(
    cost::C, first_order!!::G; evaluation::AbstractEvaluationType=AllocatingEvaluation()
) where {C,G}
    return ManifoldFirstOrderObjective{typeof(evaluation),C,G}(cost, first_order!!)
end

@doc """
    ManifoldCombinedFirstOrderObjective{T} <: AbstractManifoldFirstOrderObjective{T}

specify an objective containing one function to perform a combined computation of
the cost and first order information like gradient and/or differential

# Fields

* `costgrad!!`: a function that computes both the cost ``f: $(_math(:M)) → ℝ``
  and its gradient ``$(_tex(:grad))f: $(_math(:M)) → $(_math(:TM))``
  and/or its differential ``Df: $(_math(:TM)) → ℝ``

Depending on the [`AbstractEvaluationType`](@ref) `T` the gradient
can be of [`AllocatingEvaluation`](@ref) or [`InplaceEvaluation`](@ref).

Having a function stored in the field is equivalent to the [``]

# Constructors

    ManifoldCombinedFirstOrderObjective(costgrad; evaluation=AllocatingEvaluation())

# Used with
[`gradient_descent`](@ref), [`conjugate_gradient_descent`](@ref), [`quasi_Newton`](@ref)
"""
struct ManifoldCombinedFirstOrderObjective{E,CG} <:
       AbstractManifoldFirstOrderObjective{E,CG,CG}
    cost_first_order!!::CG
end
function ManifoldCombinedFirstOrderObjective(
    cost_first_order!!::CG; evaluation::E=AllocatingEvaluation()
) where {CG,E<:AbstractEvaluationType}
    return ManifoldCombinedFirstOrderObjective{E,CG}(cost_first_order!!)
end

function get_cost(
    mcfoo::ManifoldCombinedFirstOrderObjective{E,CostDifferentialFunction}, M, p
) where {E<:AbstractEvaluationType}
    X = zero_vector(M, p)
    (c, _) = mcfoo.cost_first_order!!(M, p, X)
    return c
end
function get_cost(
    mcfoo::ManifoldCombinedFirstOrderObjective{AllocatingEvaluation,CostGradientFunction},
    M,
    p,
)
    (c, _) = mcfoo.cost_first_order!!(M, p)
    return c
end
function get_cost(
    mcfoo::ManifoldCombinedFirstOrderObjective{InplaceEvaluation,CostGradientFunction}, M, p
)
    X = zero_vector(M, p)
    (c, _) = mcfoo.cost_first_order!!(M, X, p)
    return c
end
function get_cost(
    mcfoo::ManifoldCombinedFirstOrderObjective{
        AllocatingEvaluation,CostGradientDifferentialFunction
    },
    M,
    p,
)
    X = zero_vector(M, p)
    (c, _) = mcfoo.cost_first_order!!(M, p, X)
    return c
end
function get_cost(
    mcfoo::ManifoldCombinedFirstOrderObjective{
        InplaceEvaluation,CostGradientDifferentialFunction
    },
    M,
    p,
)
    X = zero_vector(M, p)
    (c, _, _) = mcfoo.cost_first_order!!(M, X, p, X)
    return c
end

function get_differential(
    mcfoo::ManifoldCombinedFirstOrderObjective{E,CostDifferentialFunction}, M, p, X
) where {E<:AbstractEvaluationType}
    (_, d) = mcfoo.cost_first_order!!(M, p, X)
    return d
end
function get_differential(
    mcfoo::ManifoldCombinedFirstOrderObjective{AllocatingEvaluation,CostGradientFunction},
    M,
    p,
    X,
)
    (_, Y) = mcfoo.cost_first_order!!(M, p)
    return real(inner(M, p, Y, X))
end
function get_differential(
    mcfoo::ManifoldCombinedFirstOrderObjective{InplaceEvaluation,CostGradientFunction},
    M,
    p,
    Y,
)
    X = zero_vector(M, p)
    mcfoo.cost_first_order!!(M, X, p)
    return real(inner(M, p, Y, X))
end
function get_differential(
    mcfoo::ManifoldCombinedFirstOrderObjective{
        AllocatingEvaluation,CostGradientDifferentialFunction
    },
    M,
    p,
    X,
)
    (_, _, d) = mcfoo.cost_first_order!!(M, p, X)
    return d
end
function get_differential(
    mcfoo::ManifoldCombinedFirstOrderObjective{
        InplaceEvaluation,CostGradientDifferentialFunction
    },
    M,
    p,
    X,
)
    Y = zero_vector(M, p)
    (_, _, d) = mcfoo.cost_first_order!!(M, Y, p, X)
    return d
end

function get_cost_function(cgo::ManifoldFirstOrderObjective, recursive=false)
    return (M, p) -> get_cost(M, cgo, p)
end

#
# Access the differential
# -----------------------------
@doc """
    get_differential(amgo::AbstractManifoldFirstOrderObjective, M, p, X)

return the differential of an [`AbstractManifoldFirstOrderObjective`](@ref) `amgo`
on the [`AbstractManifold`](@extref) `M` at the point `p` and with tangent vector `X`.

If the objective has access to a differential, this is called.
Otherwise, if it has access to a gradient, the inner product is used to evaluate the differential
"""
get_differential(amgo::AbstractManifoldFirstOrderObjective, M::AbstractManifold, p, X)

function get_differential(
    amfoo::AbstractManifoldFirstOrderObjective{E,F,DifferentialFunction},
    M::AbstractManifold,
    p,
    X;
) where {E,F}
    return amfoo.first_order!!.diff_f!!(M, p, X)
end
function get_differential(
    amfoo::AbstractManifoldFirstOrderObjective{
        E,F,Tuple{GradientFunction,DifferentialFunction}
    },
    M::AbstractManifold,
    p,
    X;
) where {E,F}
    return amfoo.first_order!![2].diff_f!!(M, p, X)
end
# Default – functions, functors as well as GradientFunctions
function get_differential(
    amfoo::AbstractManifoldFirstOrderObjective, M::AbstractManifold, p, X;
)
    return real(inner(M, p, get_gradient(amfoo, M, p, X), X))
end

@doc """
    get_differential_function(amfoo::AbstractManifoldFirstOrderObjective, recursive=false)

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
function get_differential_function(mfoo::ManifoldFirstOrderObjective, recursive=false)
    return (M, p, X) -> get_differential(M, mfoo, p, X)
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
    cgo::ManifoldFirstOrderObjective{AllocatingEvaluation}, recursive=false
)
    return (M, p) -> get_gradient(M, cgo, p)
end
function get_gradient_function(
    cgo::ManifoldFirstOrderObjective{InplaceEvaluation}, recursive=false
)
    return (M, X, p) -> get_gradient!(M, X, cgo, p)
end

# ------------------------------------------------------------------------------------------
#
# Access functions
#  ... that involve the gradient
# ------------------------------------------------------------------------------------------

#
#  Access cost and gradient – TODO rework for cases.
# -----------------------------
function get_cost_and_gradient(
    M::AbstractManifold, cgo::ManifoldFirstOrderObjective{AllocatingEvaluation}, p
)
    return cgo.costgrad!!(M, p)
end
function get_cost_and_gradient(
    M::AbstractManifold, cgo::ManifoldFirstOrderObjective{InplaceEvaluation}, p
)
    X = zero_vector(M, p)
    return cgo.costgrad!!(M, X, p)
end

function get_cost_and_gradient!(
    M::AbstractManifold, X, cgo::ManifoldFirstOrderObjective{AllocatingEvaluation}, p
)
    (c, Y) = cgo.costgrad!!(M, p)
    copyto!(M, X, p, Y)
    return (c, X)
end
function get_cost_and_gradient!(
    M::AbstractManifold, X, cgo::ManifoldFirstOrderObjective{InplaceEvaluation}, p
)
    return cgo.costgrad!!(M, X, p)
end

function get_cost(M::AbstractManifold, cgo::ManifoldFirstOrderObjective, p)
    v, _ = get_cost_and_gradient(M, cgo, p)
    return v
end

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
    get_gradient(M::AbstractManifold, mgo::AbstractManifoldFirstOrderObjective{T}, p)
    get_gradient!(M::AbstractManifold, X, mgo::AbstractManifoldFirstOrderObjective{T}, p)

evaluate the gradient of a [`AbstractManifoldFirstOrderObjective{T}`](@ref) `mgo` at `p`.

The evaluation is done in place of `X` for the `!`-variant.
The `T=`[`AllocatingEvaluation`](@ref) problem might still allocate memory within.
When the non-mutating variant is called with a `T=`[`InplaceEvaluation`](@ref)
memory for the result is allocated.

Note that the order of parameters follows the philosophy of `Manifolds.jl`, namely that
even for the mutating variant, the manifold is the first parameter and the (in-place) tangent
vector `X` comes second.
"""
get_gradient(M::AbstractManifold, mgo::AbstractManifoldFirstOrderObjective, p)

function get_gradient(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p)
    return get_gradient(M, get_objective(admo, false), p)
end
function get_gradient(
    M::AbstractManifold, mgo::AbstractManifoldFirstOrderObjective{AllocatingEvaluation}, p
)
    return mgo.first_order!!(M, p)
end
function get_gradient(
    M::AbstractManifold, mgo::AbstractManifoldFirstOrderObjective{InplaceEvaluation}, p
)
    X = zero_vector(M, p)
    mgo.first_order!!(M, X, p)
    return X
end
function get_gradient(M::AbstractManifold, mcgo::ManifoldFirstOrderObjective, p)
    _, X = get_cost_and_gradient(M, mcgo, p)
    return X
end

function get_gradient!(M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p)
    return get_gradient!(M, X, get_objective(admo, false), p)
end

function get_gradient!(
    M::AbstractManifold,
    X,
    mgo::AbstractManifoldFirstOrderObjective{AllocatingEvaluation},
    p,
)
    copyto!(M, X, p, mgo.first_order!!(M, p))
    return X
end
function get_gradient!(
    M::AbstractManifold, X, mgo::AbstractManifoldFirstOrderObjective{InplaceEvaluation}, p
)
    mgo.first_order!!(M, X, p)
    return X
end
function get_gradient!(M::AbstractManifold, X, mcgo::ManifoldFirstOrderObjective, p)
    get_cost_and_gradient!(M, X, mcgo, p)
    return X
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
