@doc raw"""
    AbstractManifoldGradientObjective{E<:AbstractEvaluationType, TC, TG} <: AbstractManifoldCostObjective{E, TC}

An abstract type for all objectives that provide a (full) gradient, where
`T` is a [`AbstractEvaluationType`](@ref) for the gradient function.
"""
abstract type AbstractManifoldGradientObjective{E<:AbstractEvaluationType,TC,TG} <:
              AbstractManifoldCostObjective{E,TC} end

@doc raw"""
    get_gradient_function(amgo::AbstractManifoldGradientObjective, recursive=false)

return the function to evaluate (just) the gradient ``\operatorname{grad} f(p)``,
where either the gradient function using the decorator or without the decorator is used.

By default `recursive` is set to `false`, since usually to just pass the gradient function
somewhere, one still wants for example the cached one or the one that still counts calls.

Depending on the [`AbstractEvaluationType`](@ref) `E` this is a function

* `(M, p) -> X` for the [`AllocatingEvaluation`](@ref) case
* `(M, X, p) -> X` for the [`InplaceEvaluation`](@ref) working in-place of `X`.
"""
function get_gradient_function(amgo::AbstractManifoldGradientObjective, recursive=false)
    return amgo.gradient!!
end
function get_gradient_function(admo::AbstractDecoratedManifoldObjective, recursive=false)
    return get_gradient_function(get_objective(admo, recursive))
end

@doc raw"""
    ManifoldGradientObjective{T<:AbstractEvaluationType} <: AbstractManifoldGradientObjective{T}

specify an objective containing a cost and its gradient

# Fields

* `cost`:       a function ``f: \mathcal M → ℝ``
* `gradient!!`: the gradient ``\operatorname{grad}f: \mathcal M → \mathcal T\mathcal M``
  of the cost function ``f``.

Depending on the [`AbstractEvaluationType`](@ref) `T` the gradient can have to forms

* as a function `(M, p) -> X` that allocates memory for `X`, an [`AllocatingEvaluation`](@ref)
* as a function `(M, X, p) -> X` that work in place of `X`, an [`InplaceEvaluation`](@ref)

# Constructors
    ManifoldGradientObjective(cost, gradient; evaluation=AllocatingEvaluation())

# Used with
[`gradient_descent`](@ref), [`conjugate_gradient_descent`](@ref), [`quasi_Newton`](@ref)
"""
struct ManifoldGradientObjective{T<:AbstractEvaluationType,C,G} <:
       AbstractManifoldGradientObjective{T,C,G}
    cost::C
    gradient!!::G
end
function ManifoldGradientObjective(
    cost::C, gradient::G; evaluation::AbstractEvaluationType=AllocatingEvaluation()
) where {C,G}
    return ManifoldGradientObjective{typeof(evaluation),C,G}(cost, gradient)
end

@doc raw"""
    ManifoldCostGradientObjective{T} <: AbstractManifoldObjective{T}

specify an objective containing one function to perform a combined computation of cost and its gradient

# Fields

* `costgrad!!`: a function that computes both the cost ``f: \mathcal M → ℝ``
  and its gradient ``\operatorname{grad}f: \mathcal M → \mathcal T\mathcal M``

Depending on the [`AbstractEvaluationType`](@ref) `T` the gradient can have to forms

* as a function `(M, p) -> (c, X)` that allocates memory for the gradient `X`, an [`AllocatingEvaluation`](@ref)
* as a function `(M, X, p) -> (c, X)` that work in place of `X`, an [`InplaceEvaluation`](@ref)

# Constructors

    ManifoldCostGradientObjective(costgrad; evaluation=AllocatingEvaluation())

# Used with
[`gradient_descent`](@ref), [`conjugate_gradient_descent`](@ref), [`quasi_Newton`](@ref)
"""
struct ManifoldCostGradientObjective{T,CG} <: AbstractManifoldGradientObjective{T,CG,CG}
    costgrad!!::CG
end
function ManifoldCostGradientObjective(
    costgrad::CG; evaluation::AbstractEvaluationType=AllocatingEvaluation()
) where {CG}
    return ManifoldCostGradientObjective{typeof(evaluation),CG}(costgrad)
end

get_cost_function(cgo::ManifoldCostGradientObjective) = (M, p) -> get_cost(M, cgo, p)
function get_gradient_function(cgo::ManifoldCostGradientObjective)
    return (M, p) -> get_gradient(M, cgo, p)
end

#
# and internal helper to make the dispatch nicer
#
function get_cost_and_gradient(
    M::AbstractManifold, cgo::ManifoldCostGradientObjective{AllocatingEvaluation}, p
)
    return cgo.costgrad!!(M, p)
end
function get_cost_and_gradient(
    M::AbstractManifold, cgo::ManifoldCostGradientObjective{InplaceEvaluation}, p
)
    X = zero_vector(M, p)
    return cgo.costgrad!!(M, X, p)
end

function get_cost_and_gradient!(
    M::AbstractManifold, X, cgo::ManifoldCostGradientObjective{AllocatingEvaluation}, p
)
    (c, Y) = cgo.costgrad!!(M, p)
    copyto!(M, X, p, Y)
    return (c, X)
end
function get_cost_and_gradient!(
    M::AbstractManifold, X, cgo::ManifoldCostGradientObjective{InplaceEvaluation}, p
)
    return cgo.costgrad!!(M, X, p)
end

function get_cost(M::AbstractManifold, cgo::ManifoldCostGradientObjective, p)
    v, _ = get_cost_and_gradient(M, cgo, p)
    return v
end

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
    get_gradient(M::AbstractManifold, mgo::AbstractManifoldGradientObjective{T}, p)
    get_gradient!(M::AbstractManifold, X, mgo::AbstractManifoldGradientObjective{T}, p)

evaluate the gradient of a [`AbstractManifoldGradientObjective{T}`](@ref) `mgo` at `p`.

The evaluation is done in place of `X` for the `!`-variant.
The `T=`[`AllocatingEvaluation`](@ref) problem might still allocate memory within.
When the non-mutating variant is called with a `T=`[`InplaceEvaluation`](@ref)
memory for the result is allocated.

Note that the order of parameters follows the philosophy of `Manifolds.jl`, namely that
even for the mutating variant, the manifold is the first parameter and the (in-place) tangent
vector `X` comes second.
"""
get_gradient(M::AbstractManifold, mgo::AbstractManifoldGradientObjective, p)

function get_gradient(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p)
    return get_gradient(M, get_objective(admo, false), p)
end
function get_gradient(
    M::AbstractManifold, mgo::AbstractManifoldGradientObjective{AllocatingEvaluation}, p
)
    return mgo.gradient!!(M, p)
end
function get_gradient(
    M::AbstractManifold, mgo::AbstractManifoldGradientObjective{InplaceEvaluation}, p
)
    X = zero_vector(M, p)
    mgo.gradient!!(M, X, p)
    return X
end
function get_gradient(M::AbstractManifold, mcgo::ManifoldCostGradientObjective, p)
    _, X = get_cost_and_gradient(M, mcgo, p)
    return X
end

function get_gradient!(M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p)
    return get_gradient!(M, X, get_objective(admo, false), p)
end

function get_gradient!(
    M::AbstractManifold, X, mgo::AbstractManifoldGradientObjective{AllocatingEvaluation}, p
)
    copyto!(M, X, p, mgo.gradient!!(M, p))
    return X
end
function get_gradient!(
    M::AbstractManifold, X, mgo::AbstractManifoldGradientObjective{InplaceEvaluation}, p
)
    mgo.gradient!!(M, X, p)
    return X
end
function get_gradient!(M::AbstractManifold, X, mcgo::ManifoldCostGradientObjective, p)
    get_cost_and_gradient!(M, X, mcgo, p)
    return X
end

"""
    X = get_subgradient(M::AbstractManifold, sgo::AbstractManifoldGradientObjective, p)
    get_subgradient!(M::AbstractManifold, X, sgo::AbstractManifoldGradientObjective, p)

Evaluate the subgradient, which for the case of a objective having a gradient, means evaluating the
gradient itself.

While in general, the result might not be deterministic, for this case it is.
"""
function get_subgradient(M::AbstractManifold, agmo::AbstractManifoldGradientObjective, p)
    return get_gradient(M, agmo, p)
end
function get_subgradient!(
    M::AbstractManifold, X, agmo::AbstractManifoldGradientObjective, p
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
"""
struct IdentityUpdateRule <: DirectionUpdateRule end

"""
    MomentumGradient <: DirectionUpdateRule

Append a momentum to a gradient processor, where the last direction and last iterate are
stored and the new is composed as ``η_i = m*η_{i-1}' - s d_i``,
where ``sd_i`` is the current (inner) direction and ``η_{i-1}'`` is the vector transported
last direction multiplied by momentum ``m``.

# Fields

* `p_old`:  remember the last iterate for parallel transporting the last direction
* `momentum`: factor for momentum
* `direction`: internal [`DirectionUpdateRule`](@ref) to determine directions
  to add the momentum to.
* `vector_transport_method`: vector transport method to use
* `X_old`: the last gradient/direction update added as momentum

# Constructors

Add momentum to a gradient problem, where by default just a gradient evaluation is used

    MomentumGradient(
        M::AbstractManifold;
        p=rand(M),
        s::DirectionUpdateRule=IdentityUpdateRule();
        X=zero_vector(M, p),
        momentum=0.2
        vector_transport_method=default_vector_transport_method(M, typeof(p)),
    )

Initialize a momentum gradient rule to `s`, where `p` and `X` are memory for interim values.
"""
mutable struct MomentumGradient{P,T,R<:Real,VTM<:AbstractVectorTransportMethod} <:
               DirectionUpdateRule
    momentum::R
    p_old::P
    direction::DirectionUpdateRule
    vector_transport_method::VTM
    X_old::T
end
function MomentumGradient(
    M::AbstractManifold,
    p::P=rand(M);
    direction::DirectionUpdateRule=IdentityUpdateRule(),
    vector_transport_method::VTM=default_vector_transport_method(M, typeof(p)),
    X=zero_vector(M, p),
    momentum=0.2,
) where {P,VTM<:AbstractVectorTransportMethod}
    return MomentumGradient{P,typeof(X),typeof(momentum),VTM}(
        momentum, p, direction, vector_transport_method, X
    )
end
function (mg::MomentumGradient)(
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
    AverageGradient <: DirectionUpdateRule

Add an average of gradients to a gradient processor. A set of previous directions (from the
inner processor) and the last iterate are stored, average is taken after vector transporting
them to the current iterates tangent space.

# Fields
* `gradients`:               the last `n` gradient/direction updates
* `last_iterate`:            last iterate (needed to transport the gradients)
* `direction`:               internal [`DirectionUpdateRule`](@ref) to determine directions to apply the averaging to
* `vector_transport_method`: vector transport method to use

# Constructors
    AverageGradient(
        M::AbstractManifold,
        p::P=rand(M);
        n::Int=10
        s::DirectionUpdateRule=IdentityUpdateRule();
        gradients = fill(zero_vector(p.M, o.x),n),
        last_iterate = deepcopy(x0),
        vector_transport_method = default_vector_transport_method(M, typeof(p))
    )

Add average to a gradient problem, where

* `n`:                       determines the size of averaging
* `s`:                       is the internal [`DirectionUpdateRule`](@ref) to determine the gradients to store
* `gradients`:               can be pre-filled with some history
* `last_iterate`:            stores the last iterate
* `vector_transport_method`: determines how to transport all gradients to the current iterates tangent space before averaging
"""
mutable struct AverageGradient{P,T,VTM<:AbstractVectorTransportMethod} <:
               DirectionUpdateRule
    gradients::AbstractVector{T}
    last_iterate::P
    direction::DirectionUpdateRule
    vector_transport_method::VTM
end
function AverageGradient(
    M::AbstractManifold,
    p::P=rand(M);
    n::Int=10,
    direction::DirectionUpdateRule=IdentityUpdateRule(),
    gradients=[zero_vector(M, p) for _ in 1:n],
    vector_transport_method::VTM=default_vector_transport_method(M, typeof(p)),
) where {P,VTM}
    return AverageGradient{P,eltype(gradients),VTM}(
        gradients, p, direction, vector_transport_method
    )
end
function (a::AverageGradient)(mp::AbstractManoptProblem, s::AbstractGradientSolverState, k)
    pop!(a.gradients)
    M = get_manifold(mp)
    step, d = a.direction(mp, s, k) #get inner gradient and step
    a.gradients = vcat([deepcopy(d)], a.gradients)
    for i in 1:(length(a.gradients) - 1) #transport & shift in place
        vector_transport_to!(
            M,
            a.gradients[i],
            a.last_iterate,
            a.gradients[i + 1],
            get_iterate(s),
            a.vector_transport_method,
        )
    end
    a.gradients[1] = deepcopy(d)
    copyto!(M, a.last_iterate, get_iterate(s))
    return step, 1 / length(a.gradients) .* sum(a.gradients)
end

@doc raw"""
    Nesterov <: DirectionUpdateRule

## Fields
* `γ`
* `μ` the strong convexity coefficient
* `v` (=``=v_k``, ``v_0=x_0``) an interim point to compute the next gradient evaluation point `y_k`
* `shrinkage` (`= i -> 0.8`) a function to compute the shrinkage ``β_k`` per iterate.

Assume ``f`` is ``L``-Lipschitz and ``μ``-strongly convex. Given

* a step size ``h_k<\frac{1}{L}`` (from the [`GradientDescentState`](@ref)
* a `shrinkage` parameter ``β_k``
* and a current iterate ``x_k``
* as well as the interim values ``γ_k`` and ``v_k`` from the previous iterate.

This compute a Nesterov type update using the following steps, see [ZhangSra:2018](@cite)

1. Compute the positive root ``α_k∈(0,1)`` of ``α^2 = h_k\bigl((1-α_k)γ_k+α_k μ\bigr)``.
2. Set ``\bar γ_k+1 = (1-α_k)γ_k + α_kμ``
3. ``y_k = \operatorname{retr}_{x_k}\Bigl(\frac{α_kγ_k}{γ_k + α_kμ}\operatorname{retr}^{-1}_{x_k}v_k \Bigr)``
4. ``x_{k+1} = \operatorname{retr}_{y_k}(-h_k \operatorname{grad}f(y_k))``
5. ``v_{k+1} = `\operatorname{retr}_{y_k}\Bigl(\frac{(1-α_k)γ_k}{\barγ_k}\operatorname{retr}_{y_k}^{-1}(v_k) - \frac{α_k}{\bar γ_{k+1}}\operatorname{grad}f(y_k) \Bigr)``
6. ``γ_{k+1} = \frac{1}{1+β_k}\bar γ_{k+1}``

Then the direction from ``x_k`` to ``x_k+1`` by ``d = \operatorname{retr}^{-1}_{x_k}x_{k+1}`` is returned.

# Constructor
    Nesterov(M::AbstractManifold, p::P; γ=0.001, μ=0.9, shrinkage = k -> 0.8;
        inverse_retraction_method=LogarithmicInverseRetraction())

Initialize the Nesterov acceleration, where `x0` initializes `v`.
"""
mutable struct Nesterov{P,R<:Real} <: DirectionUpdateRule
    γ::R
    μ::R
    v::P
    shrinkage::Function
    inverse_retraction_method::AbstractInverseRetractionMethod
end
function Nesterov(
    M::AbstractManifold,
    p::P;
    γ::T=0.001,
    μ::T=0.9,
    shrinkage::Function=i -> 0.8,
    inverse_retraction_method::AbstractInverseRetractionMethod=default_inverse_retraction_method(
        M, P
    ),
) where {P,T}
    p_ = _ensure_mutating_variable(p)
    return Nesterov{typeof(p_),T}(γ, μ, copy(M, p_), shrinkage, inverse_retraction_method)
end
function (n::Nesterov)(mp::AbstractManoptProblem, s::AbstractGradientSolverState, k)
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
RecordGradient(ξ::T) where {T} = RecordGradient{T}()
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
