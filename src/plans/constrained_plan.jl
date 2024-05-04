@doc raw"""
    AbstractConstraintType

An abstract type to represent different forms of representing constraints
as well as their gradients.
"""
abstract type AbstractConstraintType end

@doc raw"""
    FunctionConstraint{CT <: AbstractConstraintType}  <: AbstractConstraintType

A type to indicate that constraints are implemented one whole functions,
for example ``g(p) ∈ ℝ^m``.

For the gradient there are two possible variants available:

* [`VectorConstraint`](@ref): ``\operatorname{grad} g\colon \mathcal M \to (T_p\mathcal M)^m``,
  the gradient returns a vector of gradients, one for each component function of ``g``.
* `PowerManifoldTangentConstaint`: ``\operatorname{grad} g\colon \mathcal M \to T_P\mathcal M^m``,
  where ``P = (p,…,p) \in\mathcal M^m``, that is
  the gradient returns a tangent vector on the power manifold.
"""
struct FunctionConstraint{CT<:AbstractConstraintType} <: AbstractConstraintType end

FunctionConstraint() = FunctionConstraint(VectorConstraint())
FunctionConstraint(::CT) where {CT<:AbstractConstraintType} = FunctionConstraint{CT}()

@doc raw"""
    PowerManifoldTangentConstaint <: AbstractConstraintType

Indicate that (some part of) constraints are given on a [`PowerManifold`](@extref)
"""
struct PowerManifoldTangentConstaint <: AbstractConstraintType end

@doc raw"""
    VectorConstraint <: AbstractConstraintType

A type to indicate that (some part of) constraints are given as a vector of functions.
"""
struct VectorConstraint <: AbstractConstraintType end

@doc raw"""
    ConstrainedManifoldObjective{T<:AbstractEvaluationType, C <: ConstraintType} <: AbstractManifoldObjective{T}

Describes the constrained objective
```math
\begin{aligned}
 \operatorname*{arg\,min}_{p ∈\mathcal{M}} & f(p)\\
 \text{subject to } &g_i(p)\leq0 \quad \text{ for all } i=1,…,m,\\
 \quad &h_j(p)=0 \quad \text{ for all } j=1,…,n.
\end{aligned}
```

# Fields

* `objective`: an [`AbstractManifoldObjective`](@ref) representing the unconstrained
  objective, that is contatining cost ``f``, the gradient of the cost ``f`` and maybe the Hessian.
* `equality_constraints`: an [`AbstractManifoldObjective`](@ref) representing the equality constraints
``h: \mathcal M → \mathbb R^n`` also possibly containing its gradient and/or Hessian
* `equality_constraints`: an [`AbstractManifoldObjective`](@ref) representing the equality constraints
``h: \mathcal M → \mathbb R^n`` also possibly containing its gradient and/or Hessian

# Constructors
    ConstrainedManifoldObjective(M::AbstractManifold, f, grad_f;
        g=nothing,
        grad_g=nothing,
        h=nothing,
        grad_h=nothing;
        evaluation=AllocatingEvaluation(),
    )

    ConstrainedManifoldObjective(M::AbstractManifold, mho::AbstractManifoldObjective;
        equality_constraints::Union{Nothing, AbstractManifoldObjective} = nothing,
        inequality_constraints::Union{Nothing, AbstractManifoldObjective} = nothing
    ) where {IMO <: Union{Nothing, AbstractManifoldObjective} EMO <: Union{Nothing, AbstractManifoldObjective}}

TODO: Describe constructors
"""
struct ConstrainedManifoldObjective{
    T<:AbstractEvaluationType,
    CT<:AbstractConstraintType,
    MO<:AbstractManifoldObjective,
    IMO<:Union{Nothing,AbstractManifoldObjective},
    EMO<:Union{Nothing,AbstractManifoldObjective},
} <: AbstractManifoldObjective{T}
    objective::MO
    inequality_constrinats::IMO
    equality_constraints::EMO
end
# Generic f, grad_f -> pass on to new constructor
function ConstrainedManifoldObjective(
    f,
    grad_f,
    g,
    grad_g,
    h,
    grad_h;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    constraint::AbstractConstraintType=if all(
        isnothing(c) || isa(c, AbstractVector) for c in [g, grad_g, h, grad_h]
    )
        VectorConstraint()
    else
        FunctionConstraint()
    end,
    kwargs...,
)
    objective = ManifoldGradientObjective(f, grad_f; evaluation=evaluation)
    if isnothing(g) || isnothing(grad_g)
        eq = nothing
    else
        eq = ManifoldGradientObjective(g, grad_g; evaluation=evaluation)
    end
    if isnothing(h) || isnothing(grad_h)
        ineq = nothing
    else
        ineq = ManifoldGradientObjective(h, grad_h; evaluation=evaluation)
    end
    return ConstrainedManifoldObjective(
        objective;
        equality_constraints=eq,
        inequality_constraints=ineq,
        constraint=constraint,
        kwargs...,
    )
end
function ConstrainedManifoldObjective(
    objective::MO;
    equality_constraints::EMO=nothing,
    inequality_constraints::IMO=nothing;
    constraint_type::ACT=VectorConstraint()kwargs...,
) where {
    ACT<:AbstractConstraintType,
    E<:AbstractEvaluationType,
    MO<:AbstractManifoldObjective{E},
    IMO<:AbstractManifoldObjective{E},
    EMO<:AbstractManifoldObjective{E},
}
    if isnothing(equality_constraints) && isnothing(inequality_constraints)
        @warn """
        Neither the inequality and the equality constraints are provided.
        Consider calling `get_objective()` on this constraint objective
        and only work on the unconstraint objective instead.
        """
    end
    return ConstrainedManifoldObjective{E,ACT,MO,IMO,EMO}(
        objective, equality_constraints, inequality_constraints
    )
end
get_objective(co::ConstrainedManifoldObjective) = co.objective
function get_constraints(mp::AbstractManoptProblem, p)
    return get_constraints(get_manifold(mp), get_objective(mp), p)
end
"""
    get_constraints(M::AbstractManifold, co::ConstrainedManifoldObjective, p)

Return the vector ``(g_1(p),...g_m(p),h_1(p),...,h_n(p))`` from the [`ConstrainedManifoldObjective`](@ref) `P`
containing the values of all constraints at `p`.
"""
function get_constraints(M::AbstractManifold, co::ConstrainedManifoldObjective, p)
    return [get_inequality_constraints(M, co, p), get_equality_constraints(M, co, p)]
end
function get_constraints(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p)
    return get_constraints(M, get_objective(admo, false), p)
end

function get_cost(M::AbstractManifold, co::ConstrainedManifoldObjective, p)
    return get_cost(M, co.objective, p)
end
function get_cost_function(co::ConstrainedManifoldObjective, recursive=false)
    return get_cost_function(co.objective, recursive)
end

function get_equality_constraints(mp::AbstractManoptProblem, p)
    return get_equality_constraints(get_manifold(mp), get_objective(mp), p)
end
@doc raw"""
    get_equality_constraints(M::AbstractManifold, co::ConstrainedManifoldObjective, p)

evaluate all equality constraints ``h(p)`` of ``\bigl(h_1(p), h_2(p),\ldots,h_p(p)\bigr)``
of the [`ConstrainedManifoldObjective`](@ref) ``P`` at ``p``.
"""
get_equality_constraints(M::AbstractManifold, co::ConstrainedManifoldObjective, p)
function get_equality_constraints(
    M::AbstractManifold, co::ConstrainedManifoldObjective{T,FunctionConstraint}, p
) where {T<:AbstractEvaluationType}
    return co.h(M, p)
end
function get_equality_constraints(
    M::AbstractManifold, co::ConstrainedManifoldObjective{T,VectorConstraint}, p
) where {T<:AbstractEvaluationType}
    return [hj(M, p) for hj in co.h]
end
function get_equality_constraints(
    M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p
)
    return get_equality_constraints(M, get_objective(admo, false), p)
end

function get_equality_constraint(mp::AbstractManoptProblem, p, j)
    return get_equality_constraint(get_manifold(mp), get_objective(mp), p, j)
end
@doc raw"""
    get_equality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, j)

evaluate the `j`th equality constraint ``(h(p))_j`` or ``h_j(p)``.

!!! note
    For the [`FunctionConstraint`](@ref) representation this still evaluates all constraints.
"""
get_equality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, j)
function get_equality_constraint(
    M::AbstractManifold, co::ConstrainedManifoldObjective{T,FunctionConstraint}, p, j
) where {T<:AbstractEvaluationType}
    return co.h(M, p)[j]
end
function get_equality_constraint(
    M::AbstractManifold, co::ConstrainedManifoldObjective{T,VectorConstraint}, p, j
) where {T<:AbstractEvaluationType}
    return co.h[j](M, p)
end
function get_equality_constraint(
    M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p, j
)
    return get_equality_constraint(M, get_objective(admo, false), p, j)
end

function get_gradient(M::AbstractManifold, co::ConstrainedManifoldObjective, p)
    return get_gradient(M, co.objective, p)
end
function get_gradient!(M::AbstractManifold, X, co::ConstrainedManifoldObjective, p)
    return get_gradient!(M, X, co.objective, p)
end
function get_gradient_function(co::ConstrainedManifoldObjective, recursive=false)
    return get_gradient_function(co.objective, recursive)
end

function get_inequality_constraints(mp::AbstractManoptProblem, p)
    return get_inequality_constraints(get_manifold(mp), get_objective(mp), p)
end
@doc raw"""
    get_inequality_constraints(M::AbstractManifold, co::ConstrainedManifoldObjective, p)

Evaluate all inequality constraints ``g(p)`` or ``\bigl(g_1(p), g_2(p),\ldots,g_m(p)\bigr)``
of the [`ConstrainedManifoldObjective`](@ref) ``P`` at ``p``.
"""
get_inequality_constraints(M::AbstractManifold, co::ConstrainedManifoldObjective, p)

function get_inequality_constraints(
    M::AbstractManifold, co::ConstrainedManifoldObjective{T,FunctionConstraint}, p
) where {T<:AbstractEvaluationType}
    return co.g(M, p)
end
function get_inequality_constraints(
    M::AbstractManifold, co::ConstrainedManifoldObjective{T,VectorConstraint}, p
) where {T<:AbstractEvaluationType}
    return [gi(M, p) for gi in co.g]
end

function get_inequality_constraints(
    M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p
)
    return get_inequality_constraints(M, get_objective(admo, false), p)
end

function get_inequality_constraint(mp::AbstractManoptProblem, p, i)
    return get_inequality_constraint(get_manifold(mp), get_objective(mp), p, i)
end
@doc raw"""
    get_inequality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, i)

evaluate one equality constraint ``(g(p))_i`` or ``g_i(p)``.

!!! note
    For the [`FunctionConstraint`](@ref) representation this still evaluates all constraints.
"""
get_inequality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, i)
function get_inequality_constraint(
    M::AbstractManifold, co::ConstrainedManifoldObjective{T,FunctionConstraint}, p, i
) where {T<:AbstractEvaluationType}
    return co.g(M, p)[i]
end
function get_inequality_constraint(
    M::AbstractManifold, co::ConstrainedManifoldObjective{T,VectorConstraint}, p, i
) where {T<:AbstractEvaluationType}
    return co.g[i](M, p)
end
function get_inequality_constraint(
    M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p, i
)
    return get_inequality_constraint(M, get_objective(admo, false), p, i)
end

function get_grad_equality_constraint(mp::AbstractManoptProblem, p, j)
    return get_grad_equality_constraint(get_manifold(mp), get_objective(mp), p, j)
end
@doc raw"""
    get_grad_equality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, j)

evaluate the gradient of the `j` th equality constraint ``(\operatorname{grad} h(p))_j`` or ``\operatorname{grad} h_j(x)``.

!!! note
    For the [`FunctionConstraint`](@ref) variant of the problem, this function still evaluates the full gradient.
    For the [`InplaceEvaluation`](@ref) and [`FunctionConstraint`](@ref) of the problem, this function currently also calls [`get_equality_constraints`](@ref),
    since this is the only way to determine the number of constraints. It also allocates a full tangent vector.
"""
get_grad_equality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, j)
function get_grad_equality_constraint(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective{AllocatingEvaluation,FunctionConstraint},
    p,
    j,
)
    return co.grad_h!!(M, p)[j]
end
function get_grad_equality_constraint(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective{AllocatingEvaluation,VectorConstraint},
    p,
    j,
)
    return co.grad_h!![j](M, p)
end
function get_grad_equality_constraint(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective{InplaceEvaluation,FunctionConstraint},
    p,
    j,
)
    X = [zero_vector(M, p) for _ in 1:length(co.h(M, p))]
    co.grad_h!!(M, X, p)
    return X[j]
end
function get_grad_equality_constraint(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective{InplaceEvaluation,VectorConstraint},
    p,
    j,
)
    X = zero_vector(M, p)
    co.grad_h!![j](M, X, p)
    return X
end
function get_grad_equality_constraint(
    M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p, j
)
    return get_grad_equality_constraint(M, get_objective(admo, false), p, j)
end

function get_grad_equality_constraint!(mp::AbstractManoptProblem, X, p, j)
    return get_grad_equality_constraint!(get_manifold(mp), X, get_objective(mp), p, j)
end
@doc raw"""
    get_grad_equality_constraint!(M::AbstractManifold, X, co::ConstrainedManifoldObjective, p, j)

Evaluate the gradient of the `j`th equality constraint ``(\operatorname{grad} h(x))_j`` or ``\operatorname{grad} h_j(x)`` in place of ``X``

!!! note
    For the [`FunctionConstraint`](@ref) variant of the problem, this function still evaluates the full gradient.
    For the [`InplaceEvaluation`](@ref) of the [`FunctionConstraint`](@ref) of the problem, this function currently also calls [`get_inequality_constraints`](@ref),
    since this is the only way to determine the number of constraints and allocates a full vector of tangent vectors
"""
get_grad_equality_constraint!(
    M::AbstractManifold, X, co::ConstrainedManifoldObjective, p, j
)
function get_grad_equality_constraint!(
    M::AbstractManifold,
    X,
    co::ConstrainedManifoldObjective{AllocatingEvaluation,FunctionConstraint},
    p,
    j,
)
    copyto!(M, X, p, co.grad_h!!(M, p)[j])
    return X
end
function get_grad_equality_constraint!(
    M::AbstractManifold,
    X,
    co::ConstrainedManifoldObjective{AllocatingEvaluation,VectorConstraint},
    p,
    j,
)
    copyto!(M, X, co.grad_h!![j](M, p))
    return X
end
function get_grad_equality_constraint!(
    M::AbstractManifold,
    X,
    co::ConstrainedManifoldObjective{InplaceEvaluation,FunctionConstraint},
    p,
    j,
)
    Y = [zero_vector(M, p) for _ in 1:length(co.h(M, p))]
    co.grad_h!!(M, Y, p)
    copyto!(M, X, p, Y[j])
    return X
end
function get_grad_equality_constraint!(
    M::AbstractManifold,
    X,
    co::ConstrainedManifoldObjective{InplaceEvaluation,VectorConstraint},
    p,
    j,
)
    co.grad_h!![j](M, X, p)
    return X
end
function get_grad_equality_constraint!(
    M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p, j
)
    return get_grad_equality_constraint!(M, X, get_objective(admo, false), p, j)
end

function get_grad_equality_constraints(mp::AbstractManoptProblem, p)
    return get_grad_equality_constraints(get_manifold(mp), get_objective(mp), p)
end
@doc raw"""
    get_grad_equality_constraints(M::AbstractManifold, co::ConstrainedManifoldObjective, p)

evaluate all gradients of the equality constraints ``\operatorname{grad} h(x)`` or ``\bigl(\operatorname{grad} h_1(x), \operatorname{grad} h_2(x),\ldots, \operatorname{grad}h_n(x)\bigr)``
of the [`ConstrainedManifoldObjective`](@ref) `P` at `p`.

!!! note
    For the [`InplaceEvaluation`](@ref) and [`FunctionConstraint`](@ref) variant of the problem,
    this function currently also calls [`get_equality_constraints`](@ref),
    since this is the only way to determine the number of constraints.
"""
get_grad_equality_constraints(M::AbstractManifold, co::ConstrainedManifoldObjective, p)
function get_grad_equality_constraints(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective{AllocatingEvaluation,FunctionConstraint},
    p,
)
    return co.grad_h!!(M, p)
end
function get_grad_equality_constraints(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective{AllocatingEvaluation,VectorConstraint},
    p,
)
    return [grad_hi(M, p) for grad_hi in co.grad_h!!]
end
function get_grad_equality_constraints(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective{InplaceEvaluation,FunctionConstraint},
    p,
)
    X = [zero_vector(M, p) for _ in 1:length(co.h(M, p))]
    co.grad_h!!(M, X, p)
    return X
end
function get_grad_equality_constraints(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective{InplaceEvaluation,VectorConstraint},
    p,
)
    X = [zero_vector(M, p) for _ in 1:length(co.h)]
    [grad_hi(M, Xj, p) for (Xj, grad_hi) in zip(X, co.grad_h!!)]
    return X
end
function get_grad_equality_constraints(
    M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p
)
    return get_grad_equality_constraints(M, get_objective(admo, false), p)
end

function get_grad_equality_constraints!(mp::AbstractManoptProblem, X, p)
    return get_grad_equality_constraints!(get_manifold(mp), X, get_objective(mp), p)
end
@doc raw"""
    get_grad_equality_constraints!(M::AbstractManifold, X, co::ConstrainedManifoldObjective, p)

evaluate all gradients of the equality constraints ``\operatorname{grad} h(p)`` or ``\bigl(\operatorname{grad} h_1(p), \operatorname{grad} h_2(p),\ldots,\operatorname{grad} h_n(p)\bigr)``
of the [`ConstrainedManifoldObjective`](@ref) ``P`` at ``p`` in place of `X``, which is a vector of ``n`` tangent vectors.
"""
function get_grad_equality_constraints!(
    M::AbstractManifold,
    X,
    co::ConstrainedManifoldObjective{AllocatingEvaluation,FunctionConstraint},
    p,
)
    copyto!.(Ref(M), X, Ref(p), co.grad_h!!(M, p))
    return X
end
function get_grad_equality_constraints!(
    M::AbstractManifold,
    X,
    co::ConstrainedManifoldObjective{AllocatingEvaluation,VectorConstraint},
    p,
)
    for (Xj, grad_hj) in zip(X, co.grad_h!!)
        copyto!(M, Xj, grad_hj(M, p))
    end
    return X
end
function get_grad_equality_constraints!(
    M::AbstractManifold,
    X,
    co::ConstrainedManifoldObjective{InplaceEvaluation,FunctionConstraint},
    p,
)
    co.grad_h!!(M, X, p)
    return X
end
function get_grad_equality_constraints!(
    M::AbstractManifold,
    X,
    co::ConstrainedManifoldObjective{InplaceEvaluation,VectorConstraint},
    p,
)
    for (Xj, grad_hj) in zip(X, co.grad_h!!)
        grad_hj(M, Xj, p)
    end
    return X
end
function get_grad_equality_constraints!(
    M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p
)
    return get_grad_equality_constraints!(M, X, get_objective(admo, false), p)
end

function get_grad_inequality_constraint(mp::AbstractManoptProblem, p, i)
    return get_grad_inequality_constraint(get_manifold(mp), get_objective(mp), p, i)
end
@doc raw"""
    get_grad_inequality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, i)

Evaluate the gradient of the `i` th inequality constraints ``(\operatorname{grad} g(x))_i`` or ``\operatorname{grad} g_i(x)``.

!!! note
    For the [`FunctionConstraint`](@ref) variant of the problem, this function still evaluates the full gradient.
    For the [`InplaceEvaluation`](@ref) and [`FunctionConstraint`](@ref) of the problem, this function currently also calls [`get_inequality_constraints`](@ref),
    since this is the only way to determine the number of constraints.
"""
get_grad_inequality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, i)
function get_grad_inequality_constraint(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective{AllocatingEvaluation,FunctionConstraint},
    p,
    i,
)
    return co.grad_g!!(M, p)[i]
end
function get_grad_inequality_constraint(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective{AllocatingEvaluation,VectorConstraint},
    p,
    i,
)
    return co.grad_g!![i](M, p)
end
function get_grad_inequality_constraint(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective{InplaceEvaluation,FunctionConstraint},
    p,
    i,
)
    X = [zero_vector(M, p) for _ in 1:length(co.g(M, p))]
    co.grad_g!!(M, X, p)
    return X[i]
end
function get_grad_inequality_constraint(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective{InplaceEvaluation,VectorConstraint},
    p,
    i,
)
    X = zero_vector(M, p)
    co.grad_g!![i](M, X, p)
    return X
end
function get_grad_inequality_constraint(
    M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p, i
)
    return get_grad_inequality_constraint(M, get_objective(admo, false), p, i)
end

function get_grad_inequality_constraint!(mp::AbstractManoptProblem, X, p, i)
    return get_grad_inequality_constraint!(get_manifold(mp), X, get_objective(mp), p, i)
end
@doc raw"""
    get_grad_inequality_constraint!(P, X, p, i)

Evaluate the gradient of the `i`th inequality constraints ``(\operatorname{grad} g(x))_i`` or ``\operatorname{grad} g_i(x)``
of the [`ConstrainedManifoldObjective`](@ref) `P` in place of ``X``

!!! note
    For the [`FunctionConstraint`](@ref) variant of the problem, this function still evaluates the full gradient.
    For the [`InplaceEvaluation`](@ref) and [`FunctionConstraint`](@ref) of the problem, this function currently also calls [`get_inequality_constraints`](@ref),
  since this is the only way to determine the number of constraints.
evaluate all gradients of the inequality constraints ``\operatorname{grad} h(x)`` or ``\bigl(g_1(x), g_2(x),\ldots,g_m(x)\bigr)``
of the [`ConstrainedManifoldObjective`](@ref) ``p`` at ``x`` in place of `X``, which is a vector of ``m`` tangent vectors .
"""
function get_grad_inequality_constraint!(
    M::AbstractManifold,
    X,
    co::ConstrainedManifoldObjective{AllocatingEvaluation,FunctionConstraint},
    p,
    i,
)
    copyto!(M, X, p, co.grad_g!!(M, p)[i])
    return X
end
function get_grad_inequality_constraint!(
    M::AbstractManifold,
    X,
    co::ConstrainedManifoldObjective{AllocatingEvaluation,VectorConstraint},
    p,
    i,
)
    copyto!(M, X, co.grad_g!![i](M, p))
    return X
end
function get_grad_inequality_constraint!(
    M::AbstractManifold,
    X,
    co::ConstrainedManifoldObjective{InplaceEvaluation,FunctionConstraint},
    p,
    i,
)
    Y = [zero_vector(M, p) for _ in 1:length(co.g(M, p))]
    co.grad_g!!(M, Y, p)
    copyto!(M, X, p, Y[i])
    return X
end
function get_grad_inequality_constraint!(
    M::AbstractManifold,
    X,
    co::ConstrainedManifoldObjective{InplaceEvaluation,VectorConstraint},
    p,
    i,
)
    co.grad_g!![i](M, X, p)
    return X
end
function get_grad_inequality_constraint!(
    M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p, i
)
    return get_grad_inequality_constraint!(M, X, get_objective(admo, false), p, i)
end

function get_grad_inequality_constraints(mp::AbstractManoptProblem, p)
    return get_grad_inequality_constraints(get_manifold(mp), get_objective(mp), p)
end
@doc raw"""
    get_grad_inequality_constraints(M::AbstractManifold, co::ConstrainedManifoldObjective, p)

evaluate all gradients of the inequality constraints ``\operatorname{grad} g(p)`` or ``\bigl(\operatorname{grad} g_1(p), \operatorname{grad} g_2(p),…,\operatorname{grad} g_m(p)\bigr)``
of the [`ConstrainedManifoldObjective`](@ref) ``P`` at ``p``.

!!! note
   for the [`InplaceEvaluation`](@ref) and [`FunctionConstraint`](@ref) variant of the problem,
   this function currently also calls [`get_equality_constraints`](@ref),
   since this is the only way to determine the number of constraints.
"""
get_grad_inequality_constraints(M::AbstractManifold, co::ConstrainedManifoldObjective, x)
function get_grad_inequality_constraints(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective{AllocatingEvaluation,FunctionConstraint},
    p,
)
    return co.grad_g!!(M, p)
end
function get_grad_inequality_constraints(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective{AllocatingEvaluation,VectorConstraint},
    p,
)
    return [grad_gi(M, p) for grad_gi in co.grad_g!!]
end
function get_grad_inequality_constraints(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective{InplaceEvaluation,FunctionConstraint},
    p,
)
    X = [zero_vector(M, p) for _ in 1:length(co.g(M, p))]
    co.grad_g!!(M, X, p)
    return X
end
function get_grad_inequality_constraints(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective{InplaceEvaluation,VectorConstraint},
    p,
)
    X = [zero_vector(M, p) for _ in 1:length(co.g)]
    [grad_gi(M, Xi, p) for (Xi, grad_gi) in zip(X, co.grad_g!!)]
    return X
end
function get_grad_inequality_constraints(
    M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p
)
    return get_grad_inequality_constraints(M, get_objective(admo, false), p)
end

function get_grad_inequality_constraints!(mp::AbstractManoptProblem, X, p)
    return get_grad_inequality_constraints!(get_manifold(mp), X, get_objective(mp), p)
end
@doc raw"""
    get_grad_inequality_constraints!(M::AbstractManifold, X, co::ConstrainedManifoldObjective, p)

evaluate all gradients of the inequality constraints ``\operatorname{grad} g(x)`` or ``\bigl(\operatorname{grad} g_1(x), \operatorname{grad} g_2(x),\ldots,\operatorname{grad} g_m(x)\bigr)``
of the [`ConstrainedManifoldObjective`](@ref) `P` at `p` in place of `X`, which is a vector of ``m`` tangent vectors.
"""
function get_grad_inequality_constraints!(
    M::AbstractManifold,
    X,
    co::ConstrainedManifoldObjective{AllocatingEvaluation,FunctionConstraint},
    p,
)
    copyto!.(Ref(M), X, Ref(p), co.grad_g!!(M, p))
    return X
end
function get_grad_inequality_constraints!(
    M::AbstractManifold,
    X,
    co::ConstrainedManifoldObjective{AllocatingEvaluation,VectorConstraint},
    p,
)
    for (Xi, grad_gi) in zip(X, co.grad_g!!)
        copyto!(M, Xi, grad_gi(M, p))
    end
    return X
end
function get_grad_inequality_constraints!(
    M::AbstractManifold,
    X,
    co::ConstrainedManifoldObjective{InplaceEvaluation,FunctionConstraint},
    p,
)
    co.grad_g!!(M, X, p)
    return X
end
function get_grad_inequality_constraints!(
    M::AbstractManifold,
    X,
    co::ConstrainedManifoldObjective{InplaceEvaluation,VectorConstraint},
    p,
)
    for (Xi, grad_gi!) in zip(X, co.grad_g!!)
        grad_gi!(M, Xi, p)
    end
    return X
end
function get_grad_inequality_constraints!(
    M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p
)
    return get_grad_inequality_constraints!(M, X, get_objective(admo, false), p)
end

function get_hessian(M::AbstractManifold, co::ConstrainedManifoldObjective, p, X)
    return get_hessian(M, co.objective, p, X)
end
function get_hessian!(M::AbstractManifold, Y, co::ConstrainedManifoldObjective, p, X)
    return get_hessian!(M, Y, co.objective, p, X)
end
function get_hessian_function(co::ConstrainedManifoldObjective, recursive=false)
    return get_hessian_function(co.objective, recursive)
end

function Base.show(
    io::IO, ::ConstrainedManifoldObjective{E,V}
) where {E<:AbstractEvaluationType,V}
    return print(io, "ConstrainedManifoldObjective{$E,$V}.")
end
