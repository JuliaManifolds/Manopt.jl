@doc raw"""
    ConstraintType

An abstract type to represent different forms of representing constraints
"""
abstract type ConstraintType end

@doc raw"""
    FunctionConstraint <: ConstraintType

A type to indicate that constraints are implemented one whole functions,
for example ``g(p) ∈ ℝ^m``.
"""
struct FunctionConstraint <: ConstraintType end

@doc raw"""
    VectorConstraint <: ConstraintType

A type to indicate that constraints are implemented a  vector of functions,
for example ``g_i(p) ∈ ℝ, i=1,…,m``.
"""
struct VectorConstraint <: ConstraintType end

@doc raw"""
    ConstrainedManifoldObjective{T<:AbstractEvaluationType, C <: ConstraintType Manifold} <: AbstractManifoldObjective{T}

Describes the constrained objective
```math
\begin{aligned}
 \operatorname*{arg\,min}_{p ∈\mathcal{M}} & f(p)\\
 \text{subject to } &g_i(p)\leq0 \quad \text{ for all } i=1,…,m,\\
 \quad &h_j(p)=0 \quad \text{ for all } j=1,…,n.
\end{aligned}
```

# Fields

* `objective`: a [`AbstractManifoldObjective`](@ref) representing the unconstrained
  objective, that is contatining cost ``f``, the gradient of the cost ``f`` and maybe the Hessian.
* `g` the inequality constraints
* `grad_g!!` the gradient of the inequality constraints
* `h` the equality constraints
* `grad_h!!` the gradient of the equality constraints

It consists of

* an cost function ``f(p)``
* the gradient of ``f``, ``\operatorname{grad}f(p)``
* inequality constraints ``g(p)``, either a function `g` returning a vector or a vector `[g1, g2, ..., gm]` of functions.
* equality constraints ``h(p)``, either a function `h` returning a vector or a vector `[h1, h2, ..., hn]` of functions.
* gradients of the inequality constraints ``\operatorname{grad}g(p) ∈ (T_p\mathcal M)^m``, either a function or a vector of functions.
* gradients of the equality constraints ``\operatorname{grad}h(p) ∈ (T_p\mathcal M)^n``, either a function or a vector of functions.

There are two ways to specify the constraints ``g`` and ``h``.

1. as one `Function` returning a vector in ``ℝ^m`` and ``ℝ^n`` respectively.
   This might be easier to implement but requires evaluating all constraints even if only one is needed.
2. as a `AbstractVector{<:Function}` where each function returns a real number.
   This requires each constraint to be implemented as a single function, but it is possible to evaluate also only a single constraint.

The gradients ``\operatorname{grad}g``, ``\operatorname{grad}h`` have to follow the
same form. Additionally they can be implemented as in-place functions or as allocating ones.
The gradient ``\operatorname{grad}F`` has to be the same kind.
This difference is indicated by the `evaluation` keyword.

# Constructors

    ConstrainedManifoldObjective(objective, g, grad_g, h, grad_h)
    ConstrainedManifoldObjective(M::AbstractManifold, objective;
        g=nothing, grad_g=nothing, h=nothing, grad_h=nothing;
    )

Specify that the [`AbstractManifoldObjective`](@ref) `objective` representing an
unconstrained problem and the constraints `g` and `h` with their gradients `grad_g` and `grad_h``,
respectively.
Note that the [`AbstractEvaluationType`](@ref) from the objective has also to hold
for the (gradients of the) costraints – they are either both allocation or in-place.
The gradients can be single functions or vectors of functions as described above.
If the objective does not have inequality constraints, you can set `g` and `grad_g` to `nothing`.
If the problem does not have equality constraints, you can set `h` and `grad_h` to `nothing` or leave them out.
It is not possible to set all four to nothing, since this is equivalent to just considering `objective` independently.

    ConstrainedManifoldObjective(f, grad_f, g, grad_g, h, grad_h;
        evaluation=AllocatingEvaluation(),
    )

Where `f, g, h` describe the cost, inequality and equality constraints, respectively, as
described previously and `grad_f, grad_g, grad_h` are the corresponding gradient functions in
one of the 4 formats.

    ConstrainedManifoldObjective(M::AbstractManifold, f, grad_f;
        g=nothing,
        grad_g=nothing,
        h=nothing,
        grad_h=nothing;
        evaluation=AllocatingEvaluation(),
    )

A keyword argument variant of the preceding constructor, where you can leave out either
`g` and `grad_g` or `h` and `grad_g` but not both pairs.

"""
struct ConstrainedManifoldObjective{
    T<:AbstractEvaluationType,CT<:ConstraintType,TMO<:AbstractManifoldObjective,TG,GG,TH,GH
} <: AbstractManifoldObjective{T}
    objective::TMO
    g::TG
    grad_g!!::GG
    h::TH
    grad_h!!::GH
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
)
    objective = ManifoldGradientObjective(f, grad_f; evaluation=evaluation)
    return ConstrainedManifoldObjective(objective, g, grad_g, h, grad_h)
end
#
# Constructors I: functions
#
function ConstrainedManifoldObjective(
    objective::TMO, g::Function, grad_g::Function, h::Function, grad_h::Function; kwargs...
) where {E<:AbstractEvaluationType,TMO<:AbstractManifoldObjective{E}}
    return ConstrainedManifoldObjective{
        E,FunctionConstraint,TMO,typeof(g),typeof(grad_g),typeof(h),typeof(grad_h)
    }(
        objective, g, grad_g, h, grad_h
    )
end
# Function without inequality constraints
function ConstrainedManifoldObjective(
    objective::TMO, ::Nothing, ::Nothing, h::Function, grad_h::Function; kwargs...
) where {E<:AbstractEvaluationType,TMO<:AbstractManifoldObjective{E}}
    local_g = (M, p) -> []
    local_grad_g = E === AllocatingEvaluation ? (M, p) -> [] : (M, X, p) -> []
    return ConstrainedManifoldObjective{
        E,
        FunctionConstraint,
        TMO,
        typeof(local_g),
        typeof(local_grad_g),
        typeof(h),
        typeof(grad_h),
    }(
        objective, local_g, local_grad_g, h, grad_h
    )
end
# No equality constraints
function ConstrainedManifoldObjective(
    objective::TMO,
    g::Function,
    grad_g::Function,
    ::Nothing=nothing,
    ::Nothing=nothing;
    kwargs...,
) where {E<:AbstractEvaluationType,TMO<:AbstractManifoldObjective{E}}
    local_h = (M, p) -> []
    local_grad_h = E === AllocatingEvaluation ? (M, p) -> [] : (M, X, p) -> []
    return ConstrainedManifoldObjective{
        E,
        FunctionConstraint,
        TMO,
        typeof(g),
        typeof(grad_g),
        typeof(local_h),
        typeof(local_grad_h),
    }(
        objective, g, grad_g, local_h, local_grad_h
    )
end
#
# Vectors
#
function ConstrainedManifoldObjective(
    objective::TMO,
    g::AbstractVector{<:Function},
    grad_g::AbstractVector{<:Function},
    h::AbstractVector{<:Function},
    grad_h::AbstractVector{<:Function};
    kwargs...,
) where {E<:AbstractEvaluationType,TMO<:AbstractManifoldObjective{E}}
    return ConstrainedManifoldObjective{
        E,VectorConstraint,TMO,typeof(g),typeof(grad_g),typeof(h),typeof(grad_h)
    }(
        objective, g, grad_g, h, grad_h
    )
end
# equality not provided
function ConstrainedManifoldObjective(
    objective::TMO,
    ::Nothing,
    ::Nothing,
    h::AbstractVector{<:Function},
    grad_h::AbstractVector{<:Function};
    kwargs...,
) where {E<:AbstractEvaluationType,TMO<:AbstractManifoldObjective{E}}
    local_g = Vector{Function}()
    local_grad_g = Vector{Function}()
    return ConstrainedManifoldObjective{
        E,VectorConstraint,TMO,typeof(local_g),typeof(local_grad_g),typeof(h),typeof(grad_h)
    }(
        objective, local_g, local_grad_g, h, grad_h
    )
end
# No equality constraints provided
function ConstrainedManifoldObjective(
    objective::TMO,
    g::AbstractVector{<:Function},
    grad_g::AbstractVector{<:Function},
    ::Nothing,
    ::Nothing;
    kwargs...,
) where {E<:AbstractEvaluationType,TMO<:AbstractManifoldObjective{E}}
    local_h = Vector{Function}()
    local_grad_h = Vector{Function}()
    return ConstrainedManifoldObjective{
        E,VectorConstraint,TMO,typeof(g),typeof(grad_g),typeof(local_h),typeof(local_grad_h)
    }(
        objective, g, grad_g, local_h, local_grad_h
    )
end
#
# Neither equality nor inequality yields an error
#
function ConstrainedManifoldObjective(
    ::TF, ::TGF, ::Nothing, ::Nothing, ::Nothing, ::Nothing; kwargs...
) where {TF,TGF}
    return error(
        """
  Neither inequality constraints `g`, `grad_g` nor equality constraints `h`, `grad_h` provided.
  If you have an unconstraint problem, maybe consider using a `ManifoldGradientObjective` instead.
  """,
    )
end
function ConstrainedManifoldObjective(
    ::TMO, ::Nothing, ::Nothing, ::Nothing, ::Nothing; kwargs...
) where {TMO}
    return error(
        """
  Neither inequality constraints `g`, `grad_g` nor equality constraints `h`, `grad_h` provided.
  If you have an unconstraint problem, maybe consider using a `ManifoldGradientObjective` instead.
  """,
    )
end
function ConstrainedManifoldObjective(
    f::TF,
    grad_f::TGF;
    g=nothing,
    grad_g=nothing,
    h=nothing,
    grad_h=nothing,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
) where {TF,TGF}
    return ConstrainedManifoldObjective(
        f, grad_f, g, grad_g, h, grad_h; evaluation=evaluation
    )
end
function ConstrainedManifoldObjective(
    objective::TMO;
    g=nothing,
    grad_g=nothing,
    h=nothing,
    grad_h=nothing,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
) where {TMO}
    return ConstrainedManifoldObjective(
        objective, g, grad_g, h, grad_h; evaluation=evaluation
    )
end
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

"""
    DebugFeasibility <: DebugAction

Display information about the feasibility of the current iterate

# Fields
* `atol`:   absolute tolerance for when either equality or inequality constraints are counted as violated
* `format`: a vector of symbols and string formatting the output
* `io`:     default stream to print the debug to.

The following symbols are filled with values

* `:Feasbile` display true or false depending on whether the iterate is feasible
* `:FeasbileEq` display `=` or `≠` equality constraints are fulfilled or not
* `:FeasbileInEq` display `≤` or `>` inequality constraints are fulfilled or not
* `:NumEq` display the number of equality constraints infeasible
* `:NumEqNz` display the number of equality constraints infeasible if exists
* `:NumIneq` display the number of inequality constraints infeasible
* `:NumIneqNz` display the number of inequality constraints infeasible if exists
* `:TotalEq` display the sum of how much the equality constraints are violated
* `:TotalInEq` display the sum of how much the inequality constraints are violated

format to print the output.

# Constructor

DebugFeasibility(
    format=["feasible: ", :Feasible];
    io::IO=stdout,
    atol=1e-13
)

"""
mutable struct DebugFeasibility <: DebugAction
    atol::Float64
    format::Vector{Union{String,Symbol}}
    io::IO
    function DebugFeasibility(format=["feasible: ", :Feasible]; io::IO=stdout, atol=1e-13)
        return new(atol, format, io)
    end
end
function (d::DebugFeasibility)(
    mp::AbstractManoptProblem, st::AbstractManoptSolverState, i::Int
)
    s = ""
    p = get_iterate(st)
    eqc = get_equality_constraints(mp, p)
    eqc_nz = eqc[abs.(eqc) .> d.atol]
    ineqc = get_inequality_constraints(mp, p)
    ineqc_pos = ineqc[ineqc .> d.atol]
    feasible = (length(eqc_nz) == 0) && (length(ineqc_pos) == 0)
    n_eq = length(eqc_nz)
    n_ineq = length(ineqc_pos)
    for f in d.format
        (f isa String) && (s *= f)
        (f === :Feasible) && (s *= feasible ? "Yes" : "No")
        (f === :FeasibleEq) && (s *= n_eq == 0 ? "=" : "≠")
        (f === :FeasibleIneq) && (s *= n_ineq == 0 ? "≤" : ">")
        (f === :NumEq) && (s *= "$(n_eq)")
        (f === :NumEqNz) && (s *= n_eq == 0 ? "" : "$(n_eq)")
        (f === :NumIneq) && (s *= "$(n_ineq)")
        (f === :NumIneqNz) && (s *= n_ineq == 0 ? "" : "$(n_ineq)")
        (f === :TotalEq) && (s *= "$(sum(abs.(eqc_nz);init=0.0))")
        (f === :TotalInEq) && (s *= "$(sum(ineq_pos;init=0.0))")
    end
    print(d.io, (i > 0) ? s : "")
    return nothing
end
function show(io::IO, d::DebugFeasibility)
    sf = "[" * (join([e isa String ? "$e" : ":$e" for e in d.format], ", ")) * "]"
    return print(io, "DebugFeasibility($sf; atol=$(d.atol))")
end
function status_summary(d::DebugFeasibility)
    sf = "[" * (join([e isa String ? "$e" : ":$e" for e in d.format], ", ")) * "]"
    return "(:Feasibility, $sf)"
end
