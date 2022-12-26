@doc raw"""
    ConstraintType

An abstract type to represent different forms of representing constraints
"""
abstract type ConstraintType end

@doc raw"""
    FunctionConstraint <: ConstraintType

A type to indicate that constraints are implemented one whole functions,
e.g. ``g(p) ∈ \mathbb R^m``.
"""
struct FunctionConstraint <: ConstraintType end

@doc raw"""
    VectorConstraint <: ConstraintType

A type to indicate that constraints are implemented a  vector of functions,
e.g. ``g_i(p) ∈ \mathbb R, i=1,…,m``.
"""
struct VectorConstraint <: ConstraintType end

@doc raw"""
    ConstrainedObjective{T<:AbstractEvaluationType, C <: ConstraintType Manifold} <: AbstractManifoldObjective{T}

Describes the constrained objective
```math
\begin{aligned}
 \operatorname*{arg\,min}_{p ∈\mathcal{M}} & f(p)\\
 \text{subject to } &g_i(p)\leq0 \quad \text{ for all } i=1,…,m,\\
 \quad &h_j(p)=0 \quad \text{ for all } j=1,…,n.
\end{aligned}
```

It consists of
* an cost function ``f(p)``
* the gradient of ``f``, ``\operatorname{grad}f(p)`` (cf. [`Abstract`](@ref))
* inequality constraints ``g(p)``, either a function `g` returning a vector or a vector `[g1, g2,...,gm]` of functions.
* equality constraints ``h(p)``, either a function `h` returning a vector or a vector `[h1, h2,...,hn]` of functions.
* gradient(s) of the inequality constraints ``\operatorname{grad}g(p) ∈ (T_p\mathcal M)^m``, either a function or a vector of functions.
* gradient(s) of the equality constraints ``\operatorname{grad}h(p) ∈ (T_p\mathcal M)^n``, either a function or a vector of functions.

There are two ways to specify the constraints ``g`` and ``h``.

1. as one `Function` returning a vector in ``\mathbb R^m`` and ``\mathbb R^n`` respecively.
   This might be easier to implement but requires evaluating _all_ constraints even if only one is needed.
2. as a `AbstractVector{<:Function}` where each function returns a real number.
   This requires each constrant to be implemented as a single function, but it is possible to evaluate also only a single constraint.

The gradients ``\operatorname{grad}g``, ``\operatorname{grad}h`` have to follow the
same form. Additionally they can be implemented as in-place functions or as allocating ones.
The gradient ``\operatorname{grad}F`` has to be the same kind.
This difference is indicated by the `evaluation` keyword.

# Constructors

    ConstrainedObjective(f, grad_f, g, grad_g, h, grad_h;
        evaluation=AllocatingEvaluation()
    )

Where `f, g, h` describe the cost, inequality and equality constraints, respecitvely, as
described above and `grad_f, grad_g, grad_h` are the corresponding gradient functions in
one of the 4 formats. If the objective does not have inequality constraints, you can set `G` and `gradG` no `nothing`.
If the problem does not have equality constraints, you can set `H` and `gradH` no `nothing` or leave them out.

    ConstrainedObjective(M::AbstractManifold, F, gradF;
        G=nothing, gradG=nothing, H=nothing, gradH=nothing;
        evaluation=AllocatingEvaluation()
    )

A keyword argument variant of the constructor above, where you can leave out either
`G` and `gradG` _or_ `H` and `gradH` but not both.
"""
struct ConstrainedObjective{
    T<:AbstractEvaluationType,CT<:ConstraintType,TCost,GF,TG,GG,TH,GH
} <: AbstractManifoldGradientObjective{T}
    cost::TCost
    gradient!!::GF
    g::TG
    grad_g!!::GG
    h::TH
    grad_h!!::GH
end
#
# Constructors I: Functions
#
function ConstrainedObjective(
    f::TF,
    grad_f::TGF,
    g::Function,
    grad_g::Function,
    h::Function,
    grad_h::Function;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
) where {TF,TGF}
    return ConstrainedObjective{
        typeof(evaluation),
        FunctionConstraint,
        TF,
        TGF,
        typeof(g),
        typeof(grad_g),
        typeof(h),
        typeof(grad_h),
    }(
        f, grad_f, g, grad_g, h, grad_h
    )
end
# Function without inequality constraints
function ConstrainedObjective(
    f::TF,
    grad_f::TGF,
    ::Nothing,
    ::Nothing,
    h::Function,
    grad_h::Function;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
) where {TF,TGF}
    local_g = (M, p) -> []
    local_grad_g = evaluation === AllocatingEvaluation() ? (M, p) -> [] : (M, X, p) -> []
    return ConstrainedObjective{
        typeof(evaluation),
        FunctionConstraint,
        TF,
        TGF,
        typeof(local_g),
        typeof(local_grad_g),
        typeof(h),
        typeof(grad_h),
    }(
        f, grad_f, local_g, local_grad_g, h, grad_h
    )
end
# No equality constraints
function ConstrainedObjective(
    f::TF,
    grad_f::TGF,
    g::Function,
    grad_h::Function,
    ::Nothing=nothing,
    ::Nothing=nothing;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
) where {TF,TGF}
    local_h = (M, p) -> []
    local_grad_h = evaluation === AllocatingEvaluation() ? (M, p) -> [] : (M, X, p) -> []
    return ConstrainedObjective{
        typeof(evaluation),
        FunctionConstraint,
        TF,
        TGF,
        typeof(g),
        typeof(grad_h),
        typeof(local_h),
        typeof(local_grad_h),
    }(
        f, grad_f, g, grad_h, local_h, local_grad_h
    )
end
#
# Vectors
#
function ConstrainedObjective(
    f::TF,
    grad_f::TGF,
    g::AbstractVector{<:Function},
    grad_g::AbstractVector{<:Function},
    h::AbstractVector{<:Function},
    grad_h::AbstractVector{<:Function};
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
) where {TF,TGF}
    return ConstrainedObjective{
        typeof(evaluation),
        VectorConstraint,
        TF,
        TGF,
        typeof(g),
        typeof(grad_g),
        typeof(h),
        typeof(grad_h),
    }(
        f, grad_f, g, grad_g, h, grad_h
    )
end
# equality not provided
function ConstrainedObjective(
    f::TF,
    grad_f::TGF,
    ::Nothing,
    ::Nothing,
    h::AbstractVector{<:Function},
    grad_h::AbstractVector{<:Function};
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
) where {TF,TGF}
    local_g = Vector{Function}()
    local_grad_g = Vector{Function}()
    return ConstrainedObjective{
        typeof(evaluation),
        VectorConstraint,
        TF,
        TGF,
        typeof(local_g),
        typeof(local_grad_g),
        typeof(h),
        typeof(grad_h),
    }(
        f, grad_f, local_g, local_grad_g, h, grad_h
    )
end
# No eqality constraints provided
function ConstrainedObjective(
    f::TF,
    grad_f::TGF,
    g::AbstractVector{<:Function},
    grad_g::AbstractVector{<:Function},
    ::Nothing,
    ::Nothing;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
) where {TF,TGF}
    local_h = Vector{Function}()
    local_grad_h = Vector{Function}()
    return ConstrainedObjective{
        typeof(evaluation),
        VectorConstraint,
        TF,
        TGF,
        typeof(g),
        typeof(grad_g),
        typeof(local_h),
        typeof(local_grad_h),
    }(
        f, grad_f, g, grad_g, local_h, local_grad_h
    )
end
#
# Neither equality nor inequality yields an error
#
function ConstrainedObjective(
    ::TF, ::TGF, ::Nothing, ::Nothing, ::Nothing, ::Nothing; kwargs...
) where {TF,TGF}
    return error(
        """
  Neither inequality constraints `g`, `grad_g` nor equality constraints `h`, `grad_h` provided.
  If you have an unconstraint problem, maybe consider using a `ManifoldGradientObjective` instead.
  """,
    )
end
function ConstrainedObjective(
    f::TF,
    grad_f::TGF;
    g=nothing,
    grad_g=nothing,
    h=nothing,
    grad_h=nothing,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
) where {TF,TGF}
    return ConstrainedObjective(f, grad_f, g, grad_g, h, grad_h; evaluation=evaluation)
end

function get_constraints(mp::AbstractManoptProblem, p)
    return get_constraints(get_manifold(mp), get_objective(mp), p)
end
"""
    get_constraints(M::AbstractManifold, co::ConstrainedObjective, p)

Return the vector ``(g_1(p),...g_m(p),h_1(p),...,h_n(p))`` from the [`ConstrainedObjective`](@ref) `P`
containing the values of all constraints at `p`.
"""
function get_constraints(M::AbstractManifold, co::ConstrainedObjective, p)
    return [get_inequality_constraints(M, co, p), get_equality_constraints(M, co, p)]
end

function get_equality_constraints(mp::AbstractManoptProblem, p)
    return get_equality_constraints(get_manifold(mp), get_objective(mp), p)
end
@doc raw"""
    get_equality_constraints(M::AbstractManifold, co::ConstrainedObjective, p)

evaluate all equality constraints ``h(p)`` of ``\bigl(h_1(p), h_2(p),\ldots,h_p(p)\bigr)``
of the [`ConstrainedObjective`](@ref) ``P`` at ``p``.
"""
get_equality_constraints(M::AbstractManifold, co::ConstrainedObjective, p)
function get_equality_constraints(
    M::AbstractManifold, co::ConstrainedObjective{T,FunctionConstraint}, p
) where {T<:AbstractEvaluationType}
    return co.h(M, p)
end
function get_equality_constraints(
    M::AbstractManifold, co::ConstrainedObjective{T,VectorConstraint}, p
) where {T<:AbstractEvaluationType}
    return [hj(M, p) for hj in co.h]
end

function get_equality_constraint(mp::AbstractManoptProblem, p, j)
    return get_equality_constraint(get_manifold(mp), get_objective(mp), p, j)
end
@doc raw"""
    get_equality_constraint(M::AbstractManifold, co::ConstrainedObjective, p, j)

evaluate the `j`th equality constraint ``(h(p))_j`` or ``h_j(p)``.

!!! note
    For the [`FunctionConstraint`](@ref) representation this still evaluates all constraints.
"""
get_equality_constraint(M::AbstractManifold, co::ConstrainedObjective, p, j)
function get_equality_constraint(
    M::AbstractManifold, co::ConstrainedObjective{T,FunctionConstraint}, p, j
) where {T<:AbstractEvaluationType}
    return co.h(M, p)[j]
end
function get_equality_constraint(
    M::AbstractManifold, co::ConstrainedObjective{T,VectorConstraint}, p, j
) where {T<:AbstractEvaluationType}
    return co.h[j](M, p)
end

function get_inequality_constraints(mp::AbstractManoptProblem, p)
    return get_inequality_constraints(get_manifold(mp), get_objective(mp), p)
end
@doc raw"""
    get_inequality_constraints(M::AbstractManifold, co::ConstrainedObjective, p)

Evaluate all inequality constraints ``g(p)`` or ``\bigl(g_1(p), g_2(p),\ldots,g_m(p)\bigr)``
of the [`ConstrainedObjective`](@ref) ``P`` at ``p``.
"""
get_inequality_constraints(M::AbstractManifold, co::ConstrainedObjective, p)

function get_inequality_constraints(
    M::AbstractManifold, co::ConstrainedObjective{T,FunctionConstraint}, p
) where {T<:AbstractEvaluationType}
    return co.g(M, p)
end
function get_inequality_constraints(
    M::AbstractManifold, co::ConstrainedObjective{T,VectorConstraint}, p
) where {T<:AbstractEvaluationType}
    return [gi(M, p) for gi in co.g]
end

function get_inequality_constraint(mp::AbstractManoptProblem, p, i)
    return get_inequality_constraint(get_manifold(mp), get_objective(mp), p, i)
end
@doc raw"""
    get_inequality_constraint(M::AbstractManifold, co::ConstrainedObjective, p, i)

evaluate one equality constraint ``(g(p))_i`` or ``g_i(p)``.

!!! note
    For the [`FunctionConstraint`](@ref) representation this still evaluates all constraints.
"""
get_inequality_constraint(M::AbstractManifold, co::ConstrainedObjective, p, i)
function get_inequality_constraint(
    M::AbstractManifold, co::ConstrainedObjective{T,FunctionConstraint}, p, i
) where {T<:AbstractEvaluationType}
    return co.g(M, p)[i]
end
function get_inequality_constraint(
    M::AbstractManifold, co::ConstrainedObjective{T,VectorConstraint}, p, i
) where {T<:AbstractEvaluationType}
    return co.g[i](M, p)
end

function get_grad_equality_constraint(mp::AbstractManoptProblem, p, j)
    return get_grad_equality_constraint(get_manifold(mp), get_objective(mp), p, j)
end
@doc raw"""
    get_grad_equality_constraint(M::AbstractManifold, co::ConstrainedObjective, p, j)

evaluate the gradient of the `j` th equality constraint ``(\operatorname{grad} h(p))_j`` or ``\operatorname{grad} h_j(x)``.

!!! note
    For the [`FunctionConstraint`](@ref) variant of the problem, this function still evaluates the full gradient.
    For the [`InplaceEvaluation`](@ref) and [`FunctionConstraint`](@ref) of the problem, this function currently also calls [`get_equality_constraints`](@ref),
    since this is the only way to determine the number of cconstraints. It also allocates a full tangent vector.
"""
get_grad_equality_constraint(M::AbstractManifold, co::ConstrainedObjective, p, j)
function get_grad_equality_constraint(
    M::AbstractManifold,
    co::ConstrainedObjective{AllocatingEvaluation,FunctionConstraint},
    p,
    j,
)
    return co.grad_h!!(M, p)[j]
end
function get_grad_equality_constraint(
    M::AbstractManifold,
    co::ConstrainedObjective{AllocatingEvaluation,VectorConstraint},
    p,
    j,
)
    return co.grad_h!![j](M, p)
end
function get_grad_equality_constraint(
    M::AbstractManifold,
    co::ConstrainedObjective{InplaceEvaluation,FunctionConstraint},
    p,
    j,
)
    X = [zero_vector(M, p) for _ in 1:length(co.h(M, p))]
    co.grad_h!!(M, X, p)
    return X[j]
end
function get_grad_equality_constraint(
    M::AbstractManifold, co::ConstrainedObjective{InplaceEvaluation,VectorConstraint}, p, j
)
    X = zero_vector(M, p)
    co.grad_h!![j](M, X, p)
    return X
end

function get_grad_equality_constraint!(mp::AbstractManoptProblem, X, p, j)
    return get_grad_equality_constraint!(get_manifold(mp), X, get_objective(mp), p, j)
end
@doc raw"""
    get_grad_equality_constraint!(M::AbstractManifold, X, co::ConstrainedObjective, p, j)

Evaluate the gradient of the `j`th equality constraint ``(\operatorname{grad} h(x))_j`` or ``\operatorname{grad} h_j(x)`` in place of ``X``

!!! note
    For the [`FunctionConstraint`](@ref) variant of the problem, this function still evaluates the full gradient.
    For the [`InplaceEvaluation`](@ref) of the [`FunctionConstraint`](@ref) of the problem, this function currently also calls [`get_inequality_constraints`](@ref),
    since this is the only way to determine the number of cconstraints and allocates a full vector of tangent vectors
"""
get_grad_equality_constraint!(M::AbstractManifold, X, co::ConstrainedObjective, p, j)
function get_grad_equality_constraint!(
    M::AbstractManifold,
    X,
    co::ConstrainedObjective{AllocatingEvaluation,FunctionConstraint},
    p,
    j,
)
    copyto!(M, X, p, co.grad_h!!(M, p)[j])
    return X
end
function get_grad_equality_constraint!(
    M::AbstractManifold,
    X,
    co::ConstrainedObjective{AllocatingEvaluation,VectorConstraint},
    p,
    j,
)
    copyto!(M, X, co.grad_h!![j](M, p))
    return X
end
function get_grad_equality_constraint!(
    M::AbstractManifold,
    X,
    co::ConstrainedObjective{InplaceEvaluation,FunctionConstraint},
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
    co::ConstrainedObjective{InplaceEvaluation,VectorConstraint},
    p,
    j,
)
    co.grad_h!![j](M, X, p)
    return X
end

function get_grad_equality_constraints(mp::AbstractManoptProblem, p)
    return get_grad_equality_constraints(get_manifold(mp), get_objective(mp), p)
end
@doc raw"""
    get_grad_equality_constraints(M::AbstractManifold, co::ConstrainedObjective, p)

eevaluate all gradients of the equality constraints ``\operatorname{grad} h(x)`` or ``\bigl(\operatorname{grad} h_1(x), \operatorname{grad} h_2(x),\ldots, \operatorname{grad}h_n(x)\bigr)``
of the [`ConstrainedObjective`](@ref) `P` at `p`.

!!! note
   for the [`InplaceEvaluation`](@ref) and [`FunctionConstraint`](@ref) variant of the problem,
   this function currently also calls [`get_equality_constraints`](@ref),
   since this is the only way to determine the number of cconstraints.
"""
get_grad_equality_constraints(M::AbstractManifold, co::ConstrainedObjective, p)
function get_grad_equality_constraints(
    M::AbstractManifold,
    co::ConstrainedObjective{AllocatingEvaluation,FunctionConstraint},
    p,
)
    return co.grad_h!!(M, p)
end
function get_grad_equality_constraints(
    M::AbstractManifold, co::ConstrainedObjective{AllocatingEvaluation,VectorConstraint}, p
)
    return [grad_hi(M, p) for grad_hi in co.grad_h!!]
end
function get_grad_equality_constraints(
    M::AbstractManifold, co::ConstrainedObjective{InplaceEvaluation,FunctionConstraint}, p
)
    X = [zero_vector(M, p) for _ in 1:length(co.h(M, p))]
    co.grad_h!!(M, X, p)
    return X
end
function get_grad_equality_constraints(
    M::AbstractManifold, co::ConstrainedObjective{InplaceEvaluation,VectorConstraint}, p
)
    X = [zero_vector(M, p) for _ in 1:length(co.h)]
    [grad_hi(M, Xj, p) for (Xj, grad_hi) in zip(X, co.grad_h!!)]
    return X
end

function get_grad_equality_constraints!(mp::AbstractManoptProblem, X, p)
    return get_grad_equality_constraints!(get_manifold(mp), X, get_objective(mp), p)
end
@doc raw"""
    get_grad_equality_constraints!(M::AbstractManifold, X, co::ConstrainedObjective, p)

evaluate all gradients of the equality constraints ``\operatorname{grad} h(p)`` or ``\bigl(\operatorname{grad} h_1(p), \operatorname{grad} h_2(p),\ldots,\operatorname{grad} h_n(p)\bigr)``
of the [`ConstrainedObjective`](@ref) ``P`` at ``p`` in place of `X``, which is a vector of ``n`` tangent vectors.
"""
function get_grad_equality_constraints!(
    M::AbstractManifold,
    X,
    co::ConstrainedObjective{AllocatingEvaluation,FunctionConstraint},
    p,
)
    copyto!.(Ref(M), X, Ref(p), co.grad_h!!(M, p))
    return X
end
function get_grad_equality_constraints!(
    M::AbstractManifold,
    X,
    co::ConstrainedObjective{AllocatingEvaluation,VectorConstraint},
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
    co::ConstrainedObjective{InplaceEvaluation,FunctionConstraint},
    p,
)
    co.grad_h!!(M, X, p)
    return X
end
function get_grad_equality_constraints!(
    M::AbstractManifold, X, co::ConstrainedObjective{InplaceEvaluation,VectorConstraint}, p
)
    for (Xj, grad_hj) in zip(X, co.grad_h!!)
        grad_hj(M, Xj, p)
    end
    return X
end

function get_grad_inequality_constraint(mp::AbstractManoptProblem, p, i)
    return get_grad_inequality_constraint(get_manifold(mp), get_objective(mp), p, i)
end
@doc raw"""
    get_grad_inequality_constraint(M::AbstractManifold, co::ConstrainedObjective, p, i)

Evaluate the gradient of the `i` th inequality constraints ``(\operatorname{grad} g(x))_i`` or ``\operatorname{grad} g_i(x)``.

!!! note
    For the [`FunctionConstraint`](@ref) variant of the problem, this function still evaluates the full gradient.
    For the [`InplaceEvaluation`](@ref) and [`FunctionConstraint`](@ref) of the problem, this function currently also calls [`get_inequality_constraints`](@ref),
    since this is the only way to determine the number of cconstraints.
"""
get_grad_inequality_constraint(M::AbstractManifold, co::ConstrainedObjective, p, i)
function get_grad_inequality_constraint(
    M::AbstractManifold,
    co::ConstrainedObjective{AllocatingEvaluation,FunctionConstraint},
    p,
    i,
)
    return co.grad_g!!(M, p)[i]
end
function get_grad_inequality_constraint(
    M::AbstractManifold,
    co::ConstrainedObjective{AllocatingEvaluation,VectorConstraint},
    p,
    i,
)
    return co.grad_g!![i](M, p)
end
function get_grad_inequality_constraint(
    M::AbstractManifold,
    co::ConstrainedObjective{InplaceEvaluation,FunctionConstraint},
    p,
    i,
)
    X = [zero_vector(M, p) for _ in 1:length(co.g(M, p))]
    co.grad_g!!(M, X, p)
    return X[i]
end
function get_grad_inequality_constraint(
    M::AbstractManifold, co::ConstrainedObjective{InplaceEvaluation,VectorConstraint}, p, i
)
    X = zero_vector(M, p)
    co.grad_g!![i](M, X, p)
    return X
end

function get_grad_inequality_constraint!(mp::AbstractManoptProblem, X, p, i)
    return get_grad_inequality_constraint!(get_manifold(mp), X, get_objective(mp), p, i)
end
@doc raw"""
    get_grad_inequality_constraint!(P, X, p, i)

Evaluate the gradient of the `i`th inequality constraints ``(\operatorname{grad} g(x))_i`` or ``\operatorname{grad} g_i(x)``
of the [`ConstrainedObjective`](@ref) `P` in place of ``X``

!!! note
    For the [`FunctionConstraint`](@ref) variant of the problem, this function still evaluates the full gradient.
    For the [`InplaceEvaluation`](@ref) and [`FunctionConstraint`](@ref) of the problem, this function currently also calls [`get_inequality_constraints`](@ref),
  since this is the only way to determine the number of cconstraints.
evaluate all gradients of the inequality constraints ``\operatorname{grad} h(x)`` or ``\bigl(g_1(x), g_2(x),\ldots,g_m(x)\bigr)``
of the [`ConstrainedObjective`](@ref) ``p`` at ``x`` in place of `X``, which is a vector of ``m`` tangent vectors .
"""
function get_grad_inequality_constraint!(
    M::AbstractManifold,
    X,
    co::ConstrainedObjective{AllocatingEvaluation,FunctionConstraint},
    p,
    i,
)
    copyto!(M, X, p, co.grad_g!!(M, p)[i])
    return X
end
function get_grad_inequality_constraint!(
    M::AbstractManifold,
    X,
    co::ConstrainedObjective{AllocatingEvaluation,VectorConstraint},
    p,
    i,
)
    copyto!(M, X, co.grad_g!![i](M, p))
    return X
end
function get_grad_inequality_constraint!(
    M::AbstractManifold,
    X,
    co::ConstrainedObjective{InplaceEvaluation,FunctionConstraint},
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
    co::ConstrainedObjective{InplaceEvaluation,VectorConstraint},
    p,
    i,
)
    co.grad_g!![i](M, X, p)
    return X
end

function get_grad_inequality_constraints(mp::AbstractManoptProblem, p)
    return get_grad_inequality_constraints(get_manifold(mp), get_objective(mp), p)
end
@doc raw"""
    get_grad_inequality_constraints(M::AbstractManifold, co::ConstrainedObjective, p)

evaluate all gradients of the inequality constraints ``\operatorname{grad} g(p)`` or ``\bigl(\operatorname{grad} g_1(p), \operatorname{grad} g_2(p),…,\operatorname{grad} g_m(p)\bigr)``
of the [`ConstrainedObjective`](@ref) ``P`` at ``p``.

!!! note
   for the [`InplaceEvaluation`](@ref) and [`FunctionConstraint`](@ref) variant of the problem,
   this function currently also calls [`get_equality_constraints`](@ref),
   since this is the only way to determine the number of cconstraints.
"""
get_grad_inequality_constraints(M::AbstractManifold, co::ConstrainedObjective, x)
function get_grad_inequality_constraints(
    M::AbstractManifold,
    co::ConstrainedObjective{AllocatingEvaluation,FunctionConstraint},
    p,
)
    return co.grad_g!!(M, p)
end
function get_grad_inequality_constraints(
    M::AbstractManifold, co::ConstrainedObjective{AllocatingEvaluation,VectorConstraint}, p
)
    return [grad_gi(M, p) for grad_gi in co.grad_g!!]
end
function get_grad_inequality_constraints(
    M::AbstractManifold, co::ConstrainedObjective{InplaceEvaluation,FunctionConstraint}, p
)
    X = [zero_vector(M, p) for _ in 1:length(co.g(M, p))]
    co.grad_g!!(M, X, p)
    return X
end
function get_grad_inequality_constraints(
    M::AbstractManifold, co::ConstrainedObjective{InplaceEvaluation,VectorConstraint}, p
)
    X = [zero_vector(M, p) for _ in 1:length(co.g)]
    [grad_gi(M, Xi, p) for (Xi, grad_gi) in zip(X, co.grad_g!!)]
    return X
end

function get_grad_inequality_constraints!(mp::AbstractManoptProblem, X, p)
    return get_grad_inequality_constraints!(get_manifold(mp), X, get_objective(mp), p)
end
@doc raw"""
    get_grad_inequality_constraints!(M::AbstractManifold, X, co::ConstrainedObjective, p)

evaluate all gradients of the inequality constraints ``\operatorname{grad} g(x)`` or ``\bigl(\operatorname{grad} g_1(x), \operatorname{grad} g_2(x),\ldots,\operatorname{grad} g_m(x)\bigr)``
of the [`ConstrainedObjective`](@ref) `P` at `p` in place of `X`, which is a vector of ``m`` tangent vectors.
"""
function get_grad_inequality_constraints!(
    M::AbstractManifold,
    X,
    co::ConstrainedObjective{AllocatingEvaluation,FunctionConstraint},
    p,
)
    copyto!.(Ref(M), X, Ref(p), co.grad_g!!(M, p))
    return X
end
function get_grad_inequality_constraints!(
    M::AbstractManifold,
    X,
    co::ConstrainedObjective{AllocatingEvaluation,VectorConstraint},
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
    co::ConstrainedObjective{InplaceEvaluation,FunctionConstraint},
    p,
)
    co.grad_g!!(M, X, p)
    return X
end
function get_grad_inequality_constraints!(
    M::AbstractManifold, X, co::ConstrainedObjective{InplaceEvaluation,VectorConstraint}, p
)
    for (Xi, grad_gi!) in zip(X, co.grad_g!!)
        grad_gi!(M, Xi, p)
    end
    return X
end

function Base.show(io::IO, ::ConstrainedObjective{E,V}) where {E<:AbstractEvaluationType,V}
    return print(io, "ConstrainedObjective{$E,$V}.")
end
