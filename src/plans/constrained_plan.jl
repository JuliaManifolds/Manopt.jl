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
    ConstrainedProblem{T, Manifold} <: AbstractGradientProblem{T}

Describes the constrained problem
```math
\begin{aligned}
 \operatorname*{arg\,min}_{p ∈\mathcal{M}} & f(p)\\
 \text{subject to } &g_i(p)\leq0 \quad \text{ for all } i=1,…,m,\\
 \quad &h_j(p)=0 \quad \text{ for all } j=1,…,n.
\end{aligned}
```

It consists of
* an `AbstractManifold M`
* an cost function ``f(p)``
* the gradient of ``f``, ``\operatorname{grad}f(p)`` (cf. [`AbstractGradientProblem`](@ref))
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

    ConstrainedProblem(M::AbstractManifold, F, gradF, G, gradG, H, gradH;
        evaluation=AllocatingEvaluation()
    )

Where `F, G, H` describe the cost, inequality and equality constraints as described above
and `gradF, gradG, gradH` are the corresponding gradient functions in one of the 4 formats.
If the problem does not have inequality constraints, you can set `G` and `gradG` no `nothing`.
If the problem does not have equality constraints, you can set `H` and `gradH` no `nothing` or leave them out.

    ConstrainedProblem(M::AbstractManifold, F, gradF;
        G=nothing, gradG=nothing, H=nothing, gradH=nothing;
        evaluation=AllocatingEvaluation()
    )

A keyword argument variant of the constructor above, where you can leave out either
`G` and `gradG` _or_ `H` and `gradH` but not both.
"""
struct ConstrainedProblem{T,CT<:ConstraintType,MT<:AbstractManifold,TCost,GF,TG,GG,TH,GH} <:
       AbstractGradientProblem{T}
    M::MT
    cost::TCost
    gradient!!::GF
    G::TG
    gradG!!::GG
    H::TH
    gradH!!::GH
end
#
# Functions
#
function ConstrainedProblem(
    M::MT,
    F::TF,
    gradF::TGF,
    G::Function,
    gradG::Function,
    H::Function,
    gradH::Function;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
) where {MT<:AbstractManifold,TF,TGF}
    return ConstrainedProblem{
        typeof(evaluation),
        FunctionConstraint,
        MT,
        TF,
        TGF,
        typeof(G),
        typeof(gradG),
        typeof(H),
        typeof(gradH),
    }(
        M, F, gradF, G, gradG, H, gradH
    )
end
# Function without inequality constraints
function ConstrainedProblem(
    M::MT,
    F::TF,
    gradF::TGF,
    ::Nothing,
    ::Nothing,
    H::Function,
    gradH::Function;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
) where {MT<:AbstractManifold,TF,TGF}
    lG = (M, p) -> []
    lgradG = evaluation === AllocatingEvaluation() ? (M, p) -> [] : (M, X, p) -> []
    return ConstrainedProblem{
        typeof(evaluation),
        FunctionConstraint,
        MT,
        TF,
        TGF,
        typeof(lG),
        typeof(lgradG),
        typeof(H),
        typeof(gradH),
    }(
        M, F, gradF, lG, lgradG, H, gradH
    )
end
# No equality constraints
function ConstrainedProblem(
    M::MT,
    F::TF,
    gradF::TGF,
    G::Function,
    gradG::Function,
    ::Nothing=nothing,
    ::Nothing=nothing;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
) where {MT<:AbstractManifold,TF,TGF}
    lH = (M, p) -> []
    lgradH = evaluation === AllocatingEvaluation() ? (M, p) -> [] : (M, X, p) -> []
    return ConstrainedProblem{
        typeof(evaluation),
        FunctionConstraint,
        MT,
        TF,
        TGF,
        typeof(G),
        typeof(gradG),
        typeof(lH),
        typeof(lgradH),
    }(
        M, F, gradF, G, gradG, lH, lgradH
    )
end
#
# Vectors
#
function ConstrainedProblem(
    M::MT,
    F::TF,
    gradF::TGF,
    G::AbstractVector{<:Function},
    gradG::AbstractVector{<:Function},
    H::AbstractVector{<:Function},
    gradH::AbstractVector{<:Function};
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
) where {MT<:AbstractManifold,TF,TGF}
    return ConstrainedProblem{
        typeof(evaluation),
        VectorConstraint,
        MT,
        TF,
        TGF,
        typeof(G),
        typeof(gradG),
        typeof(H),
        typeof(gradH),
    }(
        M, F, gradF, G, gradG, H, gradH
    )
end
# equality not provided
function ConstrainedProblem(
    M::MT,
    F::TF,
    gradF::TGF,
    ::Nothing,
    ::Nothing,
    H::AbstractVector{<:Function},
    gradH::AbstractVector{<:Function};
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
) where {MT<:AbstractManifold,TF,TGF}
    lG = Vector{Function}()
    lgradG = Vector{Function}()
    return ConstrainedProblem{
        typeof(evaluation),
        VectorConstraint,
        MT,
        TF,
        TGF,
        typeof(lG),
        typeof(lgradG),
        typeof(H),
        typeof(gradH),
    }(
        M, F, gradF, lG, lgradG, H, gradH
    )
end
# No eqality constraints provided
function ConstrainedProblem(
    M::MT,
    F::TF,
    gradF::TGF,
    G::AbstractVector{<:Function},
    gradG::AbstractVector{<:Function},
    ::Nothing,
    ::Nothing;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
) where {MT<:AbstractManifold,TF,TGF}
    lH = Vector{Function}()
    lgradH = Vector{Function}()
    return ConstrainedProblem{
        typeof(evaluation),
        VectorConstraint,
        MT,
        TF,
        TGF,
        typeof(G),
        typeof(gradG),
        typeof(lH),
        typeof(lgradH),
    }(
        M, F, gradF, G, gradG, lH, lgradH
    )
end
#
# Neither equality nor inequality yields an error
#
function ConstrainedProblem(
    ::MT, ::TF, ::TGF, ::Nothing, ::Nothing, ::Nothing, ::Nothing; kwargs...
) where {MT<:AbstractManifold,TF,TGF}
    return error(
        """
  Neither inequality constraints `G`, `gradG` nor equality constraints `H` `gradH` provided.
  If you have an unconstraint problem, maybe consider using a `GradientProblem` instead.
  """
    )
end
function ConstrainedProblem(
    M::MT,
    F::TF,
    gradF::TGF;
    G=nothing,
    gradG=nothing,
    H=nothing,
    gradH=nothing,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
) where {MT<:AbstractManifold,TF,TGF}
    return ConstrainedProblem(M, F, gradF, G, gradG, H, gradH; evaluation=evaluation)
end
"""
    get_constraints(P::ConstrainedProblem, p)

Return the vector ``(g_1(p),...g_m(p),h_1(p),...,h_n(p))`` from the [`ConstrainedProblem`](@ref) `P`
containing the values of all constraints at `p`.
"""
function get_constraints(P::ConstrainedProblem, p)
    return [get_inequality_constraints(P, p), get_equality_constraints(P, p)]
end

@doc raw"""
    get_equality_constraints(P::ConstrainedProblem, p)

evaluate all equality constraints ``h(p)`` of ``\bigl(h_1(p), h_2(p),\ldots,h_p(p)\bigr)``
of the [`ConstrainedProblem`](@ref) ``P`` at ``p``.
"""
get_equality_constraints(P::ConstrainedProblem, p)
function get_equality_constraints(P::ConstrainedProblem{T,FunctionConstraint}, p) where {T}
    return P.H(P.M, p)
end
function get_equality_constraints(P::ConstrainedProblem{T,VectorConstraint}, p) where {T}
    return [hj(P.M, p) for hj in P.H]
end

@doc raw"""
    get_equality_constraint(P::ConstrainedProblem, p, j)

evaluate the `j`th equality constraint ``(h(p))_j`` or ``h_j(p)``.

!!! note
    For the [`FunctionConstraint`](@ref) representation this still evaluates all constraints.
"""
get_equality_constraint(P::ConstrainedProblem, p, j)
function get_equality_constraint(
    P::ConstrainedProblem{T,FunctionConstraint}, p, j
) where {T}
    return P.H(P.M, p)[j]
end
function get_equality_constraint(P::ConstrainedProblem{T,VectorConstraint}, p, j) where {T}
    return P.H[j](P.M, p)
end

@doc raw"""
    get_inequality_constraints(P::ConstrainedProblem, p)

Evaluate all inequality constraints ``g(p)`` or ``\bigl(g_1(p), g_2(p),\ldots,g_m(p)\bigr)``
of the [`ConstrainedProblem`](@ref) ``P`` at ``p``.
"""
get_inequality_constraints(P::ConstrainedProblem, p)

function get_inequality_constraints(
    P::ConstrainedProblem{T,FunctionConstraint}, p
) where {T}
    return P.G(P.M, p)
end
function get_inequality_constraints(P::ConstrainedProblem{T,VectorConstraint}, p) where {T}
    return [gi(P.M, p) for gi in P.G]
end

@doc raw"""
    get_inequality_constraint(P::ConstrainedProblem, p, i)

evaluate one equality constraint ``(g(p))_i`` or ``g_i(p)``.

!!! note
    For the [`FunctionConstraint`](@ref) representation this still evaluates all constraints.
"""
get_inequality_constraint(P::ConstrainedProblem, p, i)
function get_inequality_constraint(
    P::ConstrainedProblem{T,FunctionConstraint}, p, i
) where {T}
    return P.G(P.M, p)[i]
end
function get_inequality_constraint(
    P::ConstrainedProblem{T,VectorConstraint}, p, i
) where {T}
    return P.G[i](P.M, p)
end

@doc raw"""
    get_grad_equality_constraint(P, p, j)

evaluate the gradient of the `j` th equality constraint ``(\operatorname{grad} h(p))_j`` or ``\operatorname{grad} h_j(x)``.

!!! note
    For the [`FunctionConstraint`](@ref) variant of the problem, this function still evaluates the full gradient.
    For the [`MutatingEvaluation`](@ref) and [`FunctionConstraint`](@ref) of the problem, this function currently also calls [`get_equality_constraints`](@ref),
    since this is the only way to determine the number of cconstraints. It also allocates a full tangent vector.
"""
get_grad_equality_constraint(P::ConstrainedProblem, p, j)
function get_grad_equality_constraint(
    P::ConstrainedProblem{AllocatingEvaluation,FunctionConstraint}, p, j
)
    return P.gradH!!(P.M, p)[j]
end
function get_grad_equality_constraint(
    P::ConstrainedProblem{AllocatingEvaluation,VectorConstraint}, p, j
)
    return P.gradH!![j](P.M, p)
end
function get_grad_equality_constraint(
    P::ConstrainedProblem{MutatingEvaluation,FunctionConstraint}, p, j
)
    X = [zero_vector(P.M, p) for _ in 1:length(P.H(P.M, p))]
    P.gradH!!(P.M, X, p)
    return X[j]
end
function get_grad_equality_constraint(
    P::ConstrainedProblem{MutatingEvaluation,VectorConstraint}, p, j
)
    X = zero_vector(P.M, p)
    P.gradH!![j](P.M, X, p)
    return X
end

@doc raw"""
    get_grad_equality_constraint!(P, X, p, j)

Evaluate the gradient of the `j`th equality constraint ``(\operatorname{grad} h(x))_j`` or ``\operatorname{grad} h_j(x)`` in place of ``X``

!!! note
    For the [`FunctionConstraint`](@ref) variant of the problem, this function still evaluates the full gradient.
    For the [`MutatingEvaluation`](@ref) of the [`FunctionConstraint`](@ref) of the problem, this function currently also calls [`get_inequality_constraints`](@ref),
    since this is the only way to determine the number of cconstraints and allocates a full vector of tangent vectors
"""
get_grad_equality_constraint!(P::ConstrainedProblem, p, j)
function get_grad_equality_constraint!(
    P::ConstrainedProblem{AllocatingEvaluation,FunctionConstraint}, X, p, j
)
    copyto!(P.M, X, p, P.gradH!!(P.M, p)[j])
    return X
end
function get_grad_equality_constraint!(
    P::ConstrainedProblem{AllocatingEvaluation,VectorConstraint}, X, p, j
)
    copyto!(P.M, X, P.gradH!![j](P.M, p))
    return X
end
function get_grad_equality_constraint!(
    P::ConstrainedProblem{MutatingEvaluation,FunctionConstraint}, X, p, j
)
    Y = [zero_vector(P.M, p) for _ in 1:length(P.H(P.M, p))]
    P.gradH!!(P.M, Y, p)
    copyto!(P.M, X, p, Y[j])
    return X
end
function get_grad_equality_constraint!(
    P::ConstrainedProblem{MutatingEvaluation,VectorConstraint}, X, p, j
)
    P.gradH!![j](P.M, X, p)
    return X
end

### -----
@doc raw"""
    get_grad_equality_constraints(P, p)

eevaluate all gradients of the equality constraints ``\operatorname{grad} h(x)`` or ``\bigl(\operatorname{grad} h_1(x), \operatorname{grad} h_2(x),\ldots, \operatorname{grad}h_n(x)\bigr)``
of the [`ConstrainedProblem`](@ref) `P` at `p`.

!!! note
   for the [`MutatingEvaluation`](@ref) and [`FunctionConstraint`](@ref) variant of the problem,
   this function currently also calls [`get_equality_constraints`](@ref),
   since this is the only way to determine the number of cconstraints.
"""
get_grad_equality_constraints(P::ConstrainedProblem, p)
function get_grad_equality_constraints(
    P::ConstrainedProblem{AllocatingEvaluation,FunctionConstraint}, p
)
    return P.gradH!!(P.M, p)
end
function get_grad_equality_constraints(
    P::ConstrainedProblem{AllocatingEvaluation,VectorConstraint}, p
)
    return [grad_hi(P.M, p) for grad_hi in P.gradH!!]
end
function get_grad_equality_constraints(
    P::ConstrainedProblem{MutatingEvaluation,FunctionConstraint}, p
)
    X = [zero_vector(P.M, p) for _ in 1:length(P.H(P.M, p))]
    P.gradH!!(P.M, X, p)
    return X
end
function get_grad_equality_constraints(
    P::ConstrainedProblem{MutatingEvaluation,VectorConstraint}, p
)
    X = [zero_vector(P.M, p) for _ in 1:length(P.H)]
    [grad_hi(P.M, Xj, p) for (Xj, grad_hi) in zip(X, P.gradH!!)]
    return X
end

@doc raw"""
    get_grad_equality_constraints!(P, X, p)

evaluate all gradients of the equality constraints ``\operatorname{grad} h(p)`` or ``\bigl(\operatorname{grad} h_1(p), \operatorname{grad} h_2(p),\ldots,\operatorname{grad} h_n(p)\bigr)``
of the [`ConstrainedProblem`](@ref) ``P`` at ``p`` in place of `X``, which is a vector of ``n`` tangent vectors.
"""
function get_grad_equality_constraints!(
    P::ConstrainedProblem{AllocatingEvaluation,FunctionConstraint}, X, p
)
    copyto!.(Ref(P.M), X, Ref(p), P.gradH!!(P.M, p))
    return X
end
function get_grad_equality_constraints!(
    P::ConstrainedProblem{AllocatingEvaluation,VectorConstraint}, X, p
)
    for (Xj, grad_hj) in zip(X, P.gradH!!)
        copyto!(P.M, Xj, grad_hj(P.M, p))
    end
    return X
end
function get_grad_equality_constraints!(
    P::ConstrainedProblem{MutatingEvaluation,FunctionConstraint}, X, p
)
    P.gradH!!(P.M, X, p)
    return X
end
function get_grad_equality_constraints!(
    P::ConstrainedProblem{MutatingEvaluation,VectorConstraint}, X, p
)
    for (Xj, grad_hj) in zip(X, P.gradH!!)
        grad_hj(P.M, Xj, p)
    end
    return X
end

@doc raw"""
    get_grad_inequality_constraint(P, p, i)

Evaluate the gradient of the `i` th inequality constraints ``(\operatorname{grad} g(x))_i`` or ``\operatorname{grad} g_i(x)``.

!!! note
    For the [`FunctionConstraint`](@ref) variant of the problem, this function still evaluates the full gradient.
    For the [`MutatingEvaluation`](@ref) and [`FunctionConstraint`](@ref) of the problem, this function currently also calls [`get_inequality_constraints`](@ref),
    since this is the only way to determine the number of cconstraints.
"""
get_grad_inequality_constraint(P::ConstrainedProblem, p, i)
function get_grad_inequality_constraint(
    P::ConstrainedProblem{AllocatingEvaluation,FunctionConstraint}, p, i
)
    return P.gradG!!(P.M, p)[i]
end
function get_grad_inequality_constraint(
    P::ConstrainedProblem{AllocatingEvaluation,VectorConstraint}, p, i
)
    return P.gradG!![i](P.M, p)
end
function get_grad_inequality_constraint(
    P::ConstrainedProblem{MutatingEvaluation,FunctionConstraint}, p, i
)
    X = [zero_vector(P.M, p) for _ in 1:length(P.G(P.M, p))]
    P.gradG!!(P.M, X, p)
    return X[i]
end
function get_grad_inequality_constraint(
    P::ConstrainedProblem{MutatingEvaluation,VectorConstraint}, p, i
)
    X = zero_vector(P.M, p)
    P.gradG!![i](P.M, X, p)
    return X
end

@doc raw"""
    get_grad_inequality_constraint!(P, X, p, i)

Evaluate the gradient of the `i`th inequality constraints ``(\operatorname{grad} g(x))_i`` or ``\operatorname{grad} g_i(x)``
of the [`ConstrainedProblem`](@ref) `P` in place of ``X``

!!! note
    For the [`FunctionConstraint`](@ref) variant of the problem, this function still evaluates the full gradient.
    For the [`MutatingEvaluation`](@ref) and [`FunctionConstraint`](@ref) of the problem, this function currently also calls [`get_inequality_constraints`](@ref),
  since this is the only way to determine the number of cconstraints.
evaluate all gradients of the inequality constraints ``\operatorname{grad} h(x)`` or ``\bigl(g_1(x), g_2(x),\ldots,g_m(x)\bigr)``
of the [`ConstrainedProblem`](@ref) ``p`` at ``x`` in place of `X``, which is a vector of ``m`` tangent vectors .
"""
function get_grad_inequality_constraint!(
    P::ConstrainedProblem{AllocatingEvaluation,FunctionConstraint}, X, p, i
)
    copyto!(P.M, X, p, P.gradG!!(P.M, p)[i])
    return X
end
function get_grad_inequality_constraint!(
    P::ConstrainedProblem{AllocatingEvaluation,VectorConstraint}, X, p, i
)
    copyto!(P.M, X, P.gradG!![i](P.M, p))
    return X
end
function get_grad_inequality_constraint!(
    P::ConstrainedProblem{MutatingEvaluation,FunctionConstraint}, X, p, i
)
    Y = [zero_vector(P.M, p) for _ in 1:length(P.G(P.M, p))]
    P.gradG!!(P.M, Y, p)
    copyto!(P.M, X, p, Y[i])
    return X
end
function get_grad_inequality_constraint!(
    P::ConstrainedProblem{MutatingEvaluation,VectorConstraint}, X, p, i
)
    P.gradG!![i](P.M, X, p)
    return X
end

@doc raw"""
    get_grad_inequality_constraints(P, p)

evaluate all gradients of the inequality constraints ``\operatorname{grad} g(p)`` or ``\bigl(\operatorname{grad} g_1(p), \operatorname{grad} g_2(p),…,\operatorname{grad} g_m(p)\bigr)``
of the [`ConstrainedProblem`](@ref) ``P`` at ``p``.

!!! note
   for the [`MutatingEvaluation`](@ref) and [`FunctionConstraint`](@ref) variant of the problem,
   this function currently also calls [`get_equality_constraints`](@ref),
   since this is the only way to determine the number of cconstraints.
"""
get_grad_inequality_constraints(p::ConstrainedProblem, x)
function get_grad_inequality_constraints(
    P::ConstrainedProblem{AllocatingEvaluation,FunctionConstraint}, p
)
    return P.gradG!!(P.M, p)
end
function get_grad_inequality_constraints(
    P::ConstrainedProblem{AllocatingEvaluation,VectorConstraint}, p
)
    return [grad_gi(P.M, p) for grad_gi in P.gradG!!]
end
function get_grad_inequality_constraints(
    P::ConstrainedProblem{MutatingEvaluation,FunctionConstraint}, p
)
    X = [zero_vector(P.M, p) for _ in 1:length(P.G(P.M, p))]
    P.gradG!!(P.M, X, p)
    return X
end
function get_grad_inequality_constraints(
    P::ConstrainedProblem{MutatingEvaluation,VectorConstraint}, p
)
    X = [zero_vector(P.M, p) for _ in 1:length(P.G)]
    [grad_gi(P.M, Xi, p) for (Xi, grad_gi) in zip(X, P.gradG!!)]
    return X
end

@doc raw"""
    get_grad_inequality_constraints!(P, X, p)

evaluate all gradients of the inequality constraints ``\operatorname{grad} g(x)`` or ``\bigl(\operatorname{grad} g_1(x), \operatorname{grad} g_2(x),\ldots,\operatorname{grad} g_m(x)\bigr)``
of the [`ConstrainedProblem`](@ref) `P` at `p` in place of `X`, which is a vector of ``m`` tangent vectors.
"""
function get_grad_inequality_constraints!(
    P::ConstrainedProblem{AllocatingEvaluation,FunctionConstraint}, X, p
)
    copyto!.(Ref(P.M), X, Ref(p), P.gradG!!(P.M, p))
    return X
end
function get_grad_inequality_constraints!(
    P::ConstrainedProblem{AllocatingEvaluation,VectorConstraint}, X, p
)
    for (Xi, grad_gi) in zip(X, P.gradG!!)
        copyto!(P.M, Xi, grad_gi(P.M, p))
    end
    return X
end
function get_grad_inequality_constraints!(
    P::ConstrainedProblem{MutatingEvaluation,FunctionConstraint}, X, p
)
    P.gradG!!(P.M, X, p)
    return X
end
function get_grad_inequality_constraints!(
    P::ConstrainedProblem{MutatingEvaluation,VectorConstraint}, X, p
)
    for (Xi, grad_gi!) in zip(X, P.gradG!!)
        grad_gi!(P.M, Xi, p)
    end
    return X
end

function Base.show(io::IO, ::ConstrainedProblem{E,V}) where {E,V}
    return print(io, "ConstrainedProblem{$E,$V}.")
end
