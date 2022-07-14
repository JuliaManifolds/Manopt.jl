@doc raw"""
    ConstraintType

An abstract type to represent different forms of representing constraints
"""
abstract type ConstraintType end

@doc raw"""
    FunctionConstraint <: ConstraintType

A type to indicate that constraints are implemented one whole functions, e.g. ``g(x) ∈ \mathbb R^m``.
"""
struct FunctionConstraint <: ConstraintType end

@doc raw"""
    VectorConstraint <: ConstraintType

A type to indicate that constraints are implemented a  vector of functions, e.g. ``g_i(x) ∈ \mathbb R, i=1,…m``.
"""
struct VectorConstraint <: ConstraintType end

@doc raw"""
    ConstrainedProblem{T, Manifold} <: AbstractGradientProblem{T}

Describes the constrained problem
```math
 \begin{aligned}
  \min_{x ∈\mathcal{M}} &F(x)\\
  \text{subject to } &G_i(x)\leq0 \quad ∀ i= 1, …, m,\\
  \quad &H_j(x)=0 \quad ∀ j=1,…,p.
 \end{aligned}
```
It consists of
* a `Manifold M`
* an cost function ``F(x)``
* the gradient of ``F(x)``, ``\operatorname{grad}F(x)`` (whose handling is inherited from the [`AbstractGradientProblem`](@ref))
* an (optional) array of inequality constraints ``G(x)``, i.e. a function that returns an array or an array of functions ``G(x) = \{G_i(x)\}_{i=1}^m``
* an (optional) array of equality constraints ``H(x)``, i.e. a function that returns an array or an array of functions ``H(x) = \{H_j\}_{j=1}^p``
* an array of gradients for G(x), i.e. a function that returns an array or an array of functions
``\{\operatorname{grad}G_i\}_{i=1}^m``
* an array of gradients for H(x), i.e. a function that returns an array or an array of functions
``\{\operatorname{grad}H_j\}_{j=1}^p``

# Constructors
    ConstrainedProblem(M::Manifold, cost::Function, G::Function, H::Function, gradF::Function, gradG::Function, gradH::Function;
        evaluation=AllocatingEvaluation()
    )
    ConstrainedProblem(M::Manifold, cost::Function, G::AbstractVector{<:Function}, H::AbstractVector{<:Function}, gradF::Function, gradG::AbstractVector{<:Function}, gradH::AbstractVector{<:Function};
        evaluation=AllocatingEvaluation()
    )

Create a constrained problem with a `cost` function and its gradient, as well as inequality and equality contraints and their gradients either as one
function (returning an array) or a vector of functions.
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
function ConstrainedProblem(
    M::MT, F::TF, gradF::TGF, G::Function, gradG::Function, H::Function, gradH::Function
) where {MT<:AbstractManifold,TF,TGF} # G(p) ∈ R^n, H(p) ∈ R^m
    return ConstrainedProblem{
        AllocatingEvaluation,
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
function ConstrainedProblem(
    M::MT,
    F::TF,
    gradF::TGF,
    G::AbstractVector{<:Function},
    gradG::AbstractVector{<:Function},
    H::AbstractVector{<:Function},
    gradH::AbstractVector{<:Function},
) where {MT<:AbstractManifold,TF,TGF}#g_i(p), i=1,...,n, h_j(p), j=1,...,m
    return ConstrainedProblem{
        AllocatingEvaluation,
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

function get_constraints(p::ConstrainedProblem, x)
    return [get_inequality_constraints(p, x), get_equality_constraints(p, x)]
end

"""
    get_equality_constraints(p::ConstrainedProblem, x)

evaluate all equality constraints ``h(x)`` of ``\bigl(h_1(x), h_2(x),\ldots,h_p(x)\bigr)``
of the [`ConstrainedProblem`](@ref) ``p`` at ``x``.
"""
get_equality_constraints(p::ConstrainedProblem, x)
function get_equality_constraints(p::ConstrainedProblem{T,FunctionConstraint}, x) where {T}
    return p.H(p.M, x)
end
function get_equality_constraints(p::ConstrainedProblem{T,VectorConstraint}, x) where {T}
    return [hj(p.M, x) for hj in p.H]
end

"""
    get_equality_constraint(p::ConstrainedProblem, x, i)

evaluate one equality constraint ``(h(x))_i`` or ``h_i(x)``.

!!! note
  For the [`FunctionConstraint`](@ref) representation this still evaluates all constraints.
"""
get_equality_constraint(p::ConstrainedProblem, x, i)
function get_equality_constraints(
    p::ConstrainedProblem{T,FunctionConstraint}, x, i
) where {T}
    return p.H(p.M, x)[i]
end
function get_equality_constraints(p::ConstrainedProblem{T,VectorConstraint}, x, i) where {T}
    return p.H[i](p.M, x)
end

@doc raw"""
    get_inequality_constraints(p::ConstrainedProblem, x)

evaluate all equality constraints ``g(x)`` or ``\bigl(g_1(x), g_2(x),\ldots,g_m(x)\bigr)``
of the [`ConstrainedProblem`](@ref) ``p`` at ``x``.
"""
get_inequality_constraints(p::ConstrainedProblem, x)

function get_inequality_constraints(
    p::ConstrainedProblem{T,FunctionConstraint}, x
) where {T}
    return p.G(p.M, x)
end
function get_inequality_constraints(p::ConstrainedProblem{T,VectorConstraint}, x) where {T}
    return [gi(p.M, x) for gi in p.G]
end

@doc raw"""
    get_equality_constraint(p::ConstrainedProblem, x, i)

evaluate one equality constraint ``(g(x))_i`` or ``g_i(x)``.

!!! note
  For the [`FunctionConstraint`](@ref) representation this still evaluates all constraints.
"""
get_equality_constraint(p::ConstrainedProblem, x, i)
function get_inequality_constraints(
    p::ConstrainedProblem{T,FunctionConstraint}, x, i
) where {T}
    return p.G(p.M, x)[i]
end
function get_inequality_constraints(
    p::ConstrainedProblem{T,VectorConstraint}, x, i
) where {T}
    return p.G[i](p.M, x)
end

@doc raw"""
    get_grad_inequality_constraints(p, x)

eevaluate all gradients of the inequality constraints ``\operatorname{grad} g(x)`` or ``\bigl(g_1(x), g_2(x),\ldots,g_m(x)\bigr)``
of the [`ConstrainedProblem`](@ref) ``p`` at ``x``.

!!! note
  for the [`MutatingEvaluation`](@ref) and [`FunctionConstraint`](@ref) of the problem, this function currently also calls [`get_inequality_constraints`](@ref),
  since this is the only way to determine the number of cconstraints.

"""
get_grad_inequality_constraints(p::ConstrainedProblem, x)
function get_grad_inequality_constraints(
    p::ConstrainedProblem{AllocatingEvaluation,FunctionConstraint}, x
)
    return p.gradG!!(p.M, x)
end
function get_grad_inequality_constraints(
    p::ConstrainedProblem{AllocatingEvaluation,VectorConstraint}, x
)
    return [grad_gi(p.M, x) for grad_gi in p.gradG!!]
end
function get_grad_inequality_constraints(
    p::ConstrainedProblem{MutatingEvaluation,FunctionConstraint}, x
)
    X = [zero_vector(p.M, x) for i in 1:length(p.G(x))]
    p.gradG!!(p.M, X, x)
    return X
end
function get_grad_inequality_constraints(
    p::ConstrainedProblem{MutatingEvaluation,VectorConstraint}, x
)
    X = [zero_vector(p.M, x) for i in 1:length(p.G(x))]
    [grad_gi(p.M, X[i], x) for grad_gi in p.gradG!!]
    return X
end

@doc raw"""
    get_grad_inequality_constraints!(p, X, x)

evaluate all gradients of the inequality constraints ``\operatorname{grad} h(x)`` or ``\bigl(g_1(x), g_2(x),\ldots,g_m(x)\bigr)``
of the [`ConstrainedProblem`](@ref) ``p`` at ``x`` in place of `X``, which is a vector of ``m`` tangent vectors .
"""
function get_grad_inequality_constraints!(
    p::ConstrainedProblem{AllocatingEvaluation,FunctionConstraint}, X, x
)
    copyto!.(Ref(M), X, Ref(x)X, p.gradG!!(p.M, x))
    return X
end
function get_grad_inequality_constraints!(
    p::ConstrainedProblem{AllocatingEvaluation,VectorConstraint}, X, x
)
    for i in 1:length(p.gradG!!)
        copyto!(M, X[i], p.gradG!![i](p.M, x))
    end
    return X
end
function get_grad_inequality_constraints!(
    p::ConstrainedProblem{MutatingEvaluation,FunctionConstraint}, X, x
)
    p.gradG!!(p.M, X, x)
    return X
end
function get_grad_inequality_constraints!(
    p::ConstrainedProblem{MutatingEvaluation,VectorConstraint}, X, x
)
    for i in 1:length(p.gradG!!)
        p.gradG!![i](p.M, X[i], x)
    end
    return X
end

@dor raw"""
    get_grad_inequality_constraint(p, x, i)

eevaluate the gradient of the `i`th inequality constraints ``(\operatorname{grad} g(x))_i`` or ``\operatorname{grad} g_i(x)``.@

!!! note
    For the [`FunctionConstraint`](@ref) variant of the problem, this function still evaluates the full gradient.
    For the [`MutatingEvaluation`](@ref) and [`FunctionConstraint`](@ref) of the problem, this function currently also calls [`get_inequality_constraints`](@ref),
  since this is the only way to determine the number of cconstraints.

"""
get_grad_inequality_constraint(p::ConstrainedProblem, x, i)
function get_grad_inequality_constraint(
    p::ConstrainedProblem{AllocatingEvaluation,FunctionConstraint}, x, i
)
    return p.gradG!!(p.M, x)[i]
end
function get_grad_inequality_constraint(
    p::ConstrainedProblem{AllocatingEvaluation,VectorConstraint}, x, i
)
    return p.gradG!![i](p.M, x)
end
function get_grad_inequality_constraint(
    p::ConstrainedProblem{MutatingEvaluation,FunctionConstraint}, x, i
)
    X = [zero_vector(p.M, x) for i in 1:length(p.G(x))]
    p.gradG!!(p.M, X, x)
    return X[i]
end
function get_grad_inequality_constraints(
    p::ConstrainedProblem{MutatingEvaluation,VectorConstraint}, x, i
)
    X = zero_vector(p.M, x)
    p.gradG!![i](p.M, X, x)
    return X
end

@dor raw"""
    get_grad_inequality_constraint!(p, X, x, i)

eevaluate the gradient of the `i`th inequality constraints ``(\operatorname{grad} g(x))_i`` or ``\operatorname{grad} g_i(x)`` in place of ``X``

!!! note
    For the [`FunctionConstraint`](@ref) variant of the problem, this function still evaluates the full gradient.
    For the [`MutatingEvaluation`](@ref) and [`FunctionConstraint`](@ref) of the problem, this function currently also calls [`get_inequality_constraints`](@ref),
  since this is the only way to determine the number of cconstraints.
evaluate all gradients of the inequality constraints ``\operatorname{grad} h(x)`` or ``\bigl(g_1(x), g_2(x),\ldots,g_m(x)\bigr)``
of the [`ConstrainedProblem`](@ref) ``p`` at ``x`` in place of `X``, which is a vector of ``m`` tangent vectors .
"""
function get_grad_inequality_constraint!(
    p::ConstrainedProblem{AllocatingEvaluation,FunctionConstraint}, X, x, i
)
    copyto!(M, X, x, X, p.gradG!!(p.M, x)[i])
    return X
end
function get_grad_inequality_constraint!(
    p::ConstrainedProblem{AllocatingEvaluation,VectorConstraint}, X, x, i
)
    copyto!(M, X, p.gradG!![i](p.M, x))
    return X
end
function get_grad_inequality_constraint!(
    p::ConstrainedProblem{MutatingEvaluation,FunctionConstraint}, X, x, i
)
    Y = [zero_vector(p.M, x) for i in 1:length(p.G(x))]
    p.gradG!!(p.M, Y, x)
    copyto!(M, X, x, Y[i])
    return X
end
function get_grad_inequality_constraints!(
    p::ConstrainedProblem{MutatingEvaluation,VectorConstraint}, X, x
)
    p.gradG!![i](p.M, X, x)
    return X
end
