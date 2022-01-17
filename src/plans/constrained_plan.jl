## todo: write a documentation for those
abstract type ConstraintType end
struct FunctionConstraint <: ConstraintType end
struct VectorConstraint <: ConstraintType end

@doc raw"""
    ConstrainedProblem{T, Manifold} <: Problem{T}

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
* an (optional) array of inequality constraints ``G(x)``, i.e. a function that returns an array or an array of functions ``G(x) = \{G_i(x)\}_{i=1}^m``
* an (optional) array of equality constraints ``H(x)``, i.e. a function that returns an array or an array of functions ``H(x) = \{H_j\}_{j=1}^p``
* the gradient of ``F(x)``, ``\operatorname{grad}F(x)``
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
struct ConstrainedProblem{T, CT<:ConstraintType, MT<:AbstractManifold, TCost, GF, TG, GG, TH, GH} <: Problem{T}   
    M::MT
    cost::TCost
    gradF::GF
    G::TG
    gradG::GG
    H::TH
    gradH::GH
end

function ConstrainedProblem(M::MT, F::TF, gradF::TGF, G::Function, gradG::Function, H::Function, gradH::Function) where {MT<:AbstractManifold, TF, TGF} # G(p) ∈ R^n, H(p) ∈ R^m
    return ConstrainedProblem{AllocatingEvaluation, FunctionConstraint, MT, TF, TGF, typeof(G), typeof(gradG), typeof(H), typeof(gradH)}(M, F, gradF, G, gradG, H, gradH)
end 

function ConstrainedProblem(M::MT, F::TF, gradF::TGF, G::AbstractVector{<:Function}, gradG::AbstractVector{<:Function}, H::AbstractVector{<:Function}, gradH::AbstractVector{<:Function}) where {MT<:AbstractManifold, TF, TGF}#g_i(p), i=1,...,n, h_j(p), j=1,...,m
    return ConstrainedProblem{AllocatingEvaluation, VectorConstraint, MT, TF, TGF, typeof(G), typeof(gradG), typeof(H), typeof(gradH)}(M, F, gradF, G, gradG, H, gradH)
end 

function get_constraints(p::ConstrainedProblem, x)
    return [get_inequality_constraints(p,x), get_equality_constraints(p,x)]
end

function get_inequality_constraints(p::ConstrainedProblem{T, FunctionConstraint}, x) where {T}
    return p.G(p.M, x)
end

function get_inequality_constraints(p::ConstrainedProblem{T, VectorConstraint}, x) where {T}
    return [gi(p.M, x) for gi ∈ p.G]
end

function get_equality_constraints(p::ConstrainedProblem{T, FunctionConstraint}, x) where {T}
    return p.H(p.M, x)
end

function get_equality_constraints(p::ConstrainedProblem{T, VectorConstraint}, x) where {T}
    return [hj(p.M, x) for hj ∈ p.H]
end

function get_grad_ineq(p::ConstrainedProblem{T, FunctionConstraint}, x) where {T}
    return p.gradG(p.M, x)
end

function get_grad_ineq(p::ConstrainedProblem{T, VectorConstraint}, x) where {T}
    return [grad_gi(p.M, x) for grad_gi ∈ p.gradG]
end

function get_grad_eq(p::ConstrainedProblem{T, FunctionConstraint}, x) where {T}
    return p.gradH(p.M, x)
end

function get_grad_eq(p::ConstrainedProblem{T, VectorConstraint}, x) where {T}
    return [grad_hj(p.M, x) for grad_hj ∈ p.gradH]
end

function get_gradient(p::ConstrainedProblem{AllocatingEvaluation}, x)
    return p.gradF(p.M, x)
end
function get_gradient(p::ConstrainedProblem{MutatingEvaluation}, x)
    X = zero_vector(p.M, x)
    return p.gradF(p.M, X, x)
end

function get_gradient!(p::ConstrainedProblem{AllocatingEvaluation}, X, x)
    return copyto!(p.M, X, x, p.gradient!!(p.M, x))
end

function get_gradient!(p::ConstrainedProblem{MutatingEvaluation}, X, x)
    return p.gradF!(p.M, X, x)
end

