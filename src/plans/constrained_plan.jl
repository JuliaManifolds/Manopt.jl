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
\text{subject to } &G_i(x)<=0 ∀ i= 1, …, m,\\
\quad &H_j(x)=0 ∀ j=1,…,p.
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
struct ConstrainedProblem{T, FunctionConstraint, MT<:Manifold, TCost, TG, TH} <: Problem{T}    # G(p) ∈ R^n, H(p) ∈ R^m
    M::MT
    cost::TCost
    G::TG
    H::TH
    gradF::
    gradG::
    gradH::
end
struct ConstrainedProblem{T, VectorConstraint, MT<:Manifold, TCost, TG, TH} <: Problem{T}      #g_i(p), i=1,...,n, h_j(p), j=1,...,m
    M::MT
    cost::TCost
    G::AbstractVector{<:TG}
    H::AbstractVector{<:TH}
    gradF::
    gradG::AbstractVector{<:}
    gradH::AbstractVector{<:}
end

function get_constraints(p::ConstrainedProblem, x)
    return [get_inequality_constraints(p,x), get_equality_constraints(p,x)]
end

function get_inequality_constraints(p::ConstrainedProblem{T, FunctionConstraint}, x) where {T}
    return p.G(p.M, x)
end

function get_inequality_constraints(p::ConstrainedProblem{T, VectorConstraint}, x) where {T}
    return [ gi(p.M, x) for gi ∈ p.G]
end

function step_solver(p::Problem, o::ALMOptiopns) where {T}
    o.V = get_inequality_constraints(p, o.x)
end
