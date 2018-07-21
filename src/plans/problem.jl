#
# Define a global problem and ist constructors
#
# ---
export gradF,getProximalMap, getProximalMaps
export Problem, GradientProblem, ProximalProblem #, HessianProblem

"""
    Problem
Specify properties (values) and related functions for computing
a certain optimization problem.
"""
abstract type Problem end
#
# 1) Functions / Fallbacks
#
costF(p::Pr,x::P) where {Pr <: Problem, P <: MPoint} =
    throw(Exception("no costFunction found in $(typeof(p)) to evaluate for a $(typeof(x))."))
gradF(p::Pr,x::P) where {Pr <: Problem, P <: MPoint} =
    throw(Exception("no gradient found in $(typeof(p)) to evaluate for a $(typeof(x))."))
HessF(p::Pr,x::P,η::T) where {Pr <: Problem, P <: MPoint, T <: TVector} =
    throw(Exception("no Hessian found in $(typeof(p)) to evaluate for a $(typeof(x)) with tangent vector $(typeof(η))."))
proxesF(p::Pr,λ,x::P) where {Pr <: Problem, P <: MPoint} =
    throw(Exception("no proximal maps found in $(typeof(p)) to evaluate for $(typeof(x)) with $(typeof(λ))."))
proxF(p::Pr,λ,x::P,i) where {Pr <: Problem, P <: MPoint} =
    throw(Exception("no $(i)th proximal map found in $(typeof(p)) to evaluate for $(typeof(x)) with $(typeof(λ))."))

doc"""
    GradientProblem <: Problem
specify a problem for gradient based algorithms.

*See also*: [`steepestDescent`](@ref), [`conjugateGradientDescent`](@ref),
[`GradientDescentOptions`](@ref), [`ConjugateGradientOptions`](@ref)

# Fields
* `M`            : a manifold $\mathcal M$
* `costFunction` : a function $F\colon\mathcal M\to\mathbb R$ to minimize
* `gradient`     : the gradient $\nabla F\colon\mathcal M
  \to \mathcal T\mathcal M$ of the cost function $F$

# """
mutable struct GradientProblem{mT <: Manifold} <: Problem
  M::mT
  costFunction::Function
  gradient::Function
end
# Access functions for Gradient problem.
# ---
"""
    gradF(p,x)

evaluate the gradient of a problem at x, where x is either a MPoint
or an array of MPoints
"""
function gradF(p::P,x::MP) where {P <: GradientProblem{M} where M <: Manifold, MP <: MPoint}
  return p.gradient(x)
end

doc"""
    ProximalProblem <: Problem
specify a problem for solvers based on the evaluation of proximal map(s).

# Fields
* `M`            : a manifold $\mathcal M$
* `costFunction` : a function $F\colon\mathcal M\to\mathbb R$ to minimize
* `proximalMaps` : proximal maps $\operatorname{prox}_{\lambda\varphi}\colon\mathcal M\to\mathcal M$
  as functions (λ,x) -> y, i.e. the prox parameter λ also belongs to the signature of the proximal map.
# """
mutable struct ProximalProblem{mT <: Manifold} <: Problem
  M::mT
  costFunction::Function
  proximalMaps::Array{Function,N} where N
end
# Access Functions for proxes.
#
doc"""
    getProximalMaps(p,λ,x)
evaluate all proximal maps of `ProximalProblem p` at the point `x` of `p.M` and
some `λ`$>0$ which might be given as a vector the same length as the number of
proximal maps.
"""
proxesF(p::P,λ,x::MP) where {P <: ProximalProblem{M} where M <: Manifold, MP<:MPoint} =
    p.proximalMaps.(λ,x);
doc"""
    getProximalMap(p,λ,x,i)
evaluate the `i`th proximal map of `ProximalProblem p` at the point `x` of `p.M` with parameter `λ`$>0$.
"""
function proxF(p::P,λ,x::MP,i) where {P <: ProximalProblem{M} where M <: Manifold, MP<:MPoint}
    if i>len(p.proximalMaps)
        ErrorException("the $(i)th entry does not exists, only $(len(p.proximalMaps)) available.")
    end
    return p.proximalMaps[i].(λ,x);
end
