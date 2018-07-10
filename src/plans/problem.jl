#
# Define a global problem and ist constructors
#
# ---
export getGradient,getProximalMap, getProximalMaps
export Problem, GradientProblem, ProximalProblem #, HessianProblem

"""
    Problem
Specify properties (values) and related functions for computing
a certain optimization problem.
"""
abstract type Problem end

doc"""
    GradientProblem <: Problem
specify a problem for gradient basd algorithms.

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
    getGradient(p,x)

evaluate the gradient of a problem at x, where x is either a MPoint
or an array of MPoints
"""
function getGradient{P <: GradientProblem{M} where M <: Manifold, MP <: MPoint}(p::P,x::MP)
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
getProximalMaps{P <: ProximalProblem{M} where M <: Manifold, MP<:MPoint}(p::P,λ,x::MP) = p.proximalMaps.(λ,x);
doc"""
    getProximalMap(p,λ,x,i)
evaluate the `i`th proximal map of `ProximalProblem p` at the point `x` of `p.M` with parameter `λ`$>0$.
"""
function getProximalMap{P <: ProximalProblem{M} where M <: Manifold, MP<:MPoint}(p::P,λ,x::MP,i)
    if i>len(p.proximalMaps)
        ErrorException("the $(i)th entry does not exists, only $(len(p.proximalMaps)) available.")
    end
    return p.proximalMaps[i].(λ,x);
end
