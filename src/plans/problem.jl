#
# Define a global problem and ist constructors
#
# ---
import Random: randperm
export getGradient, getCost, getHessian, getProximalMap, getProximalMaps
export Problem, GradientProblem, ProximalProblem, HessianProblem

"""
    Problem
Specify properties (values) and related functions for computing
a certain optimization problem.
"""
abstract type Problem end
#
# 1) Functions / Fallbacks
#
getCost(p::Pr,x::P) where {Pr <: Problem, P <: MPoint} =
    throw(Exception("no costFunction found in $(typeof(p)) to evaluate for a $(typeof(x))."))
getGradient(p::Pr,x::P) where {Pr <: Problem, P <: MPoint} =
    throw(Exception("no gradient found in $(typeof(p)) to evaluate for a $(typeof(x))."))
getHessian(p::Pr,x::P,η::T) where {Pr <: Problem, P <: MPoint, T <: TVector} =
    throw(Exception("no Hessian found in $(typeof(p)) to evaluate for a $(typeof(x)) with tangent vector $(typeof(η))."))
getProximalMaps(p::Pr,λ,x::P) where {Pr <: Problem, P <: MPoint} =
    throw(Exception("no proximal maps found in $(typeof(p)) to evaluate for $(typeof(x)) with $(typeof(λ))."))
getProximalMap(p::Pr,λ,x::P,i) where {Pr <: Problem, P <: MPoint} =
    throw(Exception("no $(i)th proximal map found in $(typeof(p)) to evaluate for $(typeof(x)) with $(typeof(λ))."))

@doc doc"""
    GradientProblem <: Problem
specify a problem for gradient based algorithms.


# Fields
* `M`            : a manifold $\mathcal M$
* `costFunction` : a function $F\colon\mathcal M\to\mathbb R$ to minimize
* `gradient`     : the gradient $\nabla F\colon\mathcal M
  \to \mathcal T\mathcal M$ of the cost function $F$

# See also
[`steepestDescent`](@ref), [`conjugateGradientDescent`](@ref),
[`GradientDescentOptions`](@ref), [`ConjugateGradientOptions`](@ref)

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

evaluate the gradient of a [`GradientProblem`](@ref)`p` at the [`MPoint`](@ref)` x`.
"""
function getGradient(p::P,x::MP) where {P <: GradientProblem{M} where M <: Manifold, MP <: MPoint}
  return p.gradient(x)
end
"""
    getCost(p,x)

evaluate the cost function `F` stored within a [`GradientProblem`](@ref) at the [`MPoint`](@ref)` x`.
"""
function getCost(p::P,x::MP) where {P <: GradientProblem{M} where M <: Manifold, MP <: MPoint}
  return p.costFunction(x)
end
"""
    HessianProblem <: Problem
For now this is just a dummy problem to carry information about a Problem also providing a Hessian
"""
mutable struct HessianProblem{mT <: Manifold} <: Problem
    M::mT
    costFunction::Function
    Heassian::Function
end
@doc doc"""
    getHessian(p,x)
evakuate the Hessian of a [`HessianProblem`](@ref)` p` at the [`MPoint`](@ref)` x`.
"""
function getHessian(p::P,x::MP) where {P <: HessianProblem{M} where M <: Manifold, MP <: MPoint }
    return p.Hessian(x)
end
@doc doc"""
    ProximalProblem <: Problem
specify a problem for solvers based on the evaluation of proximal map(s).

# Fields
* `M`            : a manifold $\mathcal M$
* `costFunction` : a function $F\colon\mathcal M\to\mathbb R$ to minimize
* `proximalMaps` : proximal maps $\operatorname{prox}_{\lambda\varphi}\colon\mathcal M\to\mathcal M$
  as functions (λ,x) -> y, i.e. the prox parameter λ also belongs to the signature of the proximal map.
# See also
[`cyclicProximalPoint`](@ref), [`getCost`](@ref),
[`getProximalMaps`](@ref),[`getProximalMap`](@ref),
"""
mutable struct ProximalProblem{mT <: Manifold} <: Problem
  M::mT
  costFunction::Function
  proximalMaps::Array{Function,N} where N
end
# Access Functions for proxes & cost.
#
"""
    getCost(p,x)

evaluate the cost function `F` stored within a [`ProximalProblem`](@ref) at the [`MPoint`](@ref)` x`.
"""
function getCost(p::P,x::MP) where {P <: ProximalProblem{M} where M <: Manifold, MP <: MPoint}
  return p.costFunction(x)
end
@doc doc"""
    getProximalMaps(p,λ,x)
evaluate all proximal maps of `ProximalProblem p` at the point `x` of `p.M` and
some `λ`$>0$ which might be given as a vector the same length as the number of
proximal maps.
"""
getProximalMaps(p::P,λ,x::MP) where {P <: ProximalProblem{M} where M <: Manifold, MP<:MPoint} =
    p.proximalMaps.(λ,x);
@doc doc"""
    getProximalMap(p,λ,x,i)
evaluate the `i`th proximal map of `ProximalProblem p` at the point `x` of `p.M` with parameter `λ`$>0$.
"""
function getProximalMap(p::P,λ,x::MP,i) where {P <: ProximalProblem{M} where M <: Manifold, MP<:MPoint}
    if i>length(p.proximalMaps)
        ErrorException("the $(i)th entry does not exists, only $(len(p.proximalMaps)) available.")
    end
    return p.proximalMaps[i].(λ,x);
end
