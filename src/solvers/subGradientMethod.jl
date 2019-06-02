#
# A simple steepest descent algorithm implementation
#
export subGradientMethod
@doc doc"""
    subGradientMethod(M, F, ∂F, x)
perform a subgradient method $x_{k+1} = \mathrm{retr}(x_k, s_k∂F(x_k))$,

where $\mathrm{retr}$ is a retraction, $s_k$ can be specified as a function but is
usually set to a constant value. Though the subgradient might be set valued,
the argument `∂F` should always return _one_ element from the subgradient.

# Input
* `M` – a manifold $\mathcal M$
* `F` – a cost function $F\colon\mathcal M\to\mathbb R$ to minimize
* `∂F`: the (sub)gradient $\partial F\colon\mathcal M\to T\mathcal M$ of F
  restricted to always only returning one value/element from the subgradient
* `x` – an initial value $x\in\mathcal M$

# Optional
* `stepsize` – ([`ConstantStepsize`](@ref)`(1.)`) specify a [`Stepsize`](@ref)
* `retraction` – (`exp`) a `retraction(M,x,ξ)` to use.
* `stoppingCriterion` – ([`stopWhenAny`](@ref)`(`[`stopAfterIteration`](@ref)`(200), `[`stopWhenGradientNormLess`](@ref)`(10.0^-8))`)
  a functor, see[`StoppingCriterion`](@ref), indicating when to stop.

and the ones that are passed to [`decorateOptions`](@ref) for decorators.

# Output
* `xOpt` – the resulting (approximately critical) point of gradientDescent
* `record` - if activated (using the `record` key, see [`RecordOptions`](@ref)
  an array containing the recorded values.
"""
function subGradientMethod(M::mT,
        F::Function, ∂F::Function, x::MP;
        retraction::Function = exp,
        stepsize::Stepsize = DecreasingStepsize( typicalDistance(M)/5),
        stoppingCriterion::StoppingCriterion = stopAfterIteration(5000),
        kwargs... #especially may contain debug
    ) where {mT <: Manifold, MP <: MPoint}
    p = SubGradientProblem(M,F,∂F)
    o = SubGradientMethodOptions(x,stoppingCriterion, stepsize, retraction)

    o = decorateOptions(o; kwargs...)
    resultO = solve(p,o)
    if hasRecord(resultO)
        return getSolverResult(p,getOptions(resultO)), getRecord(resultO)
    end
    return getSolverResult(p,resultO)
end
function initializeSolver!(p::SubGradientProblem, o::SubGradientMethodOptions)
    o.xOptimal = o.x
    o.∂ = zeroTVector(p.M,o.x)
end
function doSolverStep!(p::SubGradientProblem, o::SubGradientMethodOptions,iter)
    o.∂ = getSubGradient(p,o.x)
    s = getStepsize!(p,o,iter)
    o.x = o.retraction(p.M,o.x,-s*o.∂)
    if getCost(p,o.x) < getCost(p,o.xOptimal)
        o.xOptimal = o.x
    end
end
function getSolverResult(p::SubGradientProblem, o::SubGradientMethodOptions)
    return o.xOptimal
end

#
# TODO specific debugs and records.
#