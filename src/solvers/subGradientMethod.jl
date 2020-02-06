#
# A simple steepest descent algorithm implementation
#
export subGradientMethod
@doc raw"""
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
* `x` – an initial value $x ∈ \mathcal M$

# Optional
* `stepsize` – ([`ConstantStepsize`](@ref)`(1.)`) specify a [`Stepsize`](@ref)
* `retraction` – (`exp`) a `retraction(M,x,ξ)` to use.
* `stoppingCriterion` – ([`stopWhenAny`](@ref)`(`[`stopAfterIteration`](@ref)`(200), `[`stopWhenGradientNormLess`](@ref)`(10.0^-8))`)
  a functor, see[`StoppingCriterion`](@ref), indicating when to stop.
* `returnOptions` – (`false`) – if actiavated, the extended result, i.e. the
    complete [`Options`](@ref) re returned. This can be used to access recorded values.
    If set to false (default) just the optimal value `xOpt` if returned
...
and the ones that are passed to [`decorateOptions`](@ref) for decorators.

# Output
* `xOpt` – the resulting (approximately critical) point of gradientDescent
OR
* `options` - the options returned by the solver (see `returnOptions`)
"""
function subGradientMethod(M::mT,
        F::Function,
        ∂F::Function,
        x;
        retraction::Function = exp,
        stepsize::Stepsize = DecreasingStepsize( typicalDistance(M)/5),
        stoppingCriterion::StoppingCriterion = stopAfterIteration(5000),
        returnOptions = false,
        kwargs... #especially may contain debug
    ) where {mT <: Manifold}
    p = SubGradientProblem(M,F,∂F)
    o = SubGradientMethodOptions(x,stoppingCriterion, stepsize, retraction)
    o = decorateOptions(o; kwargs...)
    resultO = solve(p,o)
    if returnOptions
        return resultO
    else
        return getSolverResult(resultO)
    end
end
function initializeSolver!(p::SubGradientProblem, o::SubGradientMethodOptions)
    o.xOptimal = o.x
    zero_tangent_vector!(p.M,o.∂,o.x)
end
function doSolverStep!(p::SubGradientProblem, o::SubGradientMethodOptions,iter)
    o.∂ = getSubGradient(p,o.x)
    s = getStepsize!(p,o,iter)
    o.x = o.retraction(p.M,o.x,-s*o.∂)
    if getCost(p,o.x) < getCost(p,o.xOptimal)
        o.xOptimal = o.x
    end
end
getSolverResult(o::SubGradientMethodOptions) = o.xOptimal

#
# TODO specific debugs and records.
#