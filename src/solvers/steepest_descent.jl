#
# A simple steepest descent algorithm implementation
#
export initialize_solver!, step_solver!, get_solver_result
export steepest_descent
export DebugGradient, DebugGradientNorm, DebugStepsize
@doc raw"""
    steepest_descent(M, F, ∇F, x)
perform a steepest_descent $x_{k+1} = \mathrm{retr}_{x_k} s_k\nabla f(x_k)$ with
different choices of $s_k$ available (see `stepsize` option below).

# Input
* `M` – a manifold $\mathcal M$
* `F` – a cost function $F\colon\mathcal M\to\mathbb R$ to minimize
* `∇F` – the gradient $\nabla F\colon\mathcal M\to T\mathcal M$ of F
* `x` – an initial value $x ∈ \mathcal M$

# Optional
* `stepsize` – ([`ConstantStepsize`](@ref)`(1.)`) specify a [`Stepsize`](@ref)
  functor.
* `retraction` – (`exp`) a `retraction(M,x,ξ)` to use.
* `stoppingCriterion` – (`[`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(200), `[`StopWhenGradientNormLess`](@ref)`(10.0^-8))`)
  a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.
* `returnOptions` – (`false`) – if actiavated, the extended result, i.e. the
    complete [`Options`](@ref) re returned. This can be used to access recorded values.
    If set to false (default) just the optimal value `xOpt` if returned
...
and the ones that are passed to [`decorate_options`](@ref) for decorators.

# Output
* `xOpt` – the resulting (approximately critical) point of gradientDescent
OR
* `options` - the options returned by the solver (see `returnOptions`)
"""
function steepest_descent(M::mT,
    F::Function, ∇F::Function, x;
    stepsize::Stepsize = ConstantStepsize(1.0),
    retraction::Function = exp,
    stoppingCriterion::StoppingCriterion = StopWhenAny( StopAfterIteration(200), StopWhenGradientNormLess(10.0^-8)),
    returnOptions=false,
    kwargs... #collect rest
  ) where {mT <: Manifold}
  p = GradientProblem(M,F,∇F)
  o = GradientDescentOptions(M, x, stoppingCriterion,stepsize,retraction)
  o = decorate_options(o; kwargs...)
  resultO = solve(p,o)
  if returnOptions
    return resultO
  else
    return get_solver_result(resultO)
  end
end
#
# Solver functions
#
function initialize_solver!(p::P,o::O) where {P <: GradientProblem, O <: GradientDescentOptions}
    o.∇ = getGradient(p,o.x)
end
function step_solver!(p::P,o::O,iter) where {P <: GradientProblem, O <: GradientDescentOptions}
    o.∇ = getGradient(p,o.x)
    o.x = o.retraction(p.M, o.x , -get_stepsize(p,o,iter) * o.∇)
end
get_solver_result(o::O) where {O <: GradientDescentOptions} = o.x