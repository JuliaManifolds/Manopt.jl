@doc doc"""
    conjugate_gradient_descent(M, F, ∇F, x)

perform a conjugate gradient based descent $x_{k+1} = \mathrm{retr}_{x_k} s_k\delta_k$
whith different rules to compute the direction $\delta_k$ based on the last direction
$\delta_{k-1}$ and both gradients $\nabla f(x_k)$,$\nabla f(x_{k-1})$ are available.
Further, the step size $s_k$ may be refined by a line search.

# Input
* `M` : a manifold $\mathcal M$
* `F` : a cost function $F\colon\mathcal M\to\mathbb R$ to minimize
* `∇F`: the gradient $\nabla F\colon\mathcal M\to T\mathcal M$ of F
* `x` : an initial value $x\in\mathcal M$

# Optional
* `coefficient` : ([`SteepestDirectionUpdateRule`](@ref) `<:` [`DirectionUpdateRule`](@ref)
  rule to compute the descent direction update coefficient $\beta_k$, i.e.

  $\delta_k=\nabla f(x_k) + \beta_k \delta_{k-1}$

  as a functor, i.e. the resulting function maps `(p,o,i) -> β`, where
  `p` is the current [`GradientProblem`](@ref), `o` are the
  [`ConjugateGradientDescentOptions`](@ref) `o` and `i` is the current iterate.

  Available rules are: [`SteepestDirectionUpdateRule`](@ref),
  [`HeestenesStiefelCoefficient`](@ref), [`FletcherReevesCoefficient`](@ref),
  [`PolyakCoefficient`](@ref), [`ConjugateDescentCoefficient`](@ref),
  [`LiuStoreyCoefficient`](@ref), [`DaiYuanCoefficient`](@ref),
  [`HagerZhangCoefficient`](@ref).

* `stepsize` - (Constant(1.)`) A tuple `(sF,sO)`
  consisting of a step size function `sF` (used in [`get_stepsize`](@ref) and
  its corresponding options `sO`. The default is a constant step size 1.
* `retraction` - (`ExponentialRetraction`) a type of retraction to use.
* `stoppingCriterion` : (`stopWhenAny( stopAtIteration(200), stopGradientNormLess(10.0^-8))`)
  a function indicating when to stop.
* `vector_transport_method` – (`ParallelTransport()`) vector transport method to transport
  the old descent direction when computing the new descent direction.

# Output
* `xOpt` – the resulting (approximately critical) point of gradientDescent
OR
* `options` - the options returned by the solver (see `return_options`)
"""
function conjugateGradientDescent(
    M::Manifold,
    F::Function,
    ∇F::Function,
    x;
    coefficient::DirectionUpdateRule = SteepestDirectionUpdateRule(),
    stepsize::Stepsize = ConstantStepsize(1.),
    retraction::AbstractRetractionMethod = ExponentialRetraction(),
    stoppingCriterion::StoppingCriterion = stopWhenAny(
        stopAfterIteration(500), stopWhenGradientNormLess(10^(-8))
    ),
    transport_method = ParallelTransport(),
    return_options=false,
    kwargs...
)
    p = GradientProblem(M,F,∇F)
    o = ConjugateGradientDescentOptions(
        x,
        stoppingCriterion,stepsize,
        coefficient,
        retraction_method,
        vector_transport_method,
        )
    o = decorateOptions(o; kwargs...)
    resultO = solve(p,o)
    if return_options
        return resultO
    end
    return get_solver_result(p,resultO)
end
function initialize_solver!(p::GradientProblem,o::ConjugateGradientDescentOptions)
    o.∇ = get_gradient(p,o.x)
    o.δ = -o.∇
    o.β = 0.
end
function step_solver!(p::GradientProblem, o::ConjugateGradientDescentOptions, i)
    xOld = o.x
    o.x = retract(p.M, o.x, get_stepsize!(p, o, iter, o.δ) * o.δ, o.retraction_method)
    o.∇ = get_gradient(p, o.x)
    o.β = o.coefficient(p,o,iter)
    o.δ = -o.∇ + o.β * vector_transport_to(p.M, xOld, o.δ, o.x, o.vector_transport_method)
end
get_solver_result(p::GradientProblem,o::ConjugateGradientDescentOptions)
