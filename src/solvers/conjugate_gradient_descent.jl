@doc raw"""
    conjugate_gradient_descent(M, F, ∇F, x)

perform a conjugate gradient based descent
````math
x_{k+1} = \operatorname{retr}_{x_k} \bigl( s_k\delta_k \bigr),
````
where $\operatorname{retr}$ denotes a retraction on the `Manifold` `M`
and one can employ different rules to update the descent direction $\delta_k$ based on
the last direction $\delta_{k-1}$ and both gradients $\nabla f(x_k)$,$\nabla f(x_{k-1})$.
The [`Stepsize`](@ref) $s_k$ may be determined by a [`Linesearch`](@ref).

Available update rules are [`SteepestDirectionUpdateRule`](@ref), which yields a [`gradient_descent`](@ref),
[`ConjugateDescentCoefficient`](@ref), [`DaiYuanCoefficient`](@ref), [`FletcherReevesCoefficient`](@ref),
[`HagerZhangCoefficient`](@ref), [`HeestenesStiefelCoefficient`](@ref),
[`LiuStoreyCoefficient`](@ref), and [`PolakRibiereCoefficient`](@ref).

They all compute $\beta_k$ such that this algorithm updates the search direction as
````math
\delta_k=\nabla f(x_k) + \beta_k \delta_{k-1}
````

# Input
* `M` : a manifold $\mathcal M$
* `F` : a cost function $F\colon\mathcal M\to\mathbb R$ to minimize
* `∇F`: the gradient $∇ F\colon\mathcal M\to T\mathcal M$ of F
* `x` : an initial value $x\in\mathcal M$

# Optional
* `coefficient` : ([`SteepestDirectionUpdateRule`](@ref) `<:` [`DirectionUpdateRule`](@ref)
  rule to compute the descent direction update coefficient $\beta_k$,
  as a functor, i.e. the resulting function maps `(p,o,i) -> β`, where
  `p` is the current [`GradientProblem`](@ref), `o` are the
  [`ConjugateGradientDescentOptions`](@ref) `o` and `i` is the current iterate.
* `retraction_method` - (`ExponentialRetraction`) a retraction method to use, by default the exponntial map
* `return_options` – (`false`) – if actiavated, the extended result, i.e. the
    complete [`Options`](@ref) re returned. This can be used to access recorded values.
    If set to false (default) just the optimal value `xOpt` if returned
* `stepsize` - (`Constant(1.)`) A [`Stepsize`](@ref) function applied to the
  search direction. The default is a constant step size 1.
* `stopping_criterion` : (`stopWhenAny( stopAtIteration(200), stopGradientNormLess(10.0^-8))`)
  a function indicating when to stop.
* `vector_transport_method` – (`ParallelTransport()`) vector transport method to transport
  the old descent direction when computing the new descent direction.

# Output
* `xOpt` – the resulting (approximately critical) point of gradientDescent
OR
* `options` - the options returned by the solver (see `return_options`)
"""
function conjugate_gradient_descent(
    M::Manifold,
    F::TF,
    ∇F::TDF,
    x;
    coefficient::DirectionUpdateRule = SteepestDirectionUpdateRule(),
    stepsize::Stepsize = ConstantStepsize(1.),
    retraction_method::AbstractRetractionMethod = ExponentialRetraction(),
    stopping_criterion::StoppingCriterion = StopWhenAny(
        StopAfterIteration(500), StopWhenGradientNormLess(10^(-8))
    ),
    transport_method = ParallelTransport(),
    return_options=false,
    kwargs...
) where {TF, TDF}
    p = GradientProblem(M,F,∇F)
    o = ConjugateGradientDescentOptions(
        x,
        stopping_criterion,stepsize,
        coefficient,
        retraction_method,
        transport_method,
        )
    o = decorate_options(o; kwargs...)
    resultO = solve(p,o)
    if return_options
        return resultO
    end
    return get_solver_result(resultO)
end
function initialize_solver!(p::GradientProblem,o::ConjugateGradientDescentOptions)
    o.∇ = get_gradient(p,o.x)
    o.δ = -o.∇
    o.β = 0.
end
function step_solver!(p::GradientProblem, o::ConjugateGradientDescentOptions, i)
    xOld = o.x
    o.x = retract(p.M, o.x, get_stepsize(p, o, i, o.δ) * o.δ, o.retraction_method)
    o.∇ = get_gradient(p, o.x)
    o.β = o.coefficient(p,o,i)
    o.δ = -o.∇ + o.β * vector_transport_to(p.M, xOld, o.δ, o.x, o.vector_transport_method)
end
get_solver_result(o::ConjugateGradientDescentOptions) = o.x
