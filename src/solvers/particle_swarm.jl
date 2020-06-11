@doc raw"""
    patricle_swarm(M, F)

perform the particle swarm optimization algorithm (PSO), starting with the initial
particle positions x0.
##Insert source

# Input
* `M` – a manifold $\mathcal M$
* `F` – a cost function $F\colon\mathcal M\to\mathbb R$ to minimize

# Optional
* `x0` – the initial positions of each particle in the swarm $x0_i ∈ \mathcal M$ for $i = 1, \dots, N$
* `retraction_method` – ([`ExponentialRetraction`](@ref)) a `retraction(M,x,ξ)` to use.
* `inverse_retraction_method` - ([`LogarithmicInverseRetraction`](@ref)) an `inverse_retraction(M,x,y)` to use.
* `stopping_criterion` – (`[`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(500), `[`StopWhenChangeLess`](@ref)`(10^{-4})))
  a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.
* `return_options` – (`false`) – if activated, the extended result, i.e. the
    complete [`Options`](@ref) are returned. This can be used to access recorded values.
    If set to false (default) just the optimal value `xOpt` if returned

...
and the ones that are passed to [`decorate_options`](@ref) for decorators.

# Output
* `xOpt` – the resulting point of PSO
OR
* `options` - the options returned by the solver (see `return_options`)
"""
