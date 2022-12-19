@doc raw"""
    bundle_method(M, F, ∂F, x)

perform a bundle method ``p_{k+1} = \mathrm{retr}(p_k, s_k∂F(p_k))``,

where ``\mathrm{retr}`` is a retraction, ``s_k`` can be specified as a function but is
usually set to a constant value. Though the subgradient might be set valued,
the argument `∂F` should always return _one_ element from the subgradient, but
not necessarily deterministic.

# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F:\mathcal M→ℝ`` to minimize
* `∂F`– the (sub)gradient ``\partial F: \mathcal M→ T\mathcal M`` of F
  restricted to always only returning one value/element from the subgradient.
  This function can be passed as an allocation function `(M, q) -> X` or
  a mutating function `(M, X, q) -> X`, see `evaluation`.
* `p` – an initial value ``p ∈ \mathcal M``

# Optional
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the subgradient works by
   allocation (default) form `∂F(M, q)` or [`MutatingEvaluation`](@ref) in place, i.e. is
   of the form `∂F!(M, X, p)`.
* `retraction` – (`default_retraction_method(M)`) a `retraction(M,p,X)` to use.
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(5000)`)
  a functor, see[`StoppingCriterion`](@ref), indicating when to stop.
...
and the ones that are passed to [`decorate_options`](@ref) for decorators.

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details
"""
function bundle_method(M::AbstractManifold, F::TF, ∂F::TdF, p; kwargs...) where {TF,TdF}
    p_res = copy(M, p)
    return bundle_method!(M, F, ∂F, p_res; kwargs...)
end
@doc raw"""
    bundle_method!(M, F, ∂F, p)

perform a bundle method ``p_{k+1} = \mathrm{retr}(p_k, s_k∂F(p_k))`` in place of `p`

# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F:\mathcal M→ℝ`` to minimize
* `∂F`- the (sub)gradient ``\partial F:\mathcal M→ T\mathcal M`` of F
  restricted to always only returning one value/element from the subgradient.
  This function can be passed as an allocation function `(M, q) -> X` or
  a mutating function `(M, X, q) -> X`, see `evaluation`.
* `p` – an initial value ``p ∈ \mathcal M``

for more details and all optional parameters, see [`bundle_method`](@ref).
"""
function bundle_method!(
    M::AbstractManifold,
    F::TF,
    ∂F!!::TdF,
    p;
    retraction_method::TRetr=default_retraction_method(M),
    stopping_criterion::StoppingCriterion=StopAfterIteration(5000),
    vector_transport_method::VTransp=default_vector_transport_method(M),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs..., #especially may contain debug
) where {TF,TdF,TRetr,VTransp}
    prb = BundleProblem(M, F, ∂F!!; evaluation=evaluation)
    o = BundleMethodOptions(
        M,
        p;
        retraction_method=retraction_method,
        stopping_criterion=stopping_criterion,
        vector_transport_method=vector_transport_method,
    )
    o = decorate_options(o; kwargs...)
    return get_solver_return(solve(prb, o))
end
function initialize_solver!(prb::BundleProblem, o::BundleMethodOptions)
    o.p_last_serious = o.bundle_point[1, 1]
    o.∂ = zero_vector(prb.M, o.bundle_point[1, 1])
    o.J = Set(1) # initialize index set
    o.bundle_point = [o.p, o.∂]
    o.lin_errors = [0]
    return o
end
function step_solver!(prb::BundleProblem, o::BundleMethodOptions, iter)
    get_subgradient!(prb, o.∂, o.p)
    o.bundle_point = hcat([o.bundle_point], [o.p, o.∂])
    # compute a solution λ of the minimization subproblem with some other solver
    g = sum(
        λ .* [
            vector_transport_to!(
                prb.M,
                o.bundle_point[1, j],
                o.bundle_point[2, j],
                o.p_last_serious,
                o.vector_transport_method,
            ) for j in o.J
        ],
    )
    ε = sum(λ .* o.lin_errors)
    δ = -norm(prb.M, o.p, o.∂)^2 - ε
    if δ == 0
        return o
    end
    (get_cost(prb, o.p) <= get_cost(prb, o.p_last_serious) + m * δ) &&
        (o.p_last_serious = retract(M, o.p_last_serious, -g, o.retraction_method))
    o.J_positive = intersect(o.J, Set(findall(j -> j > 0, λ)))
    o.J = union(o.J_positive, iter + 1)
    o.lin_errors = []
    for j in o.J
        push!(
            o.lin_errors,
            get_cost(prb, o.p_last_serious) - get_cost(prb, o.bundle_point[1, j]) - inner(
                TangentSpace(prb.M, o.bundle_point[1, j]),
                o.bundle_point[1, j],
                o.bundle_point[2, j],
                inverse_retract(
                    prb.M,
                    o.bundle_point[1, j],
                    o.p_last_serious,
                    o.inverse_retraction_method,
                ),
            ),
        )
    end
    o.bundle_point = hcat(o.bundle_point, [o.p_last_serious, o.∂])
    return o
end
get_solver_result(o::BundleMethodOptions) = o.p_last_serious
