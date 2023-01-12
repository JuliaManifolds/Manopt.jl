@doc raw"""
    bundle_method(M, F, gradF, x)

perform a bundle method ``p_{k+1} = \mathrm{retr}(p_k, gradF(p_k))``,

where ``\mathrm{retr}`` is a retraction, ``s_k`` can be specified as a function but is
usually set to a constant value. Though the subgradient might be set valued,
the argument `∂F` should always return _one_ element from the subgradient, but
not necessarily deterministic.

# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F:\mathcal M→ℝ`` to minimize
* `gradF`– the (sub)gradient ``\partial F: \mathcal M→ T\mathcal M`` of F
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
function bundle_method(M::AbstractManifold, F::TF, gradF::TdF, p; kwargs...) where {TF,TdF}
    p_res = copy(M, p)
    return bundle_method!(M, F, gradF, p_res; kwargs...)
end
@doc raw"""
    bundle_method!(M, F, gradF, p)

perform a bundle method ``p_{k+1} = \mathrm{retr}(p_k, s_k∂F(p_k))`` in place of `p`

# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F:\mathcal M→ℝ`` to minimize
* `gradF`- the (sub)gradient ``\partial F:\mathcal M→ T\mathcal M`` of F
  restricted to always only returning one value/element from the subgradient.
  This function can be passed as an allocation function `(M, q) -> X` or
  a mutating function `(M, X, q) -> X`, see `evaluation`.
* `p` – an initial value ``p ∈ \mathcal M``

for more details and all optional parameters, see [`bundle_method`](@ref).
"""
function bundle_method!(
    M::AbstractManifold,
    F::TF,
    gradF!!::TdF,
    p;
    m::Real=0.0125,
    tol::Real=1e-8,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    inverse_retraction_method::IR=default_inverse_retraction_method(M),
    retraction_method::TRetr=default_retraction_method(M),
    stopping_criterion::StoppingCriterion=StopWhenAny(
        StopAfterIteration(5000), StopWhenChangeLess(1e-12)
    ),
    vector_transport_method::VTransp=default_vector_transport_method(M),
    kwargs..., #especially may contain debug
) where {TF,TdF,TRetr,IR,VTransp}
    prb = BundleProblem(M, F, gradF!!; evaluation=evaluation)
    o = BundleMethodOptions(
        M,
        p;
        m=m,
        inverse_retraction_method=inverse_retraction_method,
        retraction_method=retraction_method,
        stopping_criterion=stopping_criterion,
        tol=tol,
        vector_transport_method=vector_transport_method,
    )
    o = decorate_options(o; kwargs...)
    return get_solver_return(solve(prb, o))
end
function initialize_solver!(prb::BundleProblem, o::BundleMethodOptions)
    o.p_last_serious = o.p
    o.X = zero_vector(prb.M, o.p)
    return o
end
function bundle_method_sub_solver(::Any, ::Any, ::Any)
    throw(
        ErrorException("""Both packages "QuadraticModels" and "RipQP" need to be loaded.""")
    )
end
function step_solver!(prb::BundleProblem, o::BundleMethodOptions, iter)
    transported_subgradients = [
        vector_transport_to(
            prb.M,
            o.bundle_points[j][1],
            get_bundle_subgradient!(prb, o.bundle_points[j][2], o.bundle_points[j][1]),
            o.p_last_serious,
            o.vector_transport_method,
        ) for j in 1:length(o.index_set)
    ]
    λ = bundle_method_sub_solver(prb.M, o, transported_subgradients)
    g = sum(λ .* transported_subgradients)
    ε = sum(λ .* o.lin_errors)
    if (
        get_cost(prb, o.p) >=
        get_cost(prb, o.p_last_serious) +
        inner(prb.M, o.p_last_serious, g, log(prb.M, o.p_last_serious, o.p)) - ε
    )
        println("Yes")
    else
        println("No")
    end
    δ = -norm(prb.M, o.p_last_serious, g)^2 - ε
    (δ == 0 || -δ <= o.tol) && (return o)
    q = retract(prb.M, o.p_last_serious, -g, o.retraction_method)
    X_q = get_bundle_subgradient(prb, q) # not sure about this
    if get_cost(prb, q) <= (get_cost(prb, o.p_last_serious) + o.m * δ)
        o.p_last_serious = q
        push!(o.bundle_points, (o.p_last_serious, X_q))
    else
        push!(o.bundle_points, (q, X_q))
    end
    positive_indices = intersect(o.index_set, Set(findall(j -> j > 0, λ)))
    o.index_set = union(positive_indices, iter + 1)
    o.lin_errors = [
        get_cost(prb, o.p_last_serious) - get_cost(prb, o.bundle_points[j][1]) - inner(
            prb.M,
            o.bundle_points[j][1],
            o.bundle_points[j][2],
            inverse_retract(
                prb.M, o.bundle_points[j][1], o.p_last_serious, o.inverse_retraction_method
            ),
        ) for j in 1:length(o.index_set)
    ]
    return o
end
get_solver_result(o::BundleMethodOptions) = o.p_last_serious
