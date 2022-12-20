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
    m::Real=0.0125,
    inverse_retraction_method::IR=default_inverse_retraction_method(M),
    retraction_method::TRetr=default_retraction_method(M),
    stopping_criterion::StoppingCriterion=StopAfterIteration(5000),
    tol::Real=1e-10,
    vector_transport_method::VTransp=default_vector_transport_method(M),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs..., #especially may contain debug
) where {TF,TdF,TRetr,VTransp}
    prb = BundleProblem(M, F, ∂F!!; evaluation=evaluation)
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
    o.p_last_serious = o.bundle_points[1, 1]
    o.∂ = zero_vector(prb.M, o.bundle_points[1, 1])
    # o.J = Set(1) # initialize index set
    # o.bundle_points = [o.p, o.∂]
    # o.lin_errors = [0]
    return o
end
using JuMP, COSMO, Ipopt
function jump_subsolver(J, lin_errs, X)
    vector_model = Model(COSMO.Optimizer)#, add_bridges = false) -- Removing bridges breaks this
    set_optimizer_attribute(vector_model, "verbose", false)
    @variable(vector_model, λ[1:length(J)])
    @constraint(vector_model, λ .>= 0)
    @constraint(vector_model, sum(λ) == 1)
    @objective(vector_model, Min, 0.5 * (sum(λ .* X))^2 + sum(λ .* lin_errs))
    optimize!(vector_model)
    return value.(λ), objective_value(vector_model)
end
function step_solver!(prb::BundleProblem, o::BundleMethodOptions, iter)
    get_subgradient!(prb, o.∂, o.p)
    o.bundle_points = hcat([o.bundle_points], [o.p, o.∂])
    transported_subgrads = [
        vector_transport_to!(
            prb.M,
            o.bundle_points[1, j],
            o.bundle_points[2, j],
            o.p_last_serious,
            o.vector_transport_method,
        ) for j in 1:length(o.J)
    ]
    # compute a solution λ of the minimization subproblem with some other solver
    λ = jump_subsolver(o.J, o.lin_errors, transported_subgrads)
    g = sum(λ .* transported_subgrads)
    ε = sum(λ .* o.lin_errors)
    δ = -norm(prb.M, o.p, o.∂)^2 - ε
    if δ <= o.tol
        return o
    else
        q = retract(M, o.p_last_serious, -g, o.retraction_method)
        ∂_q = get_subgradient(prb, o.∂, q) # not sure about this
        if get_cost(prb, q) <= (get_cost(prb, o.p_last_serious) + o.m * δ)
            o.p_last_serious = q
            o.bundle_points = hcat(o.bundle_points, [o.p_last_serious, ∂_q])
        else
            o.bundle_points = hcat(o.bundle_points, [q, ∂_q])
        end
    end
    J_positive = intersect(o.J, Set(findall(j -> j > 0, λ)))
    o.J = union(J_positive, iter + 1)
    o.lin_errors = []
    for j in 1:length(o.J)
        push!(
            o.lin_errors,
            get_cost(prb, o.p_last_serious) - get_cost(prb, o.bundle_points[1, j]) - inner(
                prb.M,
                o.bundle_points[1, j],
                o.bundle_points[2, j],
                inverse_retract(
                    prb.M,
                    o.bundle_points[1, j],
                    o.p_last_serious,
                    o.inverse_retraction_method,
                ),
            ),
        )
    end
    return o
end
get_solver_result(o::BundleMethodOptions) = o.p_last_serious
