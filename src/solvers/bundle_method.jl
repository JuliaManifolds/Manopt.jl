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
    tol::Real=1e-10,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    inverse_retraction_method::IR=default_inverse_retraction_method(M),
    retraction_method::TRetr=default_retraction_method(M),
    stopping_criterion::StoppingCriterion=StopAfterIteration(5000),
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
    o.index_set = Set(1) # initialize index set
    o.bundle_points = [o.p, o.X]
    o.lin_errors = [0]
    o.p_last_serious = o.bundle_points[1, 1]
    o.X = zero_vector(prb.M, o.bundle_points[1, 1])
    return o
end
# function subsolver(X, o)
#     f(λ) = 0.5 * (sum(λ .* X))^2 + sum(λ .* o.lin_errs)
#     gradf(λ) = abs(sum(λ .* X)) * X + o.lin_errs
#     g(λ) = -λ
#     gradg(λ) = -I(length(o.index_set))
#     h(λ) = sum(λ) - 1
#     gradh(λ) = ones(length(o.index_set))
#     o.sub_problem = ConstrainedProblem(
#         ℝ^(length(o.index_set)), f, gradf, g, gradg, h, gradh
#     )
#     o.sub_options = decorate_options(
#         GradientDescentOptions(
#             copy(λ); initial_gradient=zero_vector(ℝ^(length(o.index_set), λ))
#         ),
#     )
#     return get_solver_result(solve(o.sub_problem, o.sub_options))
# end
function step_solver!(prb::BundleProblem, o::BundleMethodOptions, iter)
    get_bundle_subgradient!(prb, o.X, o.p)
    #o.bundle_points = hcat(o.bundle_points, [o.p, o.X])
    transported_subgrads = [
        vector_transport_to(
            prb.M,
            o.bundle_points[1, j],
            o.bundle_points[2, j],
            o.p_last_serious,
            o.vector_transport_method,
        ) for j in 1:length(o.index_set)
    ]
    # compute a solution λ of the minimization subproblem with some other solver
    λ = BundleMethodSubsolver(prb, o, transported_subgrads)
    g = sum(λ .* transported_subgrads)
    ε = sum(λ .* o.lin_errors)
    δ = -norm(prb.M, o.p, o.X)^2 - ε
    if δ <= o.tol
        return o
    else
        q = retract(M, o.p_last_serious, -g, o.retraction_method)
        X_q = get_bundle_subgradient(prb, q) # not sure about this
        if get_cost(prb, q) <= (get_cost(prb, o.p_last_serious) + o.m * δ)
            o.p_last_serious = q
            o.bundle_points = hcat(o.bundle_points, [o.p_last_serious, X_q])
        else
            o.bundle_points = hcat(o.bundle_points, [q, X_q])
        end
    end
    positive_indices = intersect(o.index_set, Set(findall(j -> j > 0, λ)))
    o.index_set = union(positive_indices, iter + 1)
    o.lin_errors = [
        get_cost(prb, o.p_last_serious) - get_cost(prb, o.bundle_points[1, j]) - inner(
            prb.M,
            o.bundle_points[1, j],
            o.bundle_points[2, j],
            inverse_retract(
                prb.M, o.bundle_points[1, j], o.p_last_serious, o.inverse_retraction_method
            ),
        ) for j in 1:length(o.index_set)
    ]
    return o
end
get_solver_result(o::BundleMethodOptions) = o.p_last_serious

# Debugging
# M = SymmetricPositiveDefinite(3)
# F(M,y) = sum(1 / (2 * length(y)) * distance.(Ref(M), data, Ref(y)) .^ 2)
# ∇F(M,y) = sum(1 / length(y) * grad_distance.(Ref(M), data, Ref(y)))
# data = [rand(M) for i = 1:100];
# bundle_method(M, F, ∇F, data[1])
