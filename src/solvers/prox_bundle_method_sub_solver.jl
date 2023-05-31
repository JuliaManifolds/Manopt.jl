function prox_bundle_method_sub_solver(mp::AbstractManoptProblem, bms::ProxBundleMethodState, ej, Xj)
    d = length(bms.lin_errors)
    M = get_manifold(mp)
    H = bms.μ * local_metric(get_embedding(M.manifold), bms.p_last_serious)
    c = transpose(Xj) * local_metric(get_embedding(M.manifold), bms.p_last_serious)
    qm = QuadraticModel(c, tril(H); c0=get_cost(mp, bms.p_last_serious)-ej)
    return ripqp(qm; display=false).solution
end
