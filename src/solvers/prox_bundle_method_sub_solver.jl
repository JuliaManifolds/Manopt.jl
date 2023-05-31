function prox_bundle_method_sub_solver(mp::AbstractManoptProblem, bms::BundleMethodState)
    d = length(bms.lin_errors)
    M = get_manifold(mp)
    H = bms.μ * local_metric(M, bms.p_last_serious, DefaultOrthonormalBasis())
    c = maximum([-ej + Xj * local_metric(M, bms.p_last_serious, DefaultOrthonormalBasis()) for (ej, Xj) in zip(bms.approx, bms.transported_subdradients)])
    qm = QuadraticModel(
        c,
        tril(H);
        A=B,
        c0=get_cost(mp, bms.p_last_serious),
    )
    return ripqp(qm; display=false).solution
end
