function prox_bundle_method_sub_solver(M::A, bms::BundleMethodState) where {A<:AbstractManifold}
    d = length(bms.lin_errors)
    H = bms.μ/2 * local_metric(M, bms.p_last_serious, DefaultOrthonormalBasis())
    B = reshape(ones(d), 1, d)
    qm = QuadraticModel(
        bms.lin_errors,
        tril(H);
        A=B,
        lcon=[1.0],
        ucon=[1.0],
        lvar=zeros(d),
        uvar=[Inf for i in 1:d],
        c0=0.0,
    )
    return ripqp(qm; display=false).solution
end
