function bundle_method_sub_solver(M::A, bms::BundleMethodState) where {A<:AbstractManifold}
    d = length(bms.lin_errors)
    H = [
        inner(M, bms.p_last_serious, X, Y) for X in bms.transported_subgradients,
        Y in bms.transported_subgradients
    ]
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
