function bundle_method_sub_solver(M::A, bms::BundleMethodState) where {A<:AbstractManifold}
    d = length(bms.approx_errors)
    H = [
        inner(M, bms.p_last_serious, X, Y) for X in bms.transported_subgradients,
        Y in bms.transported_subgradients
    ]
    B = reshape(ones(d), 1, d)
    qm = QuadraticModel(
        bms.approx_errors,
        sparse(tril(H));
        A=B,
        lcon=[1.0],
        ucon=[1.0],
        lvar=zeros(d),
        uvar=[Inf for i in 1:d],
        c0=0.0,
    )
    return ripqp(qm; display=false).solution
    # return ripqp(qm; itol=RipQP.InputTol(ϵ_rb=1e-6, ϵ_rc=1e-6), mode=:zoom).solution
end
function bundle_method_sub_solver(
    M::A, bms::ProxBundleMethodState
) where {A<:AbstractManifold}
    d = length(bms.approx_errors)
    H =
        1 / bms.μ * [
            inner(M, bms.p_last_serious, X, Y) for X in bms.transported_subgradients,
            Y in bms.transported_subgradients
        ]
    B = reshape(ones(d), 1, d)
    qm = QuadraticModel(
        bms.approx_errors,
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
