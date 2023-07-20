function bundle_method_sub_solver(M::A, bms::BundleMethodState) where {A<:AbstractManifold}
    d = length(bms.lin_errors)
    H = [
        inner(M, bms.p_last_serious, X, Y) for X in bms.transported_subgradients,
        Y in bms.transported_subgradients
    ]
    qm = QuadraticModel(
        bms.lin_errors,
        sparse(tril(H));
        A=reshape(ones(d), 1, d),
        lcon=[one(eltype(bms.lin_errors))],
        ucon=[one(eltype(bms.lin_errors))],
        lvar=zeros(d),
        uvar=[Inf for i in 1:d],
        c0=zero(eltype(bms.lin_errors)),
    )
    return ripqp(qm; display=false).solution
end
function bundle_method_sub_solver(
    M::A, pbms::ProxBundleMethodState
) where {A<:AbstractManifold}
    d = length(pbms.approx_errors)
    H =
        1 / pbms.μ * [
            inner(M, pbms.p_last_serious, X, Y) for X in pbms.transported_subgradients,
            Y in pbms.transported_subgradients
        ]
    qm = QuadraticModel(
        pbms.approx_errors,
        sparse(tril(H));
        A=reshape(ones(d), 1, d),
        lcon=[one(eltype(pbms.lin_errors))],
        ucon=[one(eltype(pbms.lin_errors))],
        lvar=zeros(d),
        uvar=[Inf for i in 1:d],
        c0=zero(eltype(pbms.lin_errors)),
    )
    return ripqp(qm; display=false).solution
end
