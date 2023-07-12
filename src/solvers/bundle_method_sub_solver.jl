function bundle_method_sub_solver(M::A, bms::BundleMethodState) where {A<:AbstractManifold}
    d = length(findall(x -> x == true, bms.active_indices))
    H = [
        inner(M, bms.p_last_serious, X, Y) for
        X in @view(bms.transported_subgradients[bms.active_indices]),
        Y in @view(bms.transported_subgradients[bms.active_indices])
    ]
    qm = QuadraticModel(
        bms.lin_errors[bms.active_indices],
        sparse(tril(H));
        A=reshape(ones(d), 1, d),
        lcon=[one(eltype(bms.lin_errors))],
        ucon=[one(eltype(bms.lin_errors))],
        lvar=zeros(d),
        uvar=[Inf for i in 1:d],
        c0=zero(eltype(bms.lin_errors)),
    )
    return ripqp(qm; display=false).solution
    # return ripqp(qm; itol=RipQP.InputTol(ϵ_rb=1e-6, ϵ_rc=1e-6), mode=:zoom).solution
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
