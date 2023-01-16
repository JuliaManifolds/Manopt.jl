function bundle_method_sub_solver(
    M::A, bms::BundleMethodState, X::T
) where {A<:AbstractManifold,T}
    d = length(bms.index_set)
    H = reduce(hcat, [[inner(M, bms.p_last_serious, X[i], X[j]) for j in 1:d] for i in 1:d])'
    B = reshape(ones(d), 1, d)
    qm = QuadraticModel(
        bms.lin_errors, tril(H); A=B, lcon=[1.0], ucon=[1.0], lvar=zeros(d), c0=0.0
    )
    return ripqp(qm; display=false).solution
end
