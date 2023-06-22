function bundle_method_sub_solver(M::A, bms::BundleMethodState) where {A<:AbstractManifold}
    d = length(findall(x -> x != 0, bms.indices))
    H = [
        inner(
            M,
            bms.p_last_serious,
            bms.transported_subgradients[i],
            bms.transported_subgradients[j],
        ) for i in 1:d, j in 1:d
    ]
    vector_model = Model(Ipopt.Optimizer)#, add_bridges = false) -- Removing bridges breaks this
    set_optimizer_attribute(vector_model, "print_level", 0)
    # set_optimizer_attribute(vector_model, "verbose", false)
    @variable(vector_model, λ[1:d])
    @constraint(vector_model, λ[1:d] .>= 0)
    @constraint(vector_model, sum(λ[1:d]) == 1)
    @objective(vector_model, Min, 0.5 * (sum(λ .* H))^2 + sum(λ .* bms.lin_errors[1:d]))
    optimize!(vector_model)
    return JuMP.value.(λ)#, objective_value(vector_model)
end
# function bundle_method_sub_solver(M::A, bms::BundleMethodState) where {A<:AbstractManifold}
#     d = length(findall(x -> x != 0, bms.indices))
#     H = [
#         inner(
#             M,
#             bms.p_last_serious,
#             bms.transported_subgradients[i],
#             bms.transported_subgradients[j],
#         ) for i in 1:d, j in 1:d
#     ]
#     qm = QuadraticModel(
#         bms.lin_errors[1:d],
#         sparse(tril(H));
#         A=reshape(ones(d), 1, d),
#         lcon=[1.0],
#         ucon=[1.0],
#         lvar=zeros(d),
#         uvar=[Inf for i in 1:d],
#         c0=0.0,
#     )
#     return ripqp(qm; display=false).solution
#     # return ripqp(qm; itol=RipQP.InputTol(ϵ_rb=1e-6, ϵ_rc=1e-6), mode=:zoom).solution
# end
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
