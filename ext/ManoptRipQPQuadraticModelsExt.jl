module ManoptRipQPQuadraticModelsExt

using Manopt
import Manopt: bundle_method_subsolver
using ManifoldsBase

if isdefined(Base, :get_extension)
    using QuadraticModels: QuadraticModel
    using RipQP: ripqp
else
    # imports need to be relative for Requires.jl-based workflows:
    # https://github.com/JuliaArrays/ArrayInterface.jl/pull/387
    using ..QuadraticModels: QuadraticModel
    using ..RipQP: ripqp
end

function bundle_method_subsolver(
    M::A, cbms::ConvexBundleMethodState
) where {A<:AbstractManifold}
    d = length(cbms.lin_errors)
    H = [
        inner(M, cbms.p_last_serious, X, Y) for X in cbms.transported_subgradients,
        Y in cbms.transported_subgradients
    ]
    qm = QuadraticModel(
        cbms.lin_errors,
        sparse(tril(H));
        A=reshape(ones(d), 1, d),
        lcon=[one(eltype(cbms.lin_errors))],
        ucon=[one(eltype(cbms.lin_errors))],
        lvar=zeros(d),
        uvar=[Inf for i in 1:d],
        c0=zero(eltype(cbms.lin_errors)),
    )
    return ripqp(qm; display=false).solution
end
function bundle_method_subsolver(
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
end
