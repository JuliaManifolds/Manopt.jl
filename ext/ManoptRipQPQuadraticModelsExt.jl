module ManoptRipQPQuadraticModelsExt

using Manopt
import Manopt: bundle_method_subsolver
using ManifoldsBase
using LinearAlgebra: tril
using SparseArrays: sparse

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
    M::A, p_last_serious, linearization_errors, transported_subgradients
) where {A<:AbstractManifold}
    d = length(linearization_errors)
    λ = zeros(d)
    bundle_method_subsolver!(
        M, λ, p_last_serious, linearization_errors, transported_subgradients
    )
    return λ
end
function bundle_method_subsolver!(
    M::A, λ, p_last_serious, linearization_errors, transported_subgradients
) where {A<:AbstractManifold}
    d = length(linearization_errors)
    H = [
        inner(M, p_last_serious, X, Y) for X in transported_subgradients,
        Y in transported_subgradients
    ]
    qm = QuadraticModel(
        linearization_errors,
        sparse(tril(H));
        A=reshape(ones(d), 1, d),
        lcon=[one(eltype(linearization_errors))],
        ucon=[one(eltype(linearization_errors))],
        lvar=zeros(d),
        uvar=[Inf for i in 1:d],
        c0=zero(eltype(linearization_errors)),
    )
    λ .= ripqp(qm; display=false).solution
    return λ
end
function bundle_method_subsolver(
    M::A, p_last_serious, μ, approximation_errors, transported_subgradients
) where {A<:AbstractManifold}
    d = length(approximation_errors)
    λ = zeros(d)
    bundle_method_subsolver!(
        M, λ, p_last_serious, μ, approximation_errors, transported_subgradients
    )
    return λ
end
function bundle_method_subsolver!(
    M::A, λ, p_last_serious, μ, approximation_errors, transported_subgradients
) where {A<:AbstractManifold}
    d = length(approximation_errors)
    H =
        1 / μ * [
            inner(M, p_last_serious, X, Y) for X in transported_subgradients,
            Y in transported_subgradients
        ]
    qm = QuadraticModel(
        approximation_errors,
        sparse(tril(H));
        A=reshape(ones(d), 1, d),
        lcon=[one(eltype(approximation_errors))],
        ucon=[one(eltype(approximation_errors))],
        lvar=zeros(d),
        uvar=[Inf for i in 1:d],
        c0=zero(eltype(approximation_errors)),
    )
    λ .= ripqp(qm; display=false).solution
    return λ
end
end
