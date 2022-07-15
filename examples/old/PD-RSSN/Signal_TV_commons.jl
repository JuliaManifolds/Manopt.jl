#
# Prepare cost, proximal maps and differentials
rep(d) = (d > 1) ? [ones(Int, d)..., d] : d
res(s::Int) = res([s])
res(s) = (length(s) > 1) ? [s..., length(s)] : s
d = length(size(f))
s = size(f)
M = PowerManifold(pixelM, NestedPowerRepresentation(), s...)
M2 = PowerManifold(pixelM, NestedPowerRepresentation(), res(s)...)
N = base_manifold(TangentBundle(M2))
m2(m) = repeat(m; inner=rep(length(size(m))))

# M = PowerManifold(pixelM, NestedPowerRepresentation(), size(f)...)
# N = base_manifold(TangentBundle(M)) # TODO does this actually work in nD?
fidelity(M, x) = 1 / 2 * distance(M, x, f)^2
Λ(M, x) = forward_logs(M, x)
# Λ(M, x) = ProductRepr(x, forward_logs(M, x))
prior(M, x) = norm(norm.(Ref(pixelM), x, Λ(M, x)), 1) # TODO does this actually work in 2D?
# prior(M, x) = norm(norm.(Ref(pixelM), x, submanifold_component(N, Λ(M, x), 2)), 1)
cost(M, x) = (1 / α) * fidelity(M, x) + prior(M, x)

prox_F(M, λ, x) = prox_distance(M, λ / α, f, x, 2)
prox_G_dual(N, n, λ, ξ) = project_collaborative_TV(N, λ, n, ξ, 2.0, Inf, 1.0) # non-isotropic

Dprox_F(M, λ, x, η) = differential_geodesic_startpoint(M, x, f, λ / (α + λ), η)

function differential_project_collaborative_TV(
    N::PowerManifold, λ, x, Ξ, Η, p=2.0, q=1.0, γ=0.0
)
    Y = zero_vector(N, x)
    # print("Ξ = $(Ξ)")

    pdims = power_dimensions(N)
    if length(pdims) == 1
        d = 1
        s = 1
        R = CartesianIndices(Tuple(pdims))
    else
        d = pdims[end]
        s = length(pdims) - 1
        if s != d
            throw(
                ErrorException(
                    "the last dimension ($(d)) has to be equal to the number of the previous ones ($(s)) but its not.",
                ),
            )
        end
        R = CartesianIndices(Tuple(pdims[1:(end - 1)]))
    end

    # R = CartesianIndices(Tuple(power_size))
    maxInd = last(R).I
    e_k_vals = [1 * (1:d .== k) for k in 1:d]

    if q == Inf
        if p == Inf || d == 1
            norms = norm.(Ref(N.manifold), x, Ξ)

            for i in R # iterate over all pixel
                for k in 1:d # for all direction combinations
                    I = i.I # array of index
                    J = I .+ e_k_vals[k] #i + e_k is j
                    if all(J .<= maxInd)
                        # this is neighbor in range,
                        Y[N, I..., k] += if norms[I..., k] <= (1 + λ * γ)
                            Η[N, I..., k] ./ (1 + λ * γ)
                        else
                            1 / norms[I..., k] * (
                                Η[N, I..., k] .-
                                1 / norms[I..., k]^2 .* inner(
                                    N.manifold,
                                    x[N, I..., k],
                                    Η[N, I..., k],
                                    Ξ[N, I..., k],
                                ) .* Ξ[N, I..., k]
                            )
                        end
                    else
                        Y[N, I..., k] = zero_vector(N.manifold, x[N, I..., k])
                    end
                end # directions
            end # i in R
            return Y
        elseif p == 2
            norms = norm.(Ref(N.manifold), x, Ξ)
            norms_ = sqrt.(sum(norms .^ 2; dims=length(pdims))) # TODO check size

            for i in R # iterate over all pixel
                for k in 1:d # for all direction combinations
                    I = i.I # array of index
                    J = I .+ e_k_vals[k] #i + e_k is j
                    if all(J .<= maxInd)
                        # this is neighbor in range,
                        if norms_[I...] <= (1 + λ * γ)
                            Y[N, I..., k] += Η[N, I..., k] ./ (1 + λ * γ)
                        else
                            for κ in 1:d
                                Y[N, I..., κ] += if k != κ
                                    -1 / norms_[I...]^3 * inner(
                                        N.manifold,
                                        x[N, I..., k],
                                        Η[N, I..., k],
                                        Ξ[N, I..., k],
                                    ) .* Ξ[N, I..., κ]
                                else
                                    1 / norms_[I...] * (
                                        Η[N, I..., k] .-
                                        1 / norms_[I...]^2 .* inner(
                                            N.manifold,
                                            x[N, I..., k],
                                            Η[N, I..., k],
                                            Ξ[N, I..., k],
                                        ) .* Ξ[N, I..., k]
                                    )
                                end
                            end
                        end
                    else
                        Y[N, I..., k] = zero_vector(N.manifold, x[N, I..., k])
                    end
                end # directions
            end # i in R
            return Y
        else
            throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
        end
    end # end q
    throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
end

Dprox_G_dual(N, n, λ, ξ, η) = differential_project_collaborative_TV(N, λ, n, ξ, η, 2.0, Inf)
# Dprox_G_dual(N, n, λ, ξ, η; γ=0,isotropic=false) = differential_project_collaborative_TV(N, λ, n, ξ, Inf, Inf, 1.0)

DΛ(M, m, X) = differential_forward_logs(M, m, X)
# DΛ(M, m, X) = ProductRepr(zero_vector(M, m), differential_forward_logs(M, m, X))
adjoint_DΛ(N, m, n, ξ) = adjoint_differential_forward_logs(M, m, ξ)
# adjoint_DΛ(N, m, n, ξ) = adjoint_differential_forward_logs(N.manifold, m, ξ[N, :vector])
