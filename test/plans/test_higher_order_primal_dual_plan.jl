using Manopt, Manifolds, ManifoldsBase, Test

@testset "Test higher order primal dual plan" begin
    # Perform an really easy test, just compute a mid point
    #
    pixelM = Sphere(2)
    signal_section_size = 1
    M = PowerManifold(pixelM, NestedPowerRepresentation(), 2 * signal_section_size)
    data = [[1.0, 0.0, 0.0], 1 / sqrt(2) .* [1.0, 1.0, 0.0]]
    α = 1
    # known minimizer
    δ = min(α / distance(pixelM, data[1], data[2]), 0.5)
    p_hat = shortest_geodesic(M, data, reverse(data), δ)
    N = M
    fidelity(M, p) = 1 / 2 * distance(M, p, f)^2
    Λ(M, p) = ProductRepr(p, forward_logs(M, p))
    prior(M, p) = norm(norm.(Ref(M.manifold), p, submanifold_component(N, Λ(p), 2)), 1)
    f(M, p) = (1 / α) * fidelity(M, p) + prior(M, p)
    prox_f(M, λ, p) = prox_distance(M, λ / α, data, p, 2)
    prox_f!(M, q, λ, p) = prox_distance!(M, q, λ / α, data, p, 2)
    prox_g_dual(N, n, λ, X) = project_collaborative_TV(N, λ, n, X, Inf, Inf, 1.0)
    prox_g_dual!(N, η, n, λ, X) = project_collaborative_TV(N, η, λ, n, X, Inf, Inf, 1.0)
    DΛ(M, m, X) = differential_forward_logs(M, m, X)
    DΛ!(M, Y, m, X) = differential_forward_logs!(M, Y, m, X)
    adjoint_DΛ(N, m, n, X) = adjoint_differential_forward_logs(M, m, X)
    adjoint_DΛ!(N, Y, m, n, X) = adjoint_differential_forward_logs!(M, Y, m, X)

    function differential_project_collaborative_TV(
        N::PowerManifold, p, ξ, η, p1=2.0, p2=1.0
    )
        ζ = zero_vector(N, p)
        return differential_project_collaborative_TV!(N, ζ, p, ξ, η, p1, p2)
    end
    function differential_project_collaborative_TV!(
        N::PowerManifold, ζ, p, ξ, η, p1=2.0, p2=1.0
    )
        ζ = zero_vector!(N, ζ, p)
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

        if p2 == Inf
            if p1 == Inf || d == 1
                norms = norm.(Ref(N.manifold), p, ξ)

                for i in R # iterate over all pixel
                    for k in 1:d # for all direction combinations
                        I = i.I # array of index
                        J = I .+ e_k_vals[k] #i + e_k is j
                        if all(J .<= maxInd)
                            # this is neighbor in range,
                            ζ[N, I..., k] += if norms[I..., k] <= 1
                                η[N, I..., k]
                            else
                                1 / norms[I..., k] * (
                                    η[N, I..., k] .-
                                    1 / norms[I..., k]^2 .* inner(
                                        N.manifold,
                                        p[N, I..., k],
                                        η[N, I..., k],
                                        ξ[N, I..., k],
                                    ) .* ξ[N, I..., k]
                                )
                            end
                        else
                            ζ[N, I..., k] = zero_vector(N.manifold, p[N, I..., k])
                        end
                    end # directions
                end # i in R
                return ζ
            elseif p1 == 2
                norms = norm.(Ref(N.manifold), p, ξ)
                norms_ = sqrt.(sum(norms .^ 2; dims=length(pdims)))

                for i in R # iterate over all pixel
                    for k in 1:d # for all direction combinations
                        I = i.I # array of index
                        J = I .+ e_k_vals[k] #i + e_k is j
                        if all(J .<= maxInd)
                            # this is neighbor in range,
                            if norms_[I...] <= 1
                                ζ[N, I..., k] += η[N, I..., k]
                            else
                                for κ in 1:d
                                    ζ[N, I..., κ] += if k != κ
                                        -1 / norms_[I...]^3 * inner(
                                            N.manifold,
                                            p[N, I..., k],
                                            η[N, I..., k],
                                            ξ[N, I..., k],
                                        ) .* ξ[N, I..., κ]
                                    else
                                        1 / norms_[I...] * (
                                            η[N, I..., k] .-
                                            1 / norms_[I...]^2 .* inner(
                                                N.manifold,
                                                p[N, I..., k],
                                                η[N, I..., k],
                                                ξ[N, I..., k],
                                            ) .* ξ[N, I..., k]
                                        )
                                    end
                                end
                            end
                        else
                            ζ[N, I..., k] = zero_vector(N.manifold, p[N, I..., k])
                        end
                    end # directions
                end # i in R
                return ζ
            else
                throw(ErrorException("The case p=$p1, q=$p2 is not yet implemented"))
            end
        end # end q
        throw(ErrorException("The case p=$p1, q=$p2 is not yet implemented"))
    end

    function Dprox_F(M, λ, p, X)
        return Manopt.differential_shortest_geodesic_startpoint(M, p, data, λ / (α + λ), X)
    end
    function Dprox_F!(M, Y, λ, p, X)
        Manopt.differential_shortest_geodesic_startpoint!(M, Y, p, data, λ / (α + λ), X)
        return Y
    end
    function Dprox_G_dual(N, n, λ, X, Y)
        return differential_project_collaborative_TV(N, n, X, Y, Inf, Inf)
    end
    function Dprox_G_dual!(N, Z, n, λ, X, Y)
        return differential_project_collaborative_TV!(N, Z, n, X, Y, Inf, Inf)
    end

    m = fill(mid_point(pixelM, data[1], data[2]), 2)
    n = m
    p0 = deepcopy(data)
    ξ0 = zero_vector(M, m)
    X = log(M, p0, m)# TODO construct tangent vector
    Ξ = X

    @testset "test Mutating/Allocation Problem Variants" begin
        obj1 = PrimalDualManifoldSemismoothNewtonObjective(
            f, prox_f, Dprox_F, prox_g_dual, Dprox_G_dual, DΛ, adjoint_DΛ
        )
        p1 = TwoManifoldProblem(M, N, obj1)
        obj2 = PrimalDualManifoldSemismoothNewtonObjective(
            f,
            prox_f!,
            Dprox_F!,
            prox_g_dual!,
            Dprox_G_dual!,
            DΛ!,
            adjoint_DΛ!;
            evaluation=InplaceEvaluation(),
        )
        p2 = TwoManifoldProblem(M, N, obj2)
        x1 = get_differential_primal_prox(p1, 1.0, p0, X)
        x2 = get_differential_primal_prox(p2, 1.0, p0, X)
        @test x1 == x2
        get_differential_primal_prox!(p1, x1, 1.0, p0, X)
        get_differential_primal_prox!(p2, x2, 1.0, p0, X)
        @test x1 == x2

        ξ1 = get_differential_dual_prox(p1, n, 1.0, ξ0, Ξ)
        ξ2 = get_differential_dual_prox(p2, n, 1.0, ξ0, Ξ)
        @test ξ1 ≈ ξ2 atol = 2 * 1e-16
        get_differential_dual_prox!(p1, ξ1, n, 1.0, ξ0, Ξ)
        get_differential_dual_prox!(p2, ξ2, n, 1.0, ξ0, Ξ)
        @test ξ1 ≈ ξ2 atol = 2 * 1e-16
    end
end
