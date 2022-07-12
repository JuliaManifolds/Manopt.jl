using Manopt, Manifolds, ManifoldsBase, Test

@testset "Test primal dual plan" begin
    #
    # Perform an really easy test, just compute a mid point
    #
    pixelM = Sphere(2)
    signal_section_size = 1
    M = PowerManifold(pixelM, NestedPowerRepresentation(), 2 * signal_section_size)
    data = [[1.0, 0.0, 0.0], 1 / sqrt(2) .* [1.0, 1.0, 0.0]]
    α = 1
    # known minimizer
    δ = min(α / distance(pixelM, data[1], data[2]), 0.5)
    x_hat = shortest_geodesic(M, data, reverse(data), δ)
    N = M
    fidelity(M, x) = 1 / 2 * distance(M, x, f)^2
    Λ(M, x) = ProductRepr(x, forward_logs(M, x))
    prior(M, x) = norm(norm.(Ref(M.manifold), x, submanifold_component(N, Λ(x), 2)), 1)
    cost(M, x) = (1 / α) * fidelity(M, x) + prior(M, x)
    prox_F(M, λ, x) = prox_distance(M, λ / α, data, x, 2)
    prox_F!(M, y, λ, x) = prox_distance!(M, y, λ / α, data, x, 2)
    prox_G_dual(N, n, λ, ξ) = project_collaborative_TV(N, λ, n, ξ, Inf, Inf, 1.0)
    prox_G_dual!(N, η, n, λ, ξ) = project_collaborative_TV(N, η, λ, n, ξ, Inf, Inf, 1.0)
    DΛ(M, m, X) = differential_forward_logs(M, m, X)
    DΛ!(M, Y, m, X) = differential_forward_logs!(M, Y, m, X)
    adjoint_DΛ(N, m, n, ξ) = adjoint_differential_forward_logs(M, m, ξ)
    adjoint_DΛ!(N, Y, m, n, ξ) = adjoint_differential_forward_logs!(M, Y, m, ξ)

    function differential_project_collaborative_TV(N::PowerManifold, x, Ξ, Η, p=2.0, q=1.0)
        Y = zero_vector(N, x)
        return differential_project_collaborative_TV!(N, Y, x, Ξ, Η, p, q)
    end
    function differential_project_collaborative_TV!(
        N::PowerManifold, Y, x, Ξ, Η, p=2.0, q=1.0
    )
        Y = zero_vector(N, x)
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
                            Y[N, I..., k] += if norms[I..., k] <= 1
                                Η[N, I..., k]
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
                norms_ = sqrt.(sum(norms .^ 2; dims=length(pdims)))

                for i in R # iterate over all pixel
                    for k in 1:d # for all direction combinations
                        I = i.I # array of index
                        J = I .+ e_k_vals[k] #i + e_k is j
                        if all(J .<= maxInd)
                            # this is neighbor in range,
                            if norms_[I...] <= 1
                                Y[N, I..., k] += Η[N, I..., k]
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

    Dprox_F(M, λ, x, η) = differential_geodesic_startpoint(M, x, data, λ / (α + λ), η)
    Dprox_F!(M, Y, λ, x, η) = differential_geodesic_startpoint!(M, Y, x, data, λ / (α + λ), η)
    function Dprox_G_dual(N, n, λ, ξ, η)
        return differential_project_collaborative_TV(N, n, ξ, η, Inf, Inf)
    end
    function Dprox_G_dual!(N, Y, n, λ, ξ, η)
        return differential_project_collaborative_TV!(N, Y, n, ξ, η, Inf, Inf)
    end

    m = fill(mid_point(pixelM, data[1], data[2]), 2)
    n = m
    x0 = deepcopy(data)
    ξ0 = zero_vector(M, m)
    X = log(M, x0, m)# TODO construct tangent vector
    Ξ = X 

    @testset "test Mutating/Allocation Problem Variants" begin
        p1 = PrimalDualSemismoothNewtonProblem(
            M, 
            N, 
            cost, 
            prox_F,
            Dprox_F,
            prox_G_dual,
            Dprox_G_dual,
            DΛ,
            adjoint_DΛ
        )
        p2 = PrimalDualSemismoothNewtonProblem(
            M,
            N,
            cost,
            prox_F!,
            Dprox_F!,
            prox_G_dual!,
            Dprox_G_dual!,
            DΛ!,
            adjoint_DΛ!;
            evaluation=MutatingEvaluation(),
        )
        x1 = get_differential_primal_prox(p1, 1.0, x0, X)
        x2 = get_differential_primal_prox(p2, 1.0, x0, X)
        @test x1 == x2
        get_differential_primal_prox!(p1, x1, 0.8, x0, X)
        get_differential_primal_prox!(p2, x2, 0.8, x0, X)
        @test x1 == x2

        ξ1 = get_differential_dual_prox(p1, n, 1.0, ξ0, Ξ)
        ξ2 = get_differential_dual_prox(p2, n, 1.0, ξ0, Ξ)
        @test ξ1 ≈ ξ2 atol = 2 * 1e-16
        get_differential_dual_prox!(p1, ξ1, n, 1.0, ξ0, Ξ)
        get_differential_dual_prox!(p2, ξ2, n, 1.0, ξ0, Ξ)
        @test ξ1 ≈ ξ2 atol = 2 * 1e-16
    end
end
