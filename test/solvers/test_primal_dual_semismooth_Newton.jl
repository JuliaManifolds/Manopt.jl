using Manopt, Manifolds, ManifoldsBase, Test
using ManoptExamples: adjoint_differential_forward_logs
using ManifoldDiff: differential_shortest_geodesic_startpoint

@testset "PD-RSSN" begin
    #
    # Perform an really easy test, just compute a mid point
    #
    pixelM = Sphere(2)
    signal_section_size = 1
    M = PowerManifold(pixelM, NestedPowerRepresentation(), 2 * signal_section_size)
    p1 = [1.0, 0.0, 0.0]
    p2 = 1 / sqrt(2) .* [1.0, 1.0, 0.0]
    data = vcat(fill(p1, signal_section_size), fill(p2, signal_section_size))
    α = 1.0
    σ = 0.5
    τ = 0.5
    # known minimizer
    δ = min(α / distance(pixelM, data[1], data[end]), 0.5)
    x_hat = shortest_geodesic(M, data, reverse(data), δ)
    N = M
    fidelity(M, x) = 1 / 2 * distance(M, x, f)^2
    Λ(M, x) = ArrayPartition(x, forward_logs(M, x))
    prior(M, x) = norm(norm.(Ref(M.manifold), x, submanifold_component(N, Λ(x), 2)), 1)
    f(M, x) = (1 / α) * fidelity(M, x) + prior(M, x)
    prox_f(M, λ, x) = prox_distance(M, λ / α, data, x, 2)
    prox_g_dual(N, n, λ, ξ) = project_collaborative_TV(N, λ, n, ξ, Inf, Inf, 1.0) # non-isotropic
    DΛ(M, m, X) = differential_forward_logs(M, m, X)
    adjoint_DΛ(N, m, n, ξ) = adjoint_differential_forward_logs(M, m, ξ)

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
                norms_ = sqrt.(sum(norms .^ 2; dims=length(pdims)))

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

    function Dprox_F(M, λ, x, η)
        return Manopt.differential_shortest_geodesic_startpoint(M, x, data, λ / (α + λ), η)
    end
    function Dprox_G_dual(N, n, λ, ξ, η)
        return differential_project_collaborative_TV(N, λ, n, ξ, η, Inf, Inf)
    end

    m = fill(mid_point(pixelM, p1, p2), 2 * signal_section_size)
    n = m
    x0 = deepcopy(data)
    ξ0 = zero_vector(M, m)

    s = primal_dual_semismooth_Newton(
        M,
        N,
        f,
        x0,
        ξ0,
        m,
        n,
        prox_f,
        Dprox_F,
        prox_g_dual,
        Dprox_G_dual,
        DΛ,
        adjoint_DΛ;
        primal_stepsize=σ,
        dual_stepsize=τ,
        return_state=true,
    )
    @test startswith(
        repr(s), "# Solver state for `Manopt.jl`s primal dual semismooth Newton"
    )
    y = get_solver_result(s)
    @test x_hat ≈ y atol = 2 * 1e-7

    update_dual_base(p, o, i) = o.n
    o2 = primal_dual_semismooth_Newton(
        M,
        N,
        f,
        x0,
        ξ0,
        m,
        n,
        prox_f,
        Dprox_F,
        prox_g_dual,
        Dprox_G_dual,
        DΛ,
        adjoint_DΛ;
        primal_stepsize=σ,
        dual_stepsize=τ,
        update_dual_base=update_dual_base,
        return_state=false,
    )
    y2 = o2
    @test x_hat ≈ y2 atol = 2 * 1e-7
    @testset "Objective Decorator passthrough" begin
        # PDNSSN additionals
        pdmsno = PrimalDualManifoldSemismoothNewtonObjective(
            f, prox_f, Dprox_F, prox_g_dual, Dprox_G_dual, DΛ, adjoint_DΛ;
        )
        ro = DummyDecoratedObjective(pdmsno)
        X = zero_vector(M, x0)
        Y = get_differential_primal_prox(M, pdmsno, 0.1, x0, X)
        Y2 = get_differential_primal_prox(M, ro, 0.1, x0, X)
        @test Y == Y2
        get_differential_primal_prox!(M, Y, pdmsno, 0.1, x0, X)
        get_differential_primal_prox!(M, Y2, ro, 0.1, x0, X)
        @test Y == Y2

        X = zero_vector(N, ξ0)
        Y = get_differential_dual_prox(N, pdmsno, n, 0.1, ξ0, X)
        Y2 = get_differential_dual_prox(N, ro, n, 0.1, ξ0, X)
        @test Y == Y2
        get_differential_dual_prox!(N, Y, pdmsno, n, 0.1, ξ0, X)
        get_differential_dual_prox!(N, Y2, ro, n, 0.1, ξ0, X)
        @test Y == Y2
    end
end
