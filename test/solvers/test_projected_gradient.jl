using Manifolds, Manopt, Random, Test

@testset "Test the projected gradient method" begin
    M = Hyperbolic(2)
    c = Manifolds._hyperbolize(M, [0, 0])
    r = 1.0
    N = 200
    σ = 1.5
    Random.seed!(42)
    # N random points moved to top left to have a mean outside
    pts = [
        exp(
                M,
                c,
                get_vector(
                    M,
                    c,
                    σ .* randn(manifold_dimension(M)) .+ [2.5, 2.5],
                    DefaultOrthonormalBasis(),
                ),
            ) for _ in 1:N
    ]
    f(M, p) = 1 / (2 * length(pts)) .* sum(distance(M, p, q)^2 for q in pts)
    grad_f(M, p) = -1 / length(pts) .* sum(log(M, p, q) for q in pts)
    function grad_f!(M, X, p)
        zero_vector!(M, X, p)
        Y = zero_vector(M, p)
        for q in pts
            log!(M, Y, p, q)
            X .+= Y
        end
        X .*= -1 / length(pts)
        return X
    end
    function project_C(M, p)
        X = log(M, c, p)
        n = norm(M, c, X)
        q = (n > r) ? exp(M, c, (r / n) * X) : copy(M, p)
        return q
    end
    function project_C!(M, q, p; X = zero_vector(M, c))
        log!(M, X, c, p)
        n = norm(M, c, X)
        if (n > r)
            exp!(M, q, c, (r / n) * X)
        else
            copyto!(M, q, p)
        end
        return q
    end
    mean_pg_1 = projected_gradient_method(
        M,
        f,
        grad_f,
        project_C,
        c;
        stopping_criterion = StopAfterIteration(150) |
            StopWhenProjectedGradientStationary(M, 1.0e-7),
    )
    Random.seed!(42)
    mean_pg_2 = projected_gradient_method(
        M,
        f,
        grad_f,
        project_C;
        stopping_criterion = StopAfterIteration(150) |
            StopWhenProjectedGradientStationary(M, 1.0e-7),
    )
    @test isapprox(M, mean_pg_1, mean_pg_2)
    mean_pg_3 = copy(M, c)
    st = projected_gradient_method!(
        M,
        f,
        grad_f!,
        project_C!,
        mean_pg_3;
        evaluation = InplaceEvaluation(),
        stopping_criterion = StopAfterIteration(150) |
            StopWhenProjectedGradientStationary(M, 1.0e-7),
        return_state = true,
    )
    @test isapprox(M, mean_pg_1, mean_pg_3)
    @test startswith(
        Manopt.status_summary(st; inline = false),
        "# Solver state for `Manopt.jl`s Projected Gradient Method\n"
    )
    stop_when_stationary = st.stop.criteria[2]
    @test Manopt.indicates_convergence(stop_when_stationary)
    @test repr(stop_when_stationary) == "StopWhenProjectedGradientStationary($(stop_when_stationary.threshold))"
    @test length(get_reason(stop_when_stationary)) > 0
    @test length(get_reason(StopWhenProjectedGradientStationary(M, 1.0e-7))) == 0
end
