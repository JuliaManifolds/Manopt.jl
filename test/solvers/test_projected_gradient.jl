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
            M, c,
            get_vector(
                M, c, σ .* randn(manifold_dimension(M)) .+ [2.5, 2.5],
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
    @testset "A manifold with numbers as points" begin
        Mc = Circle()
        fc(N, q) = (q - 0.3)^2
        grad_fc(N, q) = 2 * (q - 0.3)
        proj_c(N, q) = clamp(q, -0.2, 0.2)
        qc = projected_gradient_method(Mc, fc, grad_fc, proj_c, 0.0)
        @test qc isa Float64
    end
    mean_pg_1 = projected_gradient_method(
        M, f, grad_f, project_C, c;
        stopping_criterion = StopAfterIteration(150) | StopWhenProjectedGradientStationary(M, 1.0e-7),
    )
    Random.seed!(42)
    mean_pg_2 = projected_gradient_method(
        M, f, grad_f, project_C;
        stopping_criterion = StopAfterIteration(150) | StopWhenProjectedGradientStationary(M, 1.0e-7),
    )
    @test isapprox(M, mean_pg_1, mean_pg_2)
    # the exported step size constructors are accepted for both step sizes
    sc_f = StopAfterIteration(20)
    mean_pg_f = projected_gradient_method(
        M, f, grad_f, project_C, c; stopping_criterion = sc_f,
        stepsize = ConstantLength(1.0), backtrack = ArmijoLinesearch(; stop_increasing_at_step = 0),
    )
    mean_pg_s = projected_gradient_method(
        M, f, grad_f, project_C, c; stopping_criterion = StopAfterIteration(20),
        stepsize = Manopt.ConstantStepsize(M, 1.0),
        backtrack = Manopt.ArmijoLinesearchStepsize(M; stop_increasing_at_step = 0),
    )
    @test mean_pg_f == mean_pg_s
    # a decorated objective reaches the projection as well
    mean_pg_c = projected_gradient_method(
        M, f, grad_f, project_C, c;
        stopping_criterion = StopAfterIteration(150) | StopWhenProjectedGradientStationary(M, 1.0e-7),
        count = [:Cost],
    )
    @test isapprox(M, mean_pg_c, mean_pg_1)
    # the result has to be feasible, that is inside the ball of radius `r` around `c`
    @test distance(M, c, mean_pg_1) <= r + 1.0e-12
    mean_pg_3 = copy(M, c)
    st = projected_gradient_method!(
        M, f, grad_f!, project_C!, mean_pg_3;
        evaluation = InplaceEvaluation(),
        stopping_criterion = StopAfterIteration(150) | StopWhenProjectedGradientStationary(M, 1.0e-7),
        return_state = true,
    )
    @test isapprox(M, mean_pg_1, mean_pg_3)
    @test startswith(
        Manopt.status_summary(st; context = :default),
        "# Solver state for `Manopt.jl`s Projected Gradient Method\n"
    )
    @test startswith(repr(st), "ProjectedGradientMethodState(; ")
    # the default backtracking of the state keeps the step at most one, as the solver does
    @test ProjectedGradientMethodState(M).backtrack.stop_increasing_at_step == 0
    stop_when_stationary = st.stop.criteria[2]
    @test Manopt.indicates_convergence(stop_when_stationary)
    @test repr(stop_when_stationary) == "StopWhenProjectedGradientStationary($(stop_when_stationary.threshold))"
    @test length(get_reason(stop_when_stationary)) > 0
    @test length(get_reason(StopWhenProjectedGradientStationary(M, 1.0e-7))) == 0
end
