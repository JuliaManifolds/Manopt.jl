using Manifolds, Manopt, Random, Test

@testset "Constained set objective" begin
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
    g(M, p) = distance(M, c, p)^2 - r^2
    indicator_C(M, p) = (g(M, p) ≤ 0) ? 0 : Inf

    csoa = ManifoldConstrainedSetObjective(f, grad_f, project_C)
    csoa2 = ManifoldConstrainedSetObjective(f, grad_f, project_C; indicator = indicator_C)
    csoi = ManifoldConstrainedSetObjective(
        f, grad_f!, project_C!; evaluation = InplaceEvaluation()
    )
    csoi2 = ManifoldConstrainedSetObjective(
        f, grad_f!, project_C!; evaluation = InplaceEvaluation(), indicator = indicator_C
    )

    for objective in [csoa, csoa2, csoi, csoi2]
        @test get_cost(M, objective, c) == f(M, c)
        @test Manopt.get_cost_function(objective)(M, c) == f(M, c)
        @test get_gradient(M, objective, c) == grad_f(M, c)
        X = zero_vector(M, c)
        get_gradient!(M, X, objective, c)
        Y = zero_vector(M, c)
        grad_f!(M, Y, c)
        @test X == Y
        if objective ∈ [csoa, csoa2]
            @test Manopt.get_gradient_function(objective)(M, c) == grad_f(M, c)
        else
            Manopt.get_gradient_function(objective)(M, X, c) == grad_f!(M, Y, c)
            @test X == Y
        end
        dmp = DefaultManoptProblem(M, objective)
        p = get_projected_point(dmp, c)
        @test p == c # c is already in C
        get_projected_point!(dmp, p, c)
        @test p == c
        p = get_projected_point(M, objective, c)
        @test p == c
        get_projected_point!(M, p, objective, c)
        @test p == c
    end
end
