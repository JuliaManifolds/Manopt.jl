#
#
#
using Random, Manifolds, Manopt, Test
Random.seed!(29)
@testset "Test Nelder-Mead" begin
    @testset "Euclidean" begin
        M = Euclidean(6)
        # From Wikipedia https://en.wikipedia.org/wiki/Rosenbrock_function
        function Rosenbrock(::Euclidean, x)
            return sum([
                100 * (x[2 * i - 1]^2 - x[2 * i])^2 + (x[2 * i - 1] - 1)^2 for
                i in 1:div(length(x), 2)
            ])
        end
        for initial_simplex in [
            NelderMeadSimplex([8 * randn(6) for i in 1:7]),
            NelderMeadSimplex(M, zeros(6)),
            NelderMeadSimplex(M),
        ]
            rst = NelderMead(
                M, Rosenbrock, initial_simplex; record=[RecordCost()], return_state=true
            )
            x = get_solver_result(rst)
            rec = get_record(rst)
            nonincreasing = [rec[i] >= rec[i + 1] for i in 1:(length(rec) - 1)]
            @test any(map(!, nonincreasing)) == false

            x2 = NelderMead(M, Rosenbrock, initial_simplex)
            @test x == x2

            set_iterate!(rst, M, ones(6))
            @test get_iterate(rst) == ones(6)
        end
    end

    @testset "Rotations" begin
        M = Rotations(3)
        A = randn(3, 3)
        A .= (A - A') ./ 2
        f(::Rotations, x) = norm(A * x * x * A)
        x0 = NelderMeadSimplex([rand(M) for _ in 1:12])
        o = NelderMead(
            M,
            f,
            x0;
            record=[RecordCost()],
            return_state=true,
            stopping_criterion=StopAfterIteration(400),
        )
        @test startswith(repr(o), "# Solver state for `Manopt.jl`s Nelder Mead Algorithm")
        x = get_solver_result(o)
        rec = get_record(o)
        nonincreasing = [rec[i] >= rec[i + 1] for i in 1:(length(rec) - 1)]
        @test any(map(!, nonincreasing)) == false
        f = StopWhenPopulationConcentrated(1e-1, 1e-2)
        sf = "StopWhenPopulationConcentrated($(1e-1), $(1e-2))\n    $(Manopt.status_summary(f))"
        @test repr(f) == sf
    end

    @testset "Circle" begin
        M = Circle()
        data = [-π / 2, π / 4, 0.0, π / 4]
        p_star = sum(data) / length(data)
        f(M, p) = 1 / 10 * sum(distance.(Ref(M), data, Ref(p)) .^ 2)
        #vector p-cost
        f2(M, p) = 1 / 10 * sum(distance.(Ref(M), data, Ref(p[])) .^ 2)
        q = NelderMead(M, f)
        @test isapprox(p_star, q; atol=1e-7)
        s = NelderMead(M, f; return_state=true)
        q2 = get_solver_result(s)[] #here we have to floatify ouselves
        @test isapprox(M, p_star, q2; atol=1e-7)
        population = NelderMeadSimplex(M)
        q3 = NelderMead(M, f, population)
        #same type also returns Float
        @test isapprox(M, p_star, q3; atol=1e-7)
        population2 = NelderMeadSimplex([[0.1], [-0.1]])
        q4 = NelderMead(M, f2, population2)
        @test isapprox(M, p_star, q4[]; atol=1e-7)
    end
end
