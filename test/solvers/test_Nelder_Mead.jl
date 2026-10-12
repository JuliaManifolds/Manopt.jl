#
#
#
using Manifolds, Manopt, Random, Test
Random.seed!(29)

@testset "Test Nelder-Mead" begin
    @testset "Euclidean" begin
        M = Euclidean(6)
        # From Wikipedia https://en.wikipedia.org/wiki/Rosenbrock_function
        function Rosenbrock(::Euclidean, x)
            return sum(
                [
                    100 * (x[2 * i - 1]^2 - x[2 * i])^2 + (x[2 * i - 1] - 1)^2 for
                        i in 1:div(length(x), 2)
                ]
            )
        end
        for initial_simplex in [
                NelderMeadSimplex([8 * randn(6) for i in 1:7]),
                NelderMeadSimplex(M, zeros(6)),
                NelderMeadSimplex(M),
            ]
            rst = NelderMead(
                M, Rosenbrock, initial_simplex; record = [RecordCost()], return_state = true
            )
            x = get_solver_result(rst)
            rec = get_record(rst)
            # initialization returns the state
            s_init = NelderMeadState(M)
            @test Manopt.initialize_solver!(DefaultManoptProblem(M, ManifoldCostObjective(Rosenbrock)), s_init) === s_init
            nonincreasing = [rec[i] >= rec[i + 1] for i in 1:(length(rec) - 1)]
            @test any(map(!, nonincreasing)) == false

            x2 = NelderMead(M, Rosenbrock, initial_simplex)
            @test x == x2

            set_iterate!(rst, M, ones(6))
            @test get_iterate(rst) == ones(6)
        end
        # `return_objective=true` was accepted but silently dropped before
        o1, q1 = NelderMead(M, Rosenbrock; return_objective = true)
        @test o1 isa ManifoldCostObjective
        @test q1 isa Vector
    end

    @testset "Rotations" begin
        M = Rotations(3)
        A = randn(3, 3)
        A .= (A - A') ./ 2
        f(::Rotations, x) = norm(A * x * x * A)
        Random.seed!(23)
        p0 = NelderMeadSimplex([rand(M) for _ in 1:12])
        s = NelderMead(
            M,
            f,
            p0;
            record = [RecordCost()],
            return_state = true,
            stopping_criterion = StopAfterIteration(400),
        )
        @test startswith(Manopt.status_summary(s; context = :default), "# Solver state for `Manopt.jl`s Nelder Mead Algorithm")
        p1 = get_solver_result(s)
        rec = get_record(s)
        nonincreasing = [rec[i] >= rec[i + 1] for i in 1:(length(rec) - 1)]
        @test any(map(!, nonincreasing)) == false
        #mutate
        p2 = NelderMeadSimplex(copy.(Ref(M), p0.pts))
        p3 = NelderMead!( #work in place of p2 but the best is the point p3
            M,
            f,
            p2;
            stopping_criterion = StopAfterIteration(400),
        )
        @test isapprox(M, p1, p3)
        # SC
        sc = StopWhenPopulationConcentrated(1.0e-1, 1.0e-2)
        sf = "StopWhenPopulationConcentrated($(1.0e-1), $(1.0e-2))"
        @test Manopt.status_summary(sc; context = :short) == sf
    end

    @testset "Circle" begin
        M = Circle()
        data = [-π / 2, π / 4, 0.0, π / 4]
        p_star = sum(data) / length(data)
        # a simplex from numbers remembers the number type and stores the points wrapped
        @test NelderMeadSimplex(Circle(), 0.0) isa NelderMeadSimplex{Float64}
        @test NelderMeadSimplex(Circle(), 0.0).pts isa Vector{Array{Float64, 0}}
        f(M, p) = 1 / 10 * sum(distance.(Ref(M), data, Ref(p)) .^ 2)
        @test isapprox(M, NelderMead(M, f, NelderMeadSimplex(M, 0.0)), p_star; atol = 1.0e-7)
        # with an objective built for numbers and in place, the simplex holds the minimizer afterwards
        @test isapprox(M, NelderMead(M, ManifoldCostObjective(f; p = 0.0), NelderMeadSimplex([-3.0, -2.75]))[], p_star; atol = 1.0e-7)
        population0 = NelderMeadSimplex([-3.0, -2.75])
        q0 = NelderMead!(M, f, population0)
        @test isapprox(M, q0, p_star; atol = 1.0e-7)
        @test population0.pts[argmin(f.(Ref(M), getindex.(population0.pts)))][] == q0
        # a function written for wrapped points with a simplex of wrapped points is not wrapped again
        f0(M, p) = f(M, p[])
        population1 = NelderMeadSimplex([fill(-3.0), fill(-2.75)])
        @test population1 isa NelderMeadSimplex{Array{Float64, 0}}
        q1 = NelderMead(M, f0, population1)
        @test q1 isa Array{Float64, 0}
        @test isapprox(M, q1[], p_star; atol = 1.0e-7)
        q1i = NelderMead!(M, f0, population1)
        @test isapprox(M, q1i[], p_star; atol = 1.0e-7)
        @test population1.pts[argmin(f0.(Ref(M), population1.pts))] == q1i
        #vector p-cost
        f2(M, p) = 1 / 10 * sum(distance.(Ref(M), data, Ref(p[])) .^ 2)
        q = NelderMead(M, f)
        @test isapprox(p_star, q; atol = 1.0e-7)
        s = NelderMead(M, f; return_state = true)
        q2 = get_solver_result(s)[] # here: return back to float
        @test isapprox(M, p_star, q2; atol = 1.0e-7)
        population = NelderMeadSimplex(M)
        q3 = NelderMead(M, f, population)
        #same type also returns Float
        @test isapprox(M, p_star, q3; atol = 1.0e-7)
        population2 = NelderMeadSimplex([[0.1], [-0.1]])
        q4 = NelderMead(M, f2, population2)
        @test isapprox(M, p_star, q4[]; atol = 1.0e-7)
        # `return_objective=true` is honoured and still unwraps the scalar point
        o5, q5 = NelderMead(M, f; return_objective = true)
        @test o5 isa ManifoldCostObjective
        @test q5 isa Number
        @test isapprox(M, p_star, q5; atol = 1.0e-7)
    end
end
