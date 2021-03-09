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
        x0 = [8 * randn(6) for i in 1:7]
        o = NelderMead(M, Rosenbrock, x0; record=[RecordCost()], return_options=true)
        x = get_solver_result(o)
        rec = get_record(o)
        nonincreasing = [rec[i] >= rec[i + 1] for i in 1:(length(rec) - 1)]
        @test any(map(!, nonincreasing)) == false

        x2 = NelderMead(M, Rosenbrock, x0)
        @test x == x2
    end
    @testset "Rotations" begin
        M = Rotations(3)
        A = randn(3, 3)
        A .= (A - A') ./ 2
        f(::Rotations, x) = norm(A * x * x * A)
        x0 = [random_point(M) for _ in 1:12]
        o = NelderMead(
            M,
            f,
            x0;
            record=[RecordCost()],
            return_options=true,
            retraction_method=QRRetraction(),
            inverse_retraction_method=QRInverseRetraction(),
            stopping_criterion=StopAfterIteration(500),
        )
        x = get_solver_result(o)
        rec = get_record(o)
        nonincreasing = [rec[i] >= rec[i + 1] for i in 1:(length(rec) - 1)]
        @test any(map(!, nonincreasing)) == false
    end
end
