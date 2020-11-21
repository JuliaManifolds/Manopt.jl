using Manopt, Manifolds, ManifoldsBase, LinearAlgebra, Test

@testset "Conjugate Gradient coefficient rules" begin
    F(x) = norm(x)^2
    ∇F(x) = 2 * x
    M = Euclidean(2)
    P = GradientProblem(M, F, ∇F)
    x0 = [0.0, 1.0]
    sC = StopAfterIteration(1)
    s = ConstantStepsize(1.0)
    retr = ExponentialRetraction()
    vtm = ParallelTransport()

    ∇1 = [1.0, 1.0]
    δ1 = [0.0, 2.0]
    ∇2 = [1.0, 1.5]
    δ2 = [0.5, 2.0]
    diff = ∇2 - ∇1

    dU = SteepestDirectionUpdateRule()
    O = ConjugateGradientDescentOptions(x0, sC, s, dU, retr, vtm)
    @test O.coefficient(P, O, 1) == 0

    dU = ConjugateDescentCoefficient()
    O = ConjugateGradientDescentOptions(x0, sC, s, dU, retr, vtm)
    O.∇ = ∇1
    O.δ = δ1
    # for the first case we get zero
    @test O.coefficient(P, O, 1) == 0.0
    O.∇ = ∇2
    O.δ = δ2
    @test O.coefficient(P, O, 2) == dot(∇2, ∇2) / dot(-δ2, ∇1)

    dU = DaiYuanCoefficient()
    O = ConjugateGradientDescentOptions(x0, sC, s, dU, retr, vtm)
    O.∇ = ∇1
    O.δ = δ1
    # for the first case we get zero
    @test O.coefficient(P, O, 1) == 0.0
    O.∇ = ∇2
    O.δ = δ2
    @test O.coefficient(P, O, 2) == dot(∇2, ∇2) / dot(δ2, ∇2 - ∇1)

    dU = FletcherReevesCoefficient()
    O = ConjugateGradientDescentOptions(x0, sC, s, dU, retr, vtm)
    O.∇ = ∇1
    O.δ = δ1
    # for the first case we get zero
    @test O.coefficient(P, O, 1) == 1.0
    O.∇ = ∇2
    O.δ = δ2
    @test O.coefficient(P, O, 2) == dot(∇2, ∇2) / dot(∇1, ∇1)

    dU = HagerZhangCoefficient()
    O = ConjugateGradientDescentOptions(x0, sC, s, dU, retr, vtm)
    O.∇ = ∇1
    O.δ = δ1
    # for the first case we get zero
    @test O.coefficient(P, O, 1) == 0.0
    O.∇ = ∇2
    O.δ = δ2
    denom = dot(δ1, diff)
    ndiffsq = dot(diff, diff)
    @test O.coefficient(P, O, 2) ==
          dot(diff, ∇2) / denom - 2 * ndiffsq * dot(δ1, ∇2) / denom^2

    dU = HeestenesStiefelCoefficient()
    O = ConjugateGradientDescentOptions(x0, sC, s, dU, retr, vtm)
    O.∇ = ∇1
    O.δ = δ1
    @test O.coefficient(P, O, 1) == 0.0
    O.∇ = ∇2
    O.δ = δ2
    @test O.coefficient(P, O, 2) == dot(diff, ∇2) / dot(δ1, diff)

    dU = LiuStoreyCoefficient()
    O = ConjugateGradientDescentOptions(x0, sC, s, dU, retr, vtm)
    O.∇ = ∇1
    O.δ = δ1
    @test O.coefficient(P, O, 1) == 0.0
    O.∇ = ∇2
    O.δ = δ2
    @test O.coefficient(P, O, 2) == -dot(∇2, diff) / dot(δ1, ∇1)

    dU = PolakRibiereCoefficient()
    O = ConjugateGradientDescentOptions(x0, sC, s, dU, retr, vtm)
    O.∇ = ∇1
    O.δ = δ1
    @test O.coefficient(P, O, 1) == 0.0
    O.∇ = ∇2
    O.δ = δ2
    @test O.coefficient(P, O, 2) == dot(∇2, diff) / dot(∇1, ∇1)
end
@testset "Conjugate Gradient runs – Low Rank matrix approx" begin
    A = Diagonal([2.0, 1.1, 1.0])
    M = Sphere(size(A, 1) - 1)
    F(x) = x' * A * x
    euclidean_∇F(x) = 2 * A * x
    ∇F(x) = project(M, x, euclidean_∇F(x))

    x0 = [2.0, 0.0, 2.0] / sqrt(8.0)
    xOpt = conjugate_gradient_descent(
        M,
        F,
        ∇F,
        x0;
        stepsize=ArmijoLinesearch(),
        coefficient=FletcherReevesCoefficient(),
        stopping_criterion=StopAfterIteration(15),
    )
    @test isapprox(F(xOpt), minimum(eigvals(A)); atol=2.0 * 10^-4)
    @test isapprox(xOpt, eigvecs(A)[:, size(A, 1)]; atol=3.0 * 10^-2)
    xOpt2 = conjugate_gradient_descent(
        M,
        F,
        ∇F,
        x0;
        stepsize=ArmijoLinesearch(),
        coefficient=FletcherReevesCoefficient(),
        stopping_criterion=StopAfterIteration(15),
        return_options = true,
    )
    @test get_solver_result(xOpt2) == xOpt
end
