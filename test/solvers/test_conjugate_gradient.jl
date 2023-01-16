using Manopt, Manifolds, ManifoldsBase, Test
using LinearAlgebra: Diagonal, dot, eigvals, eigvecs

@testset "Conjugate Gradient coefficient rules" begin
    M = Euclidean(2)
    f(M, x) = norm(x)^2
    grad_f(::Euclidean, x) = 2 * x
    dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
    x0 = [0.0, 1.0]
    sC = StopAfterIteration(1)
    s = ConstantStepsize(M)
    retr = ExponentialRetraction()
    vtm = ParallelTransport()

    grad_1 = [1.0, 1.0]
    δ1 = [0.0, 2.0]
    grad_2 = [1.0, 1.5]
    δ2 = [0.5, 2.0]
    diff = grad_2 - grad_1

    dU = SteepestDirectionUpdateRule()
    cgs = ConjugateGradientDescentState(M, x0, sC, s, dU, retr, vtm, zero_vector(M, x0))
    @test cgs.coefficient(dmp, cgs, 1) == 0
    @test default_stepsize(M, typeof(cgs)) isa ConstantStepsize

    dU = ConjugateDescentCoefficient()
    cgs = ConjugateGradientDescentState(M, x0, sC, s, dU, retr, vtm, zero_vector(M, x0))
    cgs.X = grad_1
    cgs.δ = δ1
    # for the first case we get zero
    @test cgs.coefficient(dmp, cgs, 1) == 0.0
    cgs.X = grad_2
    cgs.δ = δ2
    @test cgs.coefficient(dmp, cgs, 2) == dot(grad_2, grad_2) / dot(-δ2, grad_1)

    dU = DaiYuanCoefficient()
    cgs = ConjugateGradientDescentState(M, x0, sC, s, dU, retr, vtm)
    cgs.X = grad_1
    cgs.δ = δ1
    # for the first case we get zero
    @test cgs.coefficient(dmp, cgs, 1) == 0.0
    cgs.X = grad_2
    cgs.δ = δ2
    @test cgs.coefficient(dmp, cgs, 2) == dot(grad_2, grad_2) / dot(δ2, grad_2 - grad_1)

    dU = FletcherReevesCoefficient()
    cgs = ConjugateGradientDescentState(M, x0, sC, s, dU, retr, vtm)
    cgs.X = grad_1
    cgs.δ = δ1
    # for the first case we get zero
    @test cgs.coefficient(dmp, cgs, 1) == 1.0
    cgs.X = grad_2
    cgs.δ = δ2
    @test cgs.coefficient(dmp, cgs, 2) == dot(grad_2, grad_2) / dot(grad_1, grad_1)

    dU = HagerZhangCoefficient()
    cgs = ConjugateGradientDescentState(M, x0, sC, s, dU, retr, vtm)
    cgs.X = grad_1
    cgs.δ = δ1
    # for the first case we get zero
    @test cgs.coefficient(dmp, cgs, 1) == 0.0
    cgs.X = grad_2
    cgs.δ = δ2
    denom = dot(δ1, diff)
    ndiffsq = dot(diff, diff)
    @test cgs.coefficient(dmp, cgs, 2) ==
        dot(diff, grad_2) / denom - 2 * ndiffsq * dot(δ1, grad_2) / denom^2

    dU = HeestenesStiefelCoefficient()
    O = ConjugateGradientDescentState(M, x0, sC, s, dU, retr, vtm)
    O.X = grad_1
    O.δ = δ1
    @test O.coefficient(dmp, O, 1) == 0.0
    O.X = grad_2
    O.δ = δ2
    @test O.coefficient(dmp, O, 2) == dot(diff, grad_2) / dot(δ1, diff)

    dU = LiuStoreyCoefficient()
    O = ConjugateGradientDescentState(M, x0, sC, s, dU, retr, vtm)
    O.X = grad_1
    O.δ = δ1
    @test O.coefficient(dmp, O, 1) == 0.0
    O.X = grad_2
    O.δ = δ2
    @test O.coefficient(dmp, O, 2) == -dot(grad_2, diff) / dot(δ1, grad_1)

    dU = PolakRibiereCoefficient()
    O = ConjugateGradientDescentState(M, x0, sC, s, dU, retr, vtm)
    O.X = grad_1
    O.δ = δ1
    @test O.coefficient(dmp, O, 1) == 0.0
    O.X = grad_2
    O.δ = δ2
    @test O.coefficient(dmp, O, 2) == dot(grad_2, diff) / dot(grad_1, grad_1)
end
@testset "Conjugate Gradient runs – Low Rank matrix approx" begin
    A = Diagonal([2.0, 1.1, 1.0])
    M = Sphere(size(A, 1) - 1)
    f(::Sphere, p) = p' * A * p
    grad_f(M, p) = project(M, p, 2 * A * p) # project the Euclidean gradient

    p0 = [2.0, 0.0, 2.0] / sqrt(8.0)
    x_opt = conjugate_gradient_descent(
        M,
        f,
        grad_f,
        p0;
        stepsize=ArmijoLinesearch(M),
        coefficient=FletcherReevesCoefficient(),
        stopping_criterion=StopAfterIteration(15),
    )
    @test isapprox(f(M, x_opt), minimum(eigvals(A)); atol=2.0 * 1e-2)
    @test isapprox(x_opt, eigvecs(A)[:, size(A, 1)]; atol=3.0 * 1e-1)
    x_opt2 = conjugate_gradient_descent(
        M,
        f,
        grad_f,
        p0;
        stepsize=ArmijoLinesearch(M),
        coefficient=FletcherReevesCoefficient(),
        stopping_criterion=StopAfterIteration(15),
        return_state=true,
    )
    @test get_solver_result(x_opt2) == x_opt
end

@testset "CG on complex manifolds" begin
    M = Euclidean(2; field=ℂ)
    A = [1 im; -im 1]
    f(::Euclidean, p) = real(p' * A * p)
    grad_f(::Euclidean, p) = 2 * A * p
    p0 = [2.0, 1+im]
    x_opt = conjugate_gradient_descent(
        M,
        f,
        grad_f,
        p0;
        stepsize=ArmijoLinesearch(M),
        coefficient=FletcherReevesCoefficient(),
        stopping_criterion=StopAfterIteration(15),
    )
end
