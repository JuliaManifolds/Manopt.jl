using Manopt, ManifoldsBase, Manifolds, Test
using LinearAlgebra: I, tr

@testset "Riemannian Augmented Lagrangian Method" begin
    @testset "Test RALM with a nonneg. PCA" begin
        d = 20
        M = Sphere(d - 1)
        S = [ones(4)..., zeros(d - 4)...]
        v0 = project(M, S)
        Z = v0 * v0'
        f(M, p) = -tr(transpose(p) * Z * p) / 2
        grad_f(M, p) = project(M, p, -transpose.(Z) * p / 2 - Z * p / 2)
        g(M, p) = -p # i.e. p ≥ 0
        mI = -Matrix{Float64}(I, d, d)
        grad_g(M, p) = [project(M, p, mI[:, i]) for i in 1:d]
        p0 = project(M, ones(d))
        sol = augmented_Lagrangian_method(M, f, grad_f, p0; g=g, grad_g=grad_g)
        @test distance(M, sol, v0) < 8 * 1e-4
        sol2 = copy(M, p0)
        augmented_Lagrangian_method!(M, f, grad_f, sol2; g=g, grad_g=grad_g)
        @test sol2 == sol

        co = ConstrainedManifoldObjective(f, grad_f; g=g, grad_g=grad_g)
        mp = DefaultManoptProblem(M, co)
        # dummy ALM problem
        alms = AugmentedLagrangianMethodState(
            M, co, p0, DefaultManoptProblem(M, ManifoldCostObjective(f)), NelderMeadState(M)
        )
        set_iterate!(alms, M, 2 .* p0)
        @test Manopt.get_message(alms) == ""
        @test get_iterate(alms) == 2 .* p0
        @test startswith(
            repr(alms), "# Solver state for `Manopt.jl`s Augmented Lagrangian Method\n"
        )
    end
    @testset "Numbers" begin
        M = Euclidean()
        f(M, p) = (p + 5)^2
        grad_f(M, p) = 2 * p + 10
        g(M, p) = -p # i.e. p ≥ 0
        grad_g(M, p) = -1
        s = augmented_Lagrangian_method(
            M,
            f,
            grad_f,
            4.0;
            g=g,
            grad_g=grad_g,
            stopping_criterion=StopAfterIteration(20),
            return_state=true,
        )
        q = get_solver_result(s)[]
        @test q isa Real
        @test f(M, q) < f(M, 4)
    end
end
