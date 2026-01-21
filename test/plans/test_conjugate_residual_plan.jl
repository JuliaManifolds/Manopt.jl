using Manifolds, Manopt, Test

@testset "Conjugate Residual Plan" begin
    M = ℝ^2
    p = [1.0, 1.0]
    TpM = TangentSpace(M, p)

    Am = [2.0 1.0; 1.0 4.0]
    bv = [1.0, 2.0]
    ps = Am \ (-bv)
    X0 = [3.0, 4.0]
    A(TpM, X, V) = Am * V
    b(TpM, p) = bv
    A!(M, W, X, V) = (W .= Am * V)
    b!(M, W, p) = (W .= bv)

    slso = SymmetricLinearSystemObjective(A, b)
    slso2 = SymmetricLinearSystemObjective(A!, b!; evaluation = InplaceEvaluation())
    @testset "Objective" begin
        grad_value = A(TpM, p, X0) + b(TpM, p)
        cost_value = 0.5 * norm(M, p, grad_value)^2
        @test get_cost(TpM, slso, X0) ≈ cost_value
        @test get_cost(TpM, slso2, X0) ≈ cost_value

        @test Manopt.vector_field(TpM, slso) == bv
        @test Manopt.vector_field(TpM, slso2) == bv

        @test get_gradient(TpM, slso, X0) == grad_value
        @test get_gradient(TpM, slso2, X0) == grad_value
        Y0 = similar(X0)
        @test get_gradient!(TpM, Y0, slso, X0) == grad_value
        @test Y0 == grad_value
        zero_vector!(TpM, Y0, X0)
        @test get_gradient!(TpM, Y0, slso2, X0) == grad_value
        @test Y0 == grad_value

        hessAX0 = A(TpM, p, X0)
        @test get_hessian(TpM, slso, p, X0) == hessAX0
        @test get_hessian(TpM, slso2, p, X0) == hessAX0
        zero_vector!(TpM, Y0, X0)
        @test get_hessian!(TpM, Y0, slso, p, X0) == hessAX0
        @test Y0 == hessAX0
        zero_vector!(TpM, Y0, X0)
        @test get_hessian!(TpM, Y0, slso2, p, X0) == hessAX0
        @test Y0 == hessAX0
    end
    @testset "Conjugate residual state" begin
        crs = ConjugateResidualState(TpM, slso)
        @test set_iterate!(crs, TpM, X0) == crs # setters return state
        @test get_iterate(crs) == X0
        @test set_gradient!(crs, TpM, X0) == crs # setters return state
        @test get_gradient(crs) == X0
        @test startswith(
            repr(crs), "# Solver state for `Manopt.jl`s Conjugate Residual Method"
        )
        crs2 = ConjugateResidualState(TpM, slso2)
        @test set_iterate!(crs2, TpM, X0) == crs2 # setters return state
        @test get_iterate(crs2) == X0
        @test set_gradient!(crs2, TpM, X0) == crs2 # setters return state
        @test get_gradient(crs2) == X0
        @test startswith(
            repr(crs2), "# Solver state for `Manopt.jl`s Conjugate Residual Method"
        )
    end
    @testset "StopWhenRelativeResidualLess" begin
        dmp = DefaultManoptProblem(TpM, slso)
        crs = ConjugateResidualState(TpM, slso; X = X0)
        swrr = StopWhenRelativeResidualLess(1.0, 1.0e-3) #initial norm 1.0, ε=1e-9
        @test startswith(repr(swrr), "StopWhenRelativeResidualLess(1.0, 0.001)")
        # initially this resets norm
        swrr(dmp, crs, 0)
        @test swrr.c == norm(bv)
        @test swrr(dmp, crs, 1) == false
        # sop reason is also empty still
        @test length(get_reason(swrr)) == 0
        # Manually set residual small
        crs.r = [1.0e-5, 1.0e-5]
        @test swrr(dmp, crs, 2) == true
        @test swrr.norm_r == norm(crs.r)
        @test length(get_reason(swrr)) > 0
        @test Manopt.indicates_convergence(swrr)
    end
end
