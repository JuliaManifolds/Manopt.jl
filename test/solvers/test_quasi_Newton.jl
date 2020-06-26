using Manopt, Manifolds, ManifoldsBase, LinearAlgebra, Test

@testset "Riemannian Limited Memory BFGS Method" begin
    A = [1. 0. 0.; 0. 0. 0.; 0. 0. 0.]
    B = [0. 0. 1.; 0. 0. 0.; 0. 0. 0.]
    C = [0. 0. 0.; 0. 0. 0.; 1. 0. 0.]

    F(x) = 0.5*norm(A-x)^2 + 0.5*norm(B-x)^2 + 0.5*norm(C-x)^2
    ∇F(x) = - A - B - C + 3*x

    M = Euclidean(3,3)
    x = [0. 0. 0.; 0. 1. 0.; 0. 0. 0.]

    steps = [zero_tangent_vector(M,x) for i ∈ 1:30]
    graddiffs = [zero_tangent_vector(M,x) for i ∈ 1:30]
    basis = get_vectors(M, x, get_basis(M, x, DefaultOrthonormalBasis()))
    grad_x = zero_tangent_vector(M,x)

    p = GradientProblem(M,F,∇F)
    o_lm = RLBFGSOptions(x,graddiffs,steps)
    # o = RBFGSQuasiNewton(x,grad_x,basis)

    @test step_solver!(p,o_lm,1)
    # @test step_solver!(p,o,1)
end
