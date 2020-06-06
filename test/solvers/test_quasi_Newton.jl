using Manopt, Manifolds, ManifoldsBase, LinearAlgebra, Test

@testset "Riemannian Limited Memory BFGS Method" begin
    A = [1; 0; 0; 0; 0; 0; 0; 0; 0]
    B = [1; 0; 0; 0; 0; 0; 0; 0; 0]
    C = [1; 0; 0; 0; 0; 0; 0; 0; 0]

    F(x) = 0.5*norm(A-x)^2 + 0.5*norm(B-x)^2 + 0.5*norm(C-x)^2
    ∇F(x) = - A - B - C + 3*x

    M = Euclidean(3,3)
    x = random_point(M)

    steps = Array{TVector, 1}
    graddiffs = Array{TVector, 1}

    for i in 1 : 30
        steps[i] = zero_tangent_vector(M,x)
        graddiffs[i] = zero_tangent_vector(M,x)
    end

    p = GradientProblem(M,F,∇F)
    o = RLBFGSOptions(x,graddiffs,steps,30,0, )
