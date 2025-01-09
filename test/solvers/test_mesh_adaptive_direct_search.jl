using Manifolds, Manopt, Test, LinearAlgebra

@testset "Mesh Adaptive Direct Search" begin
    # A small spectral procrustes example
    A = [1.0 0.0; 1.0 1.0; 0.0 1.0]
    W = 1 / sqrt(2) .* [1.0 -1.0; 1.0 1.0]
    B = A * W
    M = Rotations(2)
    p0 = [1.0 0.0; 0.0 1.0]
    f(M, p) = opnorm(B - A * p)
    p_s = mesh_adaptive_direct_search(
        M,
        f,
        p0;
        # debug=[:Iteration, :Cost, " ", :poll_size, " ", :mesh_size, " ", :Stop, "\n"],
    )
    @test distance(M, p_s, W) < 1e-9
end
