using Manifolds, Manopt, Test, LinearAlgebra, Random

@testset "Mesh Adaptive Direct Search" begin
    # A small spectral procrustes example
    A = [1.0 0.0; 1.0 1.0; 0.0 1.0]
    W = 1 / sqrt(2) .* [1.0 -1.0; 1.0 1.0]
    B = A * W
    M = Rotations(2)
    p0 = [1.0 0.0; 0.0 1.0]
    f(M, p) = opnorm(B - A * p)
    Random.seed!(42)
    s = mesh_adaptive_direct_search(
        M,
        f,
        p0;
        # debug=[:Iteration, :Cost, " ", :poll_size, " ", :mesh_size, " ", :Stop, "\n"],
        return_state = true,
    )
    @test distance(M, get_solver_result(s), W) < 1.0e-9
    @test startswith(get_reason(s), "The algorithm computed a poll step size")
    #
    #
    # A bit larger example inplace
    # A small spectral Procrustes example
    A2 = [1.0 0.0 0.0; 1.0 1.0 1.0; 0.0 1.0 2.0; 1.0 1.0 1.0]
    α = π / 8
    W2 = [cos(α) -sin(α) 0.0; sin(α) cos(α) 0.0; 0.0 0.0 1.0]
    B2 = A2 * W2
    M2 = Rotations(3)
    p1 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    f2(M, p) = opnorm(B2 - A2 * p)
    Random.seed!(42)
    # start with a very small mesh size - yields a more exact result
    p_s2 = mesh_adaptive_direct_search!(M2, f2, p1; scale_mesh = 0.1)
    @test isapprox(M, p_s2, p1)
    @test distance(M2, p_s2, W2) < 1.0e-7
    #
    #
    # An example on a sphere (to ensure type coherence between injectivity radii)
    M3 = Sphere(2)
    f3(M, p) = sum(p)
    p2 = [-1.0, 0.0, 0.0]
    s3 = mesh_adaptive_direct_search(M3, f3, p2; return_state = true)
    p3 = get_solver_result(s3)
    @test f3(M3, p3) < f3(M3, p2)
    @test is_point(M3, p3; error = :error)
end
