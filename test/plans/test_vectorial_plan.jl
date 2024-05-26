using Manopt, ManifoldsBase, Test
using Manopt: get_value, get_value_function, get_gradient_function
@testset "VectorialGradientCost" begin
    M = ManifoldsBase.DefaultManifold(3)
    g(M, p) = [p[1] - 1, -p[2] - 1]
    # # Function
    grad_g(M, p) = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    function grad_g!(M, X, p)
        X[1] .= [1.0, 0.0, 0.0]
        X[2] .= [0.0, -1.0, 0.0]
        return X
    end
    # vectorial
    g1(M, p) = p[1] - 1
    grad_g1(M, p) = [1.0, 0.0, 0.0]
    grad_g1!(M, X, p) = (X .= [1.0, 0.0, 0.0])
    g2(M, p) = -p[2] - 1
    grad_g2(M, p) = [0.0, -1.0, 0.0]
    grad_g2!(M, X, p) = (X .= [0.0, -1.0, 0.0])
    # check a few case
    vgf_fa = VectorGradientFunction(g, grad_g, 2)
    @test get_value_function(vgf_fa) === g
    @test get_gradient_function(vgf_fa) == grad_g
    vgf_va = VectorGradientFunction(
        [g1, g2],
        [grad_g1, grad_g2],
        2;
        function_type=ComponentVectorialType(),
        jacobian_type=ComponentVectorialType(),
    )
    vgf_fi = VectorGradientFunction(g, grad_g!, 2; evaluation=InplaceEvaluation())
    vgf_vi = VectorGradientFunction(
        [g1, g2],
        [grad_g1!, grad_g2!],
        2;
        function_type=ComponentVectorialType(),
        jacobian_type=ComponentVectorialType(),
        evaluation=InplaceEvaluation(),
    )

    p = [1.0, 2.0, 3.0]
    c = [0.0, -3.0]
    gg = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]

    for vgf in [vgf_fa, vgf_va, vgf_fi, vgf_vi]
        @test length(vgf) == 2
        @test get_value(M, vgf, p) == c
        @test get_value(M, vgf, p, :) == c
        @test get_value(M, vgf, p, 1) == c[1]
        @test get_gradient(M, vgf, p) == gg
        @test get_gradient(M, vgf, p, :) == gg
        @test get_gradient(M, vgf, p, 1:2) == gg
        @test get_gradient(M, vgf, p, 1) == gg[1]
        @test get_gradient(M, vgf, p, 2) == gg[2]
        Y = [zero_vector(M, p), zero_vector(M, p)]
        get_gradient!(M, Y, vgf, p, :)
        @test Y == gg
        Z = zero_vector(M, p)
        get_gradient!(M, Z, vgf, p, 1)
        @test Z == gg[1]
        get_gradient!(M, Z, vgf, p, 2)
        @test Z == gg[2]
    end
end
