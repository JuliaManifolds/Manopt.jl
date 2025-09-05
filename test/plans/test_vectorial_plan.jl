using Manopt, ManifoldsBase, Test
using Manopt: get_value, get_value_function, get_gradient_function
@testset "VectorialGradientCost" begin
    M = ManifoldsBase.DefaultManifold(3)
    g(M, p) = [p[1] - 1, -p[2] - 1]
    g!(M, V, p) = (V .= [p[1] - 1, -p[2] - 1])
    # # Function
    grad_g(M, p) = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    hess_g(M, p, X) = [copy(X), -copy(X)]
    hess_g!(M, Y, p, X) = (Y .= [copy(X), -copy(X)])
    # since the ONB of M is just the identity in coefficients, JF is gradients'
    jac_g(M, p) = [1.0 0.0; 0.0 -1.0; 0.0 0.0]'
    jac_g!(M, J, p) = (J .= [1.0 0.0; 0.0 -1.0; 0.0 0.0]')
    function grad_g!(M, X, p)
        X[1] .= [1.0, 0.0, 0.0]
        X[2] .= [0.0, -1.0, 0.0]
        return X
    end
    # vectorial
    g1(M, p) = p[1] - 1
    grad_g1(M, p) = [1.0, 0.0, 0.0]
    grad_g1!(M, X, p) = (X .= [1.0, 0.0, 0.0])
    hess_g1(M, p, X) = copy(X)
    hess_g1!(M, Y, p, X) = copyto!(Y, X)
    g2(M, p) = -p[2] - 1
    grad_g2(M, p) = [0.0, -1.0, 0.0]
    grad_g2!(M, X, p) = (X .= [0.0, -1.0, 0.0])
    hess_g2(M, p, X) = copy(-X)
    hess_g2!(M, Y, p, X) = copyto!(Y, -X)
    # verify a few case
    vgf_fa = VectorGradientFunction(g, grad_g, 2)
    @test get_value_function(vgf_fa) === g
    @test get_gradient_function(vgf_fa) == grad_g
    vgf_va = VectorGradientFunction(
        [g1, g2],
        [grad_g1, grad_g2],
        2;
        function_type = ComponentVectorialType(),
        jacobian_type = ComponentVectorialType(),
    )
    vgf_fi = VectorGradientFunction(g!, grad_g!, 2; evaluation = InplaceEvaluation())
    vgf_vi = VectorGradientFunction(
        [g1, g2],
        [grad_g1!, grad_g2!],
        2;
        function_type = ComponentVectorialType(),
        jacobian_type = ComponentVectorialType(),
        evaluation = InplaceEvaluation(),
    )
    vgf_ja = VectorGradientFunction(
        g, jac_g, 2; jacobian_type = CoordinateVectorialType(DefaultOrthonormalBasis())
    )
    vgf_ji = VectorGradientFunction(
        g!,
        jac_g!,
        2;
        jacobian_type = CoordinateVectorialType(DefaultOrthonormalBasis()),
        evaluation = InplaceEvaluation(),
    )
    @test Manopt.get_jacobian_basis(vgf_ji) == vgf_ji.jacobian_type.basis
    @test Manopt.get_jacobian_basis(vgf_vi) == DefaultOrthonormalBasis()
    vgf_jib = VectorGradientFunction(
        g!,
        jac_g!,
        2;
        jacobian_type = CoordinateVectorialType(DefaultBasis()),
        evaluation = InplaceEvaluation(),
    )
    @test Manopt.get_jacobian_basis(vgf_ji) == vgf_ji.jacobian_type.basis
    @test Manopt.get_jacobian_basis(vgf_jib) == DefaultBasis()
    @test Manopt.get_jacobian_basis(vgf_vi) == DefaultOrthonormalBasis()
    p = [1.0, 2.0, 3.0]
    c = [0.0, -3.0]
    gg = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]

    # With Hessian
    vhf_fa = VectorHessianFunction(g, grad_g, hess_g, 2)
    vhf_va = VectorHessianFunction(
        [g1, g2],
        [grad_g1, grad_g2],
        [hess_g1, hess_g2],
        2;
        function_type = ComponentVectorialType(),
        jacobian_type = ComponentVectorialType(),
        hessian_type = ComponentVectorialType(),
    )
    vhf_fi = VectorHessianFunction(g!, grad_g!, hess_g!, 2; evaluation = InplaceEvaluation())
    vhf_vi = VectorHessianFunction(
        [g1, g2],
        [grad_g1!, grad_g2!],
        [hess_g1!, hess_g2!],
        2;
        function_type = ComponentVectorialType(),
        jacobian_type = ComponentVectorialType(),
        hessian_type = ComponentVectorialType(),
        evaluation = InplaceEvaluation(),
    )

    for vgf in
        [vgf_fa, vgf_va, vgf_fi, vgf_vi, vgf_ja, vgf_ji, vhf_fa, vhf_fi, vhf_va, vhf_vi]
        @test length(vgf) == 2
        @test get_value(M, vgf, p) == c
        @test get_value(M, vgf, p, :) == c
        @test get_value(M, vgf, p, 1) == c[1]
        @test get_gradient(M, vgf, p) == gg
        @test get_gradient(M, vgf, p, :) == gg
        @test get_gradient(M, vgf, p, 1:2) == gg
        @test get_gradient(M, vgf, p, [1, 2]) == gg
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

    X = [1.0, 0.5, 0.25]
    gh = [X, -X]
    # Hessian
    @test Manopt.get_hessian_function(vhf_fa) === hess_g
    @test all(Manopt.get_hessian_function(vhf_va) .=== [hess_g1, hess_g2])
    @test Manopt.get_hessian_function(vhf_fi) === hess_g!
    @test all(Manopt.get_hessian_function(vhf_vi) .=== [hess_g1!, hess_g2!])
    for vhf in [vhf_fa, vhf_va, vhf_fi, vhf_vi]
        @test get_hessian(M, vhf, p, X) == gh
        @test get_hessian(M, vhf, p, X, :) == gh
        @test get_hessian(M, vhf, p, X, 1:2) == gh
        @test get_hessian(M, vhf, p, X, [1, 2]) == gh
        @test get_hessian(M, vhf, p, X, 1) == gh[1]
        @test get_hessian(M, vhf, p, X, 2) == gh[2]
        Y = [zero_vector(M, p), zero_vector(M, p)]
        get_hessian!(M, Y, vhf, p, X, :)
        @test Y == gh
        Z = zero_vector(M, p)
        get_hessian!(M, Z, vhf, p, X, 1)
        @test Z == gh[1]
        get_hessian!(M, Z, vhf, p, X, 2)
        @test Z == gh[2]
    end
end
