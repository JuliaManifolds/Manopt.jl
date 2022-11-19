using Manifolds, Manopt, Test
using RecursiveArrayTools
using ForwardDiff
using LinearAlgebra

pts_LM = randn(3, 100)

function F_RLM(M::AbstractManifold, p)
    # model: let's assume that we want to approximate the the set of points pts_LM by a
    # rotated and translated paraboloid z = x^2/a^2 + y^2/b^2, with positive a and b
    rot = p[M, 1]
    transl = p[M, 2]
    ab = p[M, 3]
    pts_rot_transl = rot * (pts_LM .+ transl)
    xs = pts_rot_transl[1, :]
    ys = pts_rot_transl[2, :]
    zs = pts_rot_transl[3, :]
    a2 = ab[1]^2
    b2 = ab[2]^2
    pred_zs = xs .^ 2 ./ a2 .+ ys .^ 2 ./ b2
    return zs - pred_zs
end

function jacF_RLM(M::AbstractManifold, p)
    X0 = zeros(manifold_dimension(M))
    B = DefaultOrthonormalBasis()
    J = ForwardDiff.jacobian(x -> F_RLM(M, exp(M, p, get_vector(M, p, x, B))), X0)

    return J
end

@testset "LevenbergMarquardt" begin
    M = ProductManifold(SpecialOrthogonal(3), Euclidean(3), PositiveVectors(2))
    x0 = ArrayPartition(diagm([1.0, 1.0, 1.0]), zeros(3), [1.0, 1.0])
    Manopt.LevenbergMarquardt(M, F_RLM, jacF_RLM, x0)
end
