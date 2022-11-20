using Manifolds, Manopt, Test
using RecursiveArrayTools
using ForwardDiff
using LinearAlgebra

function get_test_pts(a, ta, b, tb, R, n=10, σ=0.1)
    xs = randn(n) .+ ta
    ys = randn(n) .+ tb
    zs = (xs ./ a) .^ 2 .+ (ys ./ b) .^ 2 .+ randn(n) .* σ
    tab = R * hcat(xs, ys, zs)'
    return tab
end

ref_a = 1.0
ref_b = 2.0
ref_ta = 0.2
ref_tb = 0.4
ref_R = [
    -0.41571494227143946 0.6951647316993429 -0.5864529670601336
    0.059465856397869304 -0.6226564643891606 -0.7802324905291101
    -0.9075488409419746 -0.35922823268191284 0.21750903004038213
]

# pts_LM = get_test_pts(ref_a, ref_ta, ref_b, ref_tb, ref_R)
pts_LM = [
    0.10108193195702327 -0.8764284104731094 -3.307138431083348 0.0488962756369265 0.18244430954834168 -0.8550225288642191 -0.32898036928095753 0.9796256670268546 -0.8874178117371324 -4.648724558861507
    -2.5898739444779446 -1.2653014644508795 -4.852480171678251 -0.9387505540177796 -0.4143657457881405 0.49828149840711095 0.1424468445934662 -2.326328245656781 -1.3434279088669636 -3.623148694797671
    -1.0830522025857943 1.8791855370819648 -1.282692591839118 -0.7347979252747048 -0.4084584263925703 0.1982254246078313 -0.07420988508644097 -0.415136126735132 1.8810552568366594 -0.6541142317145087
]

function F_RLM(M::AbstractManifold, p)
    # model: let's assume that we want to approximate the the set of points pts_LM by a
    # rotated and translated paraboloid z = x^2/a^2 + y^2/b^2, with positive a and b
    rot = p[M, 1]
    transl = p[M, 2]
    ab = p[M, 3]
    pts_rot_transl = rot * pts_LM .+ transl
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
    x0 = ArrayPartition(ref_R, zeros(3), [1.0, 1.0])
    o = Manopt.LevenbergMarquardt(
        M,
        F_RLM,
        jacF_RLM,
        x0;
        debug=[DebugCost(), DebugIterate(), DebugGradient(), 10],
        return_options=true,
    )
    @test norm(M, get_options(o).x, get_gradient(o)) < 1e-3
end
