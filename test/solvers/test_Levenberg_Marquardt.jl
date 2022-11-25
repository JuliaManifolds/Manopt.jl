using Manifolds, Manopt, Test
using ForwardDiff
using LinearAlgebra

const ref_points = [1.0 0.2 1.0 0.4 2.4; 2.0 1.0 -1.0 0.0 1.0; -1.0 0.0 0.3 -0.2 -0.2]
const ref_R = [
    -0.41571494227143946 0.6951647316993429 -0.5864529670601336
    0.059465856397869304 -0.6226564643891606 -0.7802324905291101
    -0.9075488409419746 -0.35922823268191284 0.21750903004038213
]

function get_test_pts(σ=0.02)
    return ref_R * ref_points .+ randn(size(ref_points)) .* σ
    return tab
end

# pts_LM = get_test_pts()
const pts_LM = [
    1.5747534054082786 0.613147933039177 -1.3027520893952775 -0.049873207038674865 -0.1844917133708744
    -0.4084272005242818 -0.6041141589589345 0.4460538942260599 0.19161082134842558 -0.31006162766339573
    -1.847396743433755 -0.5550994802446793 -0.4912051771949371 -0.4108425833860989 -2.5887967299906682
]

function F_RLM(::AbstractManifold, p)
    # find optimal rotation    
    return vec(pts_LM - p * ref_points)
end

function jacF_RLM(M::AbstractManifold, p; B_dom::AbstractBasis=DefaultOrthogonalBasis())
    X0 = zeros(manifold_dimension(M))
    J = ForwardDiff.jacobian(x -> F_RLM(M, exp(M, p, get_vector(M, p, x, B_dom))), X0)

    return J
end

# regression in R^2

const ts_r2 = [1.0, 2.0, 2.4, 3.1, 4.3, 5.1, 5.7, 6.2]
const xs_r2 = [
    1.7926888218350978,
    4.080701652578803,
    4.510454784478316,
    6.44163899446108,
    8.779266091173081,
    9.99046003643165,
    11.383567405022262,
    12.661526739028028,
]
const ys_r2 = [
    -2.89981608417956,
    -6.062187792164395,
    -7.217187782362094,
    -9.204644225025552,
    -13.026122239508274,
    -15.15659923171402,
    -17.076511907478242,
    -18.475183785109927,
]

function F_reg_r2(::AbstractManifold, p)
    return vcat(ts_r2 .* p[1] .- xs_r2, ts_r2 .* p[2] .- ys_r2)
end

function jacF_reg_r2(::AbstractManifold, p; B_dom::AbstractBasis=DefaultOrthogonalBasis())
    return [ts_r2 zero(ts_r2); zero(ts_r2) ts_r2]
end

@testset "LevenbergMarquardt" begin
    # testing on rotations
    M = Rotations(3)
    x0 = exp(M, ref_R, get_vector(M, ref_R, randn(3) * 0.00001, DefaultOrthonormalBasis()))

    o = Manopt.LevenbergMarquardt(M, F_RLM, jacF_RLM, x0; return_options=true)
    x_opt = get_options(o).x
    @test norm(M, x_opt, get_gradient(o)) < 2e-3
    @test distance(M, ref_R, x_opt) < 1e-2

    # plain R2 regression
    M = Euclidean(2)
    x0 = [0.0, 0.0]
    o = Manopt.LevenbergMarquardt(M, F_reg_r2, jacF_reg_r2, x0; return_options=true)
    @test isapprox(o.options.x[1], 2, atol=0.01)
    @test isapprox(o.options.x[2], -3, atol=0.01)
end
