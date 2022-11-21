using Manifolds, Manopt, Test
using ForwardDiff
using LinearAlgebra

ref_points = [1.0 0.2 1.0 0.4 2.4; 2.0 1.0 -1.0 0.0 1.0; -1.0 0.0 0.3 -0.2 -0.2]
ref_R = [
    -0.41571494227143946 0.6951647316993429 -0.5864529670601336
    0.059465856397869304 -0.6226564643891606 -0.7802324905291101
    -0.9075488409419746 -0.35922823268191284 0.21750903004038213
]

function get_test_pts(σ=0.02)
    return ref_R * ref_points .+ randn(size(ref_points)) .* σ
    return tab
end

# pts_LM = get_test_pts()
pts_LM = [
    1.5747534054082786 0.613147933039177 -1.3027520893952775 -0.049873207038674865 -0.1844917133708744
    -0.4084272005242818 -0.6041141589589345 0.4460538942260599 0.19161082134842558 -0.31006162766339573
    -1.847396743433755 -0.5550994802446793 -0.4912051771949371 -0.4108425833860989 -2.5887967299906682
]

function F_RLM(::AbstractManifold, p)
    # find optimal rotation    
    return vec(pts_LM - p * ref_points)
end

function jacF_RLM(M::AbstractManifold, p)
    X0 = zeros(manifold_dimension(M))
    B = DefaultOrthonormalBasis()
    J = ForwardDiff.jacobian(x -> F_RLM(M, exp(M, p, get_vector(M, p, x, B))), X0)

    return J
end

function F_qN(M::AbstractManifold, p)
    if !all(isfinite.(p))
        println(p)
        error()
    end
    return 1//2 * norm(F_RLM(M, p))^2
end

function grad_F_qN(M::AbstractManifold, p)
    X0 = zeros(manifold_dimension(M))
    B = DefaultOrthonormalBasis()
    return get_vector(
        M, p, ForwardDiff.gradient(x -> F_qN(M, exp(M, p, get_vector(M, p, x, B))), X0), B
    )
end

@testset "LevenbergMarquardt" begin
    M = SpecialOrthogonal(3)
    x0 = exp(M, ref_R, get_vector(M, ref_R, randn(3) * 0.00001, DefaultOrthonormalBasis()))
    o = Manopt.LevenbergMarquardt(M, F_RLM, jacF_RLM, x0; return_options=true)
    x_opt = get_options(o).x
    @test norm(M, x_opt, get_gradient(o)) < 1e-3
    @test distance(M, ref_R, x_opt) < 1e-2
end
