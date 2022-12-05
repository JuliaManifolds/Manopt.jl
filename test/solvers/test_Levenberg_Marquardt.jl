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

function jacF_RLM(
    M::AbstractManifold, p; basis_domain::AbstractBasis=DefaultOrthogonalBasis()
)
    X0 = zeros(manifold_dimension(M))
    J = ForwardDiff.jacobian(
        x -> F_RLM(M, exp(M, p, get_vector(M, p, x, basis_domain))), X0
    )

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

struct F_reg_r2
    ts_r2::Vector{Float64}
    xs_r2::Vector{Float64}
    ys_r2::Vector{Float64}
end

function (f::F_reg_r2)(::AbstractManifold, p)
    return vcat(f.ts_r2 .* p[1] .- f.xs_r2, f.ts_r2 .* p[2] .- f.ys_r2)
end

struct jacF_reg_r2
    ts_r2::Vector{Float64}
    xs_r2::Vector{Float64}
    ys_r2::Vector{Float64}
end

function (f::jacF_reg_r2)(
    ::AbstractManifold, p; basis_domain::AbstractBasis=DefaultOrthogonalBasis()
)
    return [f.ts_r2 zero(f.ts_r2); zero(f.ts_r2) f.ts_r2]
end

function F_reg_r2!(::AbstractManifold, x, p)
    midpoint = div(length(x), 2)
    view(x, 1:midpoint) .= ts_r2 .* p[1] .- xs_r2
    view(x, (midpoint + 1):length(x)) .= ts_r2 .* p[2] .- ys_r2
    return x
end

function jacF_reg_r2!(
    ::AbstractManifold, J, p; basis_domain::AbstractBasis=DefaultOrthogonalBasis()
)
    midpoint = div(size(J, 1), 2)
    view(J, 1:midpoint, 1) .= ts_r2
    view(J, 1:midpoint, 2) .= 0
    view(J, (midpoint + 1):size(J, 1), 1) .= 0
    view(J, (midpoint + 1):size(J, 1), 2) .= ts_r2
    return J
end

@testset "LevenbergMarquardt" begin
    # testing on rotations
    M = Rotations(3)
    x0 = exp(M, ref_R, get_vector(M, ref_R, randn(3) * 0.00001, DefaultOrthonormalBasis()))

    o = LevenbergMarquardt(M, F_RLM, jacF_RLM, x0, length(pts_LM); return_options=true)
    x_opt = get_options(o).x
    @test norm(M, x_opt, get_gradient(o)) < 2e-3
    @test distance(M, ref_R, x_opt) < 1e-2

    # allocating R2 regression, nonzero residual
    M = Euclidean(2)
    x0 = [0.0, 0.0]
    o = LevenbergMarquardt(
        M,
        F_reg_r2(ts_r2, xs_r2, ys_r2),
        jacF_reg_r2(ts_r2, xs_r2, ys_r2),
        x0,
        length(ts_r2) * 2;
        return_options=true,
    )
    @test isapprox(o.options.x[1], 2, atol=0.01)
    @test isapprox(o.options.x[2], -3, atol=0.01)

    # testing with a basis that requires caching
    M = Euclidean(2)
    x0 = [0.0, 0.0]
    o = LevenbergMarquardt(
        M,
        F_reg_r2(ts_r2, xs_r2, ys_r2),
        jacF_reg_r2(ts_r2, xs_r2, ys_r2),
        x0,
        length(ts_r2) * 2;
        return_options=true,
        jacB=ProjectedOrthonormalBasis(:svd),
    )
    @test isapprox(o.options.x[1], 2, atol=0.01)
    @test isapprox(o.options.x[2], -3, atol=0.01)

    # allocating R2 regression, zero residual
    M = Euclidean(2)
    x0 = [0.0, 0.0]
    o = LevenbergMarquardt(
        M,
        F_reg_r2(ts_r2, 2 * ts_r2, -3 * ts_r2),
        jacF_reg_r2(ts_r2, 2 * ts_r2, -3 * ts_r2),
        x0;
        return_options=true,
        expect_zero_residual=true,
    )
    @test isapprox(o.options.x[1], 2, atol=0.01)
    @test isapprox(o.options.x[2], -3, atol=0.01)

    # mutating R2 regression
    x0 = [0.0, 0.0]
    o_mut = LevenbergMarquardt(
        M,
        F_reg_r2!,
        jacF_reg_r2!,
        x0,
        length(ts_r2) * 2;
        return_options=true,
        evaluation=MutatingEvaluation(),
    )
    @test isapprox(o_mut.options.x[1], 2, atol=0.01)
    @test isapprox(o_mut.options.x[2], -3, atol=0.01)

    x0 = [4.0, 2.0]
    o_r2 = LevenbergMarquardtState(
        M,
        x0,
        similar(x0, length(ts_r2)),
        similar(x0, 2 * length(ts_r2), 2);
        stopping_criterion=StopAfterIteration(20),
    )
    p_r2 = NonlinearLeastSquaresProblem(
        M,
        F_reg_r2(ts_r2, xs_r2, ys_r2),
        jacF_reg_r2(ts_r2, xs_r2, ys_r2),
        length(ts_r2) * 2,
    )

    X_r2 = similar(x0)
    get_gradient!(p_r2, X_r2, x0)
    @test isapprox(X_r2, [270.3617451389837, 677.6730784956912])

    p_r2_mut = NonlinearLeastSquaresProblem(
        M, F_reg_r2!, jacF_reg_r2!, length(ts_r2) * 2; evaluation=MutatingEvaluation()
    )

    X_r2 = similar(x0)
    get_gradient!(p_r2_mut, X_r2, x0)
    @test isapprox(X_r2, [270.3617451389837, 677.6730784956912])

    @testset "debug options" begin
        io = IOBuffer()
        # Additional Specific Debugs
        a1 = DebugGradient(; long=false, io=io)
        a1(p_r2, o_r2, 1)
        @test String(take!(io)) == "gradF(x):[0.0, 0.0]"
    end

    @testset "errors" begin
        @test_throws ArgumentError LevenbergMarquardtState(
            M,
            x0,
            similar(x0, length(ts_r2)),
            similar(x0, 2 * length(ts_r2), 2);
            stopping_criterion=StopAfterIteration(20),
            η=2,
        )

        @test_throws ArgumentError LevenbergMarquardtState(
            M,
            x0,
            similar(x0, length(ts_r2)),
            similar(x0, 2 * length(ts_r2), 2);
            stopping_criterion=StopAfterIteration(20),
            damping_term_min=-1,
        )

        @test_throws ArgumentError LevenbergMarquardtState(
            M,
            x0,
            similar(x0, length(ts_r2)),
            similar(x0, 2 * length(ts_r2), 2);
            stopping_criterion=StopAfterIteration(20),
            β=0.5,
        )

        @test_throws ArgumentError LevenbergMarquardt(
            M,
            F_reg_r2!,
            jacF_reg_r2!,
            x0;
            return_options=true,
            evaluation=MutatingEvaluation(),
        )
    end
end