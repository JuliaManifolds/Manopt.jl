using Manifolds, Manopt, Test, ManifoldsBase
using ForwardDiff
using LinearAlgebra

const ref_points = [1.0 0.2 1.0 0.4 2.4; 2.0 1.0 -1.0 0.0 1.0; -1.0 0.0 0.3 -0.2 -0.2]
const ref_R = [
    -0.41571494227143946 0.6951647316993429 -0.5864529670601336
    0.059465856397869304 -0.6226564643891606 -0.7802324905291101
    -0.9075488409419746 -0.35922823268191284 0.21750903004038213
]

function get_test_pts(σ = 0.02)
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
        M::AbstractManifold, p; basis_domain::AbstractBasis = default_basis(M, typeof(p))
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
        M::AbstractManifold, p; basis_domain::AbstractBasis = default_basis(M, typeof(p))
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
        M::AbstractManifold, J, p; basis_domain::AbstractBasis = default_basis(M, typeof(p))
    )
    midpoint = div(size(J, 1), 2)
    view(J, 1:midpoint, 1) .= ts_r2
    view(J, 1:midpoint, 2) .= 0
    view(J, (midpoint + 1):size(J, 1), 1) .= 0
    view(J, (midpoint + 1):size(J, 1), 2) .= ts_r2
    return J
end

function test_lm_lin_solve!(sk, JJ, grad_f_c)
    ldiv!(sk, qr(JJ), grad_f_c)
    return sk
end

@testset "LevenbergMarquardt" begin
    # testing on rotations
    M = Rotations(3)
    p0 = exp(M, ref_R, get_vector(M, ref_R, randn(3) * 0.00001, DefaultOrthonormalBasis()))

    lm_r = LevenbergMarquardt(M, F_RLM, jacF_RLM, p0, length(pts_LM); return_state = true)
    lm_rs = "# Solver state for `Manopt.jl`s Levenberg Marquardt Algorithm\n"
    @test startswith(repr(lm_r), lm_rs)
    p_opt = get_state(lm_r).p
    @test norm(M, p_opt, get_gradient(lm_r)) < 2.0e-3
    p_atol = 1.5e-2
    @test isapprox(M, ref_R, p_opt; atol = p_atol)

    # allocating R2 regression, nonzero residual
    M = Euclidean(2)
    p0 = [0.0, 0.0]
    p_star = [2, -3]
    ds = LevenbergMarquardt(
        M,
        F_reg_r2(ts_r2, xs_r2, ys_r2),
        jacF_reg_r2(ts_r2, xs_r2, ys_r2),
        p0,
        length(ts_r2) * 2;
        return_state = true,
    )
    lms = get_state(ds)
    @test isapprox(M, p_star, lms.p; atol = p_atol)

    # testing with a basis that requires caching
    ds = LevenbergMarquardt(
        M,
        F_reg_r2(ts_r2, xs_r2, ys_r2),
        jacF_reg_r2(ts_r2, xs_r2, ys_r2),
        p0,
        length(ts_r2) * 2;
        return_state = true,
        jacobian_tangent_basis = ProjectedOrthonormalBasis(:svd),
    )
    lms = get_state(ds)
    @test isapprox(M, p_star, lms.p; atol = p_atol)
    # start with random point
    p2 = LevenbergMarquardt(
        M,
        F_reg_r2(ts_r2, xs_r2, ys_r2),
        jacF_reg_r2(ts_r2, xs_r2, ys_r2),
        length(ts_r2) * 2
    )
    @test isapprox(M, p_star, p2; atol = p_atol)
    # testing in-place
    p3 = copy(M, p0)
    LevenbergMarquardt!(
        M,
        F_reg_r2(ts_r2, xs_r2, ys_r2),
        jacF_reg_r2(ts_r2, xs_r2, ys_r2),
        p3,
        length(ts_r2) * 2
    )
    @test isapprox(M, p_star, p3; atol = p_atol)

    # allocating R2 regression, zero residual, custom subsolver
    M = Euclidean(2)
    ds = LevenbergMarquardt(
        M,
        F_reg_r2(ts_r2, 2 * ts_r2, -3 * ts_r2),
        jacF_reg_r2(ts_r2, 2 * ts_r2, -3 * ts_r2),
        p0;
        return_state = true,
        expect_zero_residual = true,
        (linear_subsolver!) = (test_lm_lin_solve!),
    )
    lms = get_state(ds)
    @test lms.sub_problem === test_lm_lin_solve!
    @test isapprox(M, p_star, lms.p; atol = p_atol)

    p1 = copy(M, p0)
    LevenbergMarquardt!(
        M,
        F_reg_r2(ts_r2, 2 * ts_r2, -3 * ts_r2),
        jacF_reg_r2(ts_r2, 2 * ts_r2, -3 * ts_r2),
        p1;
        expect_zero_residual = true,
    )
    @test isapprox(M, p_star, p1; atol = p_atol)

    # mutating R2 regression
    p0 = [0.0, 0.0]
    ds = LevenbergMarquardt(
        M,
        F_reg_r2!,
        jacF_reg_r2!,
        p0,
        length(ts_r2) * 2;
        return_state = true,
        evaluation = InplaceEvaluation(),
    )
    lms = get_state(ds)
    @test isapprox(M, p_star, lms.p; atol = p_atol)

    x0 = [4.0, 2.0]
    o_r2 = LevenbergMarquardtState(
        M,
        similar(x0, length(ts_r2)),
        similar(x0, 2 * length(ts_r2), 2);
        p = x0,
        stopping_criterion = StopAfterIteration(20),
    )
    p_r2 = DefaultManoptProblem(
        M,
        NonlinearLeastSquaresObjective(
            F_reg_r2(ts_r2, xs_r2, ys_r2),
            jacF_reg_r2(ts_r2, xs_r2, ys_r2),
            length(ts_r2) * 2,
        ),
    )

    X_r2 = similar(x0)
    get_gradient!(p_r2, X_r2, x0)
    @test isapprox(X_r2, [270.3617451389837, 677.6730784956912])
    @test isapprox(get_gradient(p_r2, x0), [270.3617451389837, 677.6730784956912])

    p_r2_mut = DefaultManoptProblem(
        M,
        NonlinearLeastSquaresObjective(
            F_reg_r2!, jacF_reg_r2!, length(ts_r2) * 2; evaluation = InplaceEvaluation()
        ),
    )

    X_r2 = similar(x0)
    get_gradient!(p_r2_mut, X_r2, x0)
    @test isapprox(X_r2, [270.3617451389837, 677.6730784956912])
    @test isapprox(get_gradient(p_r2_mut, x0), [270.3617451389837, 677.6730784956912])

    @testset "errors" begin
        @test_throws ArgumentError LevenbergMarquardtState(
            M,
            similar(x0, length(ts_r2)),
            similar(x0, 2 * length(ts_r2), 2);
            p = x0,
            stopping_criterion = StopAfterIteration(20),
            η = 2,
        )

        @test_throws ArgumentError LevenbergMarquardtState(
            M,
            similar(x0, length(ts_r2)),
            similar(x0, 2 * length(ts_r2), 2);
            p = x0,
            stopping_criterion = StopAfterIteration(20),
            damping_term_min = -1,
        )

        @test_throws ArgumentError LevenbergMarquardtState(
            M,
            similar(x0, length(ts_r2)),
            similar(x0, 2 * length(ts_r2), 2);
            p = x0,
            stopping_criterion = StopAfterIteration(20),
            β = 0.5,
        )

        @test_throws ArgumentError LevenbergMarquardt(
            M,
            F_reg_r2!,
            jacF_reg_r2!,
            x0;
            return_state = true,
            evaluation = InplaceEvaluation(),
        )
        @test_throws ArgumentError LevenbergMarquardt!(
            M,
            F_reg_r2!,
            jacF_reg_r2!,
            x0;
            return_state = true,
            evaluation = InplaceEvaluation(),
        )
    end

    @testset "linear subproblem numerical issues" begin
        JJ = [2.0 eps(); 0.0 2.0]
        grad_f_c = [1.0, 2.0]
        sk = similar(grad_f_c)
        Manopt.default_lm_lin_solve!(sk, JJ, grad_f_c)
        @test isapprox(sk, [0.5, 1.0])
        @test_throws SingularException Manopt.default_lm_lin_solve!(sk, NaN .* JJ, grad_f_c)
    end
end
