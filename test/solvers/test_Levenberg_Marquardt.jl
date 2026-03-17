using Manifolds, Manopt, Test, ManifoldsBase
using ForwardDiff
using LinearAlgebra
using RecursiveArrayTools
using SparseArrays

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
    x0 = [4.0, 2.0]

    ds = LevenbergMarquardt(
        M, F_reg_r2(ts_r2, xs_r2, ys_r2), jacF_reg_r2(ts_r2, xs_r2, ys_r2), p0, length(ts_r2) * 2;
        return_state = true,
    )
    lms = get_state(ds)
    @test isapprox(M, p_star, lms.p; atol = p_atol)

    # testing with a basis that requires caching
    ds = LevenbergMarquardt(
        M, F_reg_r2(ts_r2, xs_r2, ys_r2), jacF_reg_r2(ts_r2, xs_r2, ys_r2), p0, length(ts_r2) * 2;
        return_state = true, jacobian_tangent_basis = ProjectedOrthonormalBasis(:svd),
    )
    lms = get_state(ds)
    @test isapprox(M, p_star, lms.p; atol = p_atol)
    # start with random point
    p2 = LevenbergMarquardt(
        M, F_reg_r2(ts_r2, xs_r2, ys_r2), jacF_reg_r2(ts_r2, xs_r2, ys_r2), length(ts_r2) * 2
    )
    @test isapprox(M, p_star, p2; atol = p_atol)
    # testing in-place
    p3 = copy(M, p0)
    LevenbergMarquardt!(
        M, F_reg_r2(ts_r2, xs_r2, ys_r2), jacF_reg_r2(ts_r2, xs_r2, ys_r2), p3, length(ts_r2) * 2
    )
    @test isapprox(M, p_star, p3; atol = p_atol)

    # allocating R2 regression, zero residual, custom subsolver
    M = Euclidean(2)
    ds = LevenbergMarquardt(
        M, F_reg_r2(ts_r2, 2 * ts_r2, -3 * ts_r2), jacF_reg_r2(ts_r2, 2 * ts_r2, -3 * ts_r2), p0;
        return_state = true, expect_zero_residual = true, linear_subsolver! = test_lm_lin_solve!,
    )
    lms = get_state(ds)
    @test lms.sub_state.linsolve!! === test_lm_lin_solve!
    @test isapprox(M, p_star, lms.p; atol = p_atol)

    p1 = copy(M, p0)
    LevenbergMarquardt!(
        M, F_reg_r2(ts_r2, 2 * ts_r2, -3 * ts_r2), jacF_reg_r2(ts_r2, 2 * ts_r2, -3 * ts_r2), p1;
        expect_zero_residual = true,
    )
    @test isapprox(M, p_star, p1; atol = p_atol)

    # mutating R2 regression
    p0 = [0.0, 0.0]
    ds = LevenbergMarquardt(
        M, F_reg_r2!, jacF_reg_r2!, p0, length(ts_r2) * 2;
        return_state = true, evaluation = InplaceEvaluation(),
    )
    lms = get_state(ds)
    @test isapprox(M, p_star, lms.p; atol = p_atol)

    p_r2 = DefaultManoptProblem(
        M,
        NonlinearLeastSquaresObjective(
            F_reg_r2(ts_r2, xs_r2, ys_r2), jacF_reg_r2(ts_r2, xs_r2, ys_r2), length(ts_r2) * 2,
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
        sub_fake_f = (args...) -> 0
        sub_state = AllocatingEvaluation()
        i_res = similar(x0, length(ts_r2))
        i_JF = similar(x0, 2 * length(ts_r2), 2)
        # η too large (≥ 1)
        @test_throws ArgumentError LevenbergMarquardtState(
            M, i_res; initial_jacobian_f = i_JF, p = x0, η = 2, sub_problem = sub_fake_f, sub_state = sub_state
        )
        # η too small (≤ 0)
        @test_throws ArgumentError LevenbergMarquardtState(
            M, i_res; initial_jacobian_f = i_JF, p = x0, η = -1, sub_problem = sub_fake_f, sub_state = sub_state
        )
        # damping term negative
        @test_throws ArgumentError LevenbergMarquardtState(
            M, i_res; initial_jacobian_f = i_JF, p = x0, damping_term_min = -1, sub_problem = sub_fake_f, sub_state = sub_state
        )
        # β too small (≤ 1)
        @test_throws ArgumentError LevenbergMarquardtState(
            M, i_res; initial_jacobian_f = i_JF, p = x0, β = 0.5, sub_problem = sub_fake_f, sub_state = sub_state
        )
        # no sub problem provided
        @test_throws ArgumentError LevenbergMarquardtState(
            M, i_res; initial_jacobian_f = i_JF, p = x0, sub_state = sub_state
        )
        # no sub state provided
        @test_throws ArgumentError LevenbergMarquardtState(
            M, i_res; initial_jacobian_f = i_JF, p = x0, sub_problem = sub_fake_f
        )
        # The next two tests check that the error "For mutating evaluation num_components needs to be explicitly specified" is thrown
        @test_throws ArgumentError LevenbergMarquardt(
            M, F_reg_r2!, jacF_reg_r2!, x0; return_state = true, evaluation = InplaceEvaluation(),
        )
        @test_throws ArgumentError LevenbergMarquardt!(
            M, F_reg_r2!, jacF_reg_r2!, x0; return_state = true, evaluation = InplaceEvaluation(),
        )
    end

    @testset "coordinate surrogate agrees with operator surrogate" begin
        M = Euclidean(2)
        p = [0.7, -1.2]
        B = DefaultOrthonormalBasis()

        nlso = NonlinearLeastSquaresObjective(
            F_reg_r2(ts_r2, xs_r2, ys_r2), jacF_reg_r2(ts_r2, xs_r2, ys_r2), length(ts_r2) * 2,
        )

        lmso = LevenbergMarquardtLinearSurrogateObjective(nlso; penalty = 1.0e-3)
        lmcso = Manopt.LevenbergMarquardtLinearSurrogateCoordinatesObjective(
            nlso;
            penalty = 1.0e-3, basis = B, jacobian_cache = [zeros(length(ts_r2) * 2, 2) for _ in eachindex(nlso.objective)],
            residuals = zeros(length(ts_r2) * 2)
        )

        # Coordinate surrogate requires explicit caches, which are normally updated in LM steps.
        get_residuals!(M, lmcso.value_cache, nlso, p)
        for (i, o) in enumerate(nlso.objective)
            lmcso.jacobian_cache[i] = get_jacobian(M, o, p; basis = B)
        end

        slso = Manopt.SymmetricLinearSystem(lmso)
        slco = Manopt.SymmetricLinearSystem(lmcso)

        n = number_of_coordinates(M, B)
        A_lmso = zeros(n, n)
        A_lmcso = zeros(n, n)
        Manopt.get_linear_operator!(M, A_lmso, slso, p, B)
        Manopt.get_linear_operator!(M, A_lmcso, slco, p, B)
        @test isapprox(A_lmso, A_lmcso; atol = 1.0e-12, rtol = 1.0e-12)

        nvf_lmso = zeros(n)
        nvf_lmcso = zeros(n)
        Manopt.get_normal_vector_field!(M, nvf_lmso, lmso, p, B)
        Manopt.get_normal_vector_field_coord!(M, nvf_lmcso, lmcso, p, B)
        @test isapprox(nvf_lmso, nvf_lmcso; atol = 1.0e-12, rtol = 1.0e-12)

        # Directly test add_normal_vector_field_coord! (no-basis overload that uses mul!).
        len_o = length(nlso.objective[1])
        val_cache = view(lmcso.value_cache, 1:len_o)
        jc = lmcso.jacobian_cache[1]

        nvf_direct = zeros(n)
        Manopt.add_normal_vector_field_coord!(
            M,
            nvf_direct,
            nlso.objective[1],
            nlso.robustifier[1],
            p;
            value_cache = val_cache,
            jacobian_cache = jc,
            ε = lmcso.ε,
            mode = lmcso.mode,
        )
        @test isapprox(nvf_direct, nvf_lmcso; atol = 1.0e-12, rtol = 1.0e-12)

        # Verify accumulation semantics from mul!(..., true, true).
        seed = fill(0.7, n)
        nvf_acc = copy(seed)
        Manopt.add_normal_vector_field_coord!(
            M,
            nvf_acc,
            nlso.objective[1],
            nlso.robustifier[1],
            p;
            value_cache = val_cache,
            jacobian_cache = jc,
            ε = lmcso.ε,
            mode = lmcso.mode,
        )
        @test isapprox(nvf_acc, seed .+ nvf_direct; atol = 1.0e-12, rtol = 1.0e-12)

        # Cross-check with the basis overload of add_normal_vector_field_coord!.
        nvf_direct_B = zeros(n)
        Manopt.add_normal_vector_field_coord!(
            M,
            nvf_direct_B,
            nlso.objective[1],
            nlso.robustifier[1],
            p,
            B;
            value_cache = val_cache,
            jacobian_cache = jc,
            ε = lmcso.ε,
            mode = lmcso.mode,
        )
        @test isapprox(nvf_direct_B, nvf_direct; atol = 1.0e-12, rtol = 1.0e-12)

        n_res = sum(length(o) for o in nlso.objective)
        vf_lmso = zeros(n_res)
        vf_lmcso = zeros(n_res)

        Manopt.get_vector_field!(M, vf_lmso, lmso, p)
        Manopt.get_vector_field!(M, vf_lmcso, lmcso, p)
        @test isapprox(vf_lmso, vf_lmcso; atol = 1.0e-12, rtol = 1.0e-12)

        TpM = TangentSpace(M, p)
        X0 = Manopt.ZeroTangentVector()
        cX = [0.3, -0.5]
        X = get_vector(M, p, cX, B)
        @test isapprox(get_cost(TpM, slso, X0), get_cost(TpM, slco, X0); atol = 1.0e-12, rtol = 1.0e-12)
        @test isapprox(get_cost(TpM, slso, X), get_cost(TpM, slco, X); atol = 1.0e-12, rtol = 1.0e-12)

        # Coordinate normal operator action should match the assembled normal matrix.
        c_lmso = A_lmso * cX
        c_lmcso = zeros(n)
        Manopt.add_linear_normal_operator_coord!(M, c_lmcso, lmcso, p, cX)
        @test isapprox(c_lmso, c_lmcso; atol = 1.0e-12, rtol = 1.0e-12)

        # Coordinate residual-space operator action should match operator-form action.
        y_lmso = zeros(n_res)
        Manopt.get_linear_operator!(M, y_lmso, lmso, p, X)
        y_lmcso = zeros(n_res)
        Manopt.add_linear_operator_coord!(M, y_lmcso, lmcso, p, cX)
        @test isapprox(y_lmso, y_lmcso; atol = 1.0e-12, rtol = 1.0e-12)

        # Symmetric system coordinate RHS is minus the coordinate normal vector field.
        rhs_slco = zeros(n)
        Manopt.get_vector_field!(M, rhs_slco, slco, p, B)
        @test isapprox(rhs_slco, -nvf_lmcso; atol = 1.0e-12, rtol = 1.0e-12)

        # Coordinate linear-system solution coefficients map back to the right tangent vector.
        dmp = DefaultManoptProblem(TpM, slco)
        cnss = Manopt.solve!(dmp, CoordinatesNormalSystemState(M, p; basis = B))
        X_sub = get_vector(M, p, cnss.c, B)
        @test isapprox(M, p, get_solver_result(dmp, cnss), X_sub; atol = 1.0e-12, rtol = 1.0e-12)
    end

    @testset "coordinate surrogate robustified high-damping regression" begin
        M = Euclidean(2)
        p = [0.7, -1.2]
        B = DefaultOrthonormalBasis()
        cX = [0.3, -0.5]
        X = get_vector(M, p, cX, B)
        penalty = 1.0e3

        for r in (CauchyRobustifier(), SoftL1Robustifier())
            vgf = VectorGradientFunction(
                F_reg_r2(ts_r2, xs_r2, ys_r2),
                jacF_reg_r2(ts_r2, xs_r2, ys_r2),
                length(ts_r2) * 2,
                function_type = FunctionVectorialType(),
                jacobian_type = CoefficientVectorialType(B),
            )
            # Build as a single block with one robustifier (not componentwise wrapping).
            nlso = NonlinearLeastSquaresObjective([vgf], [r])

            lmso = LevenbergMarquardtLinearSurrogateObjective(nlso; penalty = penalty)

            lmcso = Manopt.LevenbergMarquardtLinearSurrogateCoordinatesObjective(
                nlso;
                penalty = penalty,
                basis = B,
                jacobian_cache = [zeros(length(ts_r2) * 2, 2) for _ in eachindex(nlso.objective)],
                residuals = zeros(length(ts_r2) * 2),
            )

            # Coordinate surrogate requires explicit caches, which are normally updated in LM steps.
            get_residuals!(M, lmcso.value_cache, nlso, p)
            for (i, o) in enumerate(nlso.objective)
                lmcso.jacobian_cache[i] = get_jacobian(M, o, p; basis = B)
            end

            slso = Manopt.SymmetricLinearSystem(lmso)
            slco = Manopt.SymmetricLinearSystem(lmcso)

            n = number_of_coordinates(M, B)
            n_res = sum(length(o) for o in nlso.objective)

            A_lmso = zeros(n, n)
            A_lmcso = zeros(n, n)
            Manopt.get_linear_operator!(M, A_lmso, slso, p, B)
            Manopt.get_linear_operator!(M, A_lmcso, slco, p, B)
            @test isapprox(A_lmso, A_lmcso; atol = 1.0e-12, rtol = 1.0e-12)

            nvf_lmso = zeros(n)
            nvf_lmcso = zeros(n)
            Manopt.get_normal_vector_field!(M, nvf_lmso, lmso, p, B)
            Manopt.get_normal_vector_field_coord!(M, nvf_lmcso, lmcso, p, B)
            @test isapprox(nvf_lmso, nvf_lmcso; atol = 1.0e-12, rtol = 1.0e-12)

            vf_lmso = zeros(n_res)
            vf_lmcso = zeros(n_res)
            Manopt.get_vector_field!(M, vf_lmso, lmso, p)
            Manopt.get_vector_field!(M, vf_lmcso, lmcso, p)
            @test isapprox(vf_lmso, vf_lmcso; atol = 1.0e-12, rtol = 1.0e-12)

            TpM = TangentSpace(M, p)
            X0 = Manopt.ZeroTangentVector()
            @test isapprox(get_cost(TpM, slso, X0), get_cost(TpM, slco, X0); atol = 1.0e-12, rtol = 1.0e-12)

            # The LM-relevant regression: both surrogate systems should produce the same step.
            dmp_so = DefaultManoptProblem(TpM, slso)
            dmp_co = DefaultManoptProblem(TpM, slco)
            cnss_so = Manopt.solve!(dmp_so, CoordinatesNormalSystemState(M, p; basis = B))
            cnss_co = Manopt.solve!(dmp_co, CoordinatesNormalSystemState(M, p; basis = B))
            @test isapprox(cnss_so.c, cnss_co.c; atol = 1.0e-12, rtol = 1.0e-12)
            @test isapprox(
                M,
                p,
                get_solver_result(dmp_so, cnss_so),
                get_solver_result(dmp_co, cnss_co);
                atol = 1.0e-12,
                rtol = 1.0e-12,
            )
        end
    end

    @testset "Coordinate-based LM tests" begin
        struct TinyBAObservation
            camera_index::Int
            point_index::Int
            xy::Vector{Float64}
        end

        struct TinyBAResidual
            observations::Vector{TinyBAObservation}
            obs_idx::Int
        end

        struct TinyBAJacobian
            observations::Vector{TinyBAObservation}
            obs_idx::Int
        end

        project_tiny(R, t, X) = [-((R * X + t)[1]) / (R * X + t)[3], -((R * X + t)[2]) / (R * X + t)[3]]

        function (f::TinyBAResidual)(::AbstractManifold, r, p)
            obs = f.observations[f.obs_idx]
            p_cam, p_t, p_pt = p.x
            cam_idx = obs.camera_index
            pt_idx = obs.point_index
            pred = project_tiny(view(p_cam, :, :, cam_idx), view(p_t, :, cam_idx), view(p_pt, :, pt_idx))
            @inbounds begin
                r[1] = pred[1] - obs.xy[1]
                r[2] = pred[2] - obs.xy[2]
            end
            return r
        end

        function (f::TinyBAJacobian)(
                M::AbstractManifold,
                J::BlockNonzeroMatrix,
                p;
                basis_arg::AbstractBasis = DefaultOrthonormalBasis(),
            )
            obs = f.observations[f.obs_idx]
            cam_idx = obs.camera_index
            pt_idx = obs.point_index

            M_cam, M_t, M_pt = M.manifolds
            p_cam, p_t, p_pt = p.x

            d_cam = manifold_dimension(M_cam)
            d_t = manifold_dimension(M_t)

            col_cam = (cam_idx - 1) * 3 + 1
            col_t = d_cam + (cam_idx - 1) * 3 + 1
            col_p = d_cam + d_t + (pt_idx - 1) * 3 + 1

            row_starts = (1, 1, 1)
            col_starts = (col_cam, col_t, col_p)
            @test J.row_starts == row_starts
            @test J.col_starts == col_starts

            function residual_from_local_coords(cY_local)
                cY = zeros(eltype(cY_local), manifold_dimension(M))
                @views begin
                    cY[col_cam:(col_cam + 2)] .= cY_local[1:3]
                    cY[col_t:(col_t + 2)] .= cY_local[4:6]
                    cY[col_p:(col_p + 2)] .= cY_local[7:9]
                end

                Y = get_vector(M, p, cY, basis_arg)
                Y_cam, Y_t, Y_pt = Y.x

                R_new = exp(Rotations(3), p_cam[M_cam, cam_idx], Y_cam[M_cam, cam_idx])
                t_new = p_t[:, cam_idx] + Y_t[:, cam_idx]
                X_new = p_pt[:, pt_idx] + Y_pt[:, pt_idx]
                return project_tiny(R_new, t_new, X_new) - obs.xy
            end

            J_local = ForwardDiff.jacobian(residual_from_local_coords, zeros(9))

            J.blocks[1] .= view(J_local, :, 1:3)
            J.blocks[2] .= view(J_local, :, 4:6)
            J.blocks[3] .= view(J_local, :, 7:9)
            return J
        end

        function Manopt.allocate_jacobian(
                M::AbstractManifold,
                vgf::VectorGradientFunction{
                    InplaceEvaluation,
                    <:FunctionVectorialType,
                    <:CoefficientVectorialType,
                    <:TinyBAResidual,
                    <:TinyBAJacobian,
                },
                ::AbstractBasis = DefaultOrthonormalBasis();
                T::Type = Float64,
            )
            obs = vgf.jacobian!!.observations[vgf.jacobian!!.obs_idx]
            M_cam, M_t, _ = M.manifolds
            d_cam = manifold_dimension(M_cam)
            d_t = manifold_dimension(M_t)

            col_cam = (obs.camera_index - 1) * 3 + 1
            col_t = d_cam + (obs.camera_index - 1) * 3 + 1
            col_p = d_cam + d_t + (obs.point_index - 1) * 3 + 1

            return BlockNonzeroMatrix(
                vgf.range_dimension,
                manifold_dimension(M),
                (1, 1, 1),
                (col_cam, col_t, col_p),
                (
                    zeros(T, vgf.range_dimension, 3),
                    zeros(T, vgf.range_dimension, 3),
                    zeros(T, vgf.range_dimension, 3),
                ),
            )
        end

        n_cameras = 2
        n_points = 4
        Rot3 = Rotations(3)
        B_rot = DefaultOrthonormalBasis()

        R1 = Matrix{Float64}(I, 3, 3)
        R2 = exp(Rot3, R1, get_vector(Rot3, R1, [0.08, -0.05, 0.03], B_rot))
        t_true = [0.1 -0.15; -0.08 0.04; 0.02 -0.03]
        P_true = [0.4 -0.1 0.2 0.0; -0.2 0.1 0.3 -0.15; 3.0 3.3 2.8 3.5]
        R_true = cat(R1, R2; dims = 3)

        observations = TinyBAObservation[]
        for cam_idx in 1:n_cameras, pt_idx in 1:n_points
            xy = project_tiny(view(R_true, :, :, cam_idx), view(t_true, :, cam_idx), view(P_true, :, pt_idx))
            push!(observations, TinyBAObservation(cam_idx, pt_idx, xy))
        end

        M = ProductManifold(
            PowerManifold(Rot3, ArrayPowerRepresentation(), n_cameras),
            Euclidean(3, n_cameras),
            Euclidean(3, n_points),
        )

        R0 = cat(
            exp(Rot3, R1, get_vector(Rot3, R1, [0.03, -0.01, 0.0], B_rot)),
            exp(Rot3, R1, get_vector(Rot3, R1, [-0.02, 0.02, -0.01], B_rot));
            dims = 3,
        )
        t0 = t_true .+ [0.05 -0.03; -0.02 0.01; 0.04 -0.02]
        P0 = P_true .+ [0.08 -0.04 0.03 0.02; -0.05 0.02 -0.03 0.01; 0.15 -0.1 0.07 -0.08]
        p0 = ArrayPartition(R0, t0, P0)

        Fi = [TinyBAResidual(observations, i) for i in eachindex(observations)]
        Ji = [TinyBAJacobian(observations, i) for i in eachindex(observations)]

        Fs = [
            VectorGradientFunction(
                    Fi[i],
                    Ji[i],
                    2;
                    evaluation = InplaceEvaluation(),
                    function_type = FunctionVectorialType(),
                    jacobian_type = CoefficientVectorialType(DefaultOrthonormalBasis()),
                ) for i in eachindex(observations)
        ]

        robustifier = fill((1 / 20) ∘ HuberRobustifier(), length(Fs))
        nlso = NonlinearLeastSquaresObjective(Fs, robustifier)
        init_cost = get_cost(M, nlso, p0)

        A = spzeros(manifold_dimension(M), manifold_dimension(M))

        lms = LevenbergMarquardt(
            M,
            Fs,
            p0;
            initial_jacobian_f = [Manopt.allocate_jacobian(M, fi) for fi in Fs],
            robustifier = robustifier,
            stopping_criterion = StopAfterIteration(75) | StopWhenGradientNormLess(1.0e-11) | StopWhenStepsizeLess(1.0e-11),
            sub_state = CoordinatesNormalSystemState(M; A = A),
            use_fast_coordinate_system = true,
            return_state = true,
        )
        s = get_state(lms)

        @test s.sub_state isa CoordinatesNormalSystemState
        @test all(J isa BlockNonzeroMatrix for J in s.jacobian_f)
        @test get_cost(M, nlso, s.p) < init_cost
        @test norm(M, s.p, get_gradient(lms)) < 1.0e-5
    end

    @testset "WIP test" begin
        M = Euclidean(3)
        pts = [
            [0.0, 0.0, 1.0],
            [sqrt(0.19), 0.0, 0.9],
            [-1 / sqrt(2), 1 / sqrt(2), 0.0],
        ]
        p0 = [0.0, 0.0, 78.0]
        # We do a full function approach here

        struct Fib{TPI}
            p_i::TPI
        end
        (f::Fib)(::AbstractManifold, p) = distance(M, p, f.p_i)

        struct jacFi{TPI}
            p_i::TPI
        end
        function (f::jacFi)(M::AbstractManifold, Y, p)
            if distance(M, p, f.p_i) == 0
                fill!(Y, 0.0)
            else
                copyto!(Y, -log(M, p, f.p_i) / distance(M, p, f.p_i))
            end
            return Y
        end

        Fi = [ Fib(q) for q in pts]
        grad_Fi = [ jacFi(q) for q in pts]

        # Block s normal ones
        Fs = [
            VectorGradientFunction(
                    [Fi[i]], [grad_Fi[i]], 1;
                    evaluation = InplaceEvaluation(), function_type = ComponentVectorialType(),
                    jacobian_type = ComponentVectorialType()
                ) for i in eachindex(pts)
        ]
        robustifier = fill((1 / 30) ∘ HuberRobustifier(), length(Fs))
        nlso = NonlinearLeastSquaresObjective(Fs, robustifier)
        lmso = LevenbergMarquardtLinearSurrogateObjective(nlso)

        X = zero_vector(M, p0)
        Y = similar(X)
        Manopt.get_linear_normal_operator!(M, Y, lmso, p0, X)

        Manopt.get_vector_field!(M, Y, lmso, p0)
        initial_residuals = similar(X, sum(length(o) for o in get_objective(nlso).objective))

        sub_objective = Manopt.SymmetricLinearSystem(
            Manopt.LevenbergMarquardtLinearSurrogateObjective(nlso; residuals = copy(initial_residuals))
        )
        sub_problem = DefaultManoptProblem(TangentSpace(M, p0), sub_objective)

        # TODO: Continue here, this still needs a sub state.
        # lms = LevenbergMarquardtState(M, initial_residuals; sub_problem = sub_problem)
        #
        # Manopt.set_parameter!(lms.sub_problem, :Objective, :Penalty, 1.0)
    end
end
