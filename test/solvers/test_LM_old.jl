#
#
# TODO: These older tests are a bit involved and should be superseeded
# by the new ones, We keep them for a while to check back and not forget something.
#
using Manifolds, Manopt, Test, ManifoldsBase
using ForwardDiff
using LinearAlgebra
using RecursiveArrayTools
using SparseArrays

@testset "LevenbergMarquardt" begin
    # TODO: Rewrite the following two sets in an easier fashion?
    # The following block I am not sure what it aims to do or test? More than the accessors from before?
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
        nlso = ManifoldNonlinearLeastSquaresObjective(Fs, robustifier)
        init_cost = get_cost(M, nlso, p0)

        A = spzeros(manifold_dimension(M), manifold_dimension(M))

        lms = LevenbergMarquardt(
            M, Fs, p0;
            initial_jacobian_f = [Manopt.allocate_jacobian(M, fi) for fi in Fs],
            robustifier = robustifier,
            stopping_criterion = StopAfterIteration(75) | StopWhenGradientNormLess(1.0e-11) | StopWhenStepsizeLess(1.0e-11),
            sub_state = CoordinatesNormalSystemState(M; A = A),
            damping_reduction_factor = 0.3, damping_reduction_threshold = 0.5,
            use_unified_basis = true, return_state = true,
        )
        s = get_state(lms)

        @test s.sub_state isa CoordinatesNormalSystemState
        @test all(J isa BlockNonzeroMatrix for J in s.jacobian_f)
        @test get_cost(M, nlso, s.p) < init_cost
        @test norm(M, s.p, get_gradient(lms)) < 1.0e-5
    end
    # Is this a block approach? We also would need something like that for sure.
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
        nlso = ManifoldNonlinearLeastSquaresObjective(Fs, robustifier)
        lmso = LevenbergMarquardtLinearSurrogateObjective(nlso)

        X = zero_vector(M, p0)
        Y = similar(X)
        Manopt.get_normal_linear_operator!(M, Y, lmso, p0, X)

        Manopt.get_vector_field!(M, Y, lmso, p0)
        initial_residuals = similar(X, Manopt.residuals_count(nlso))

        sub_objective = Manopt.NormalEquationsObjective(
            Manopt.LevenbergMarquardtLinearSurrogateObjective(nlso; residuals = copy(initial_residuals))
        )
        sub_problem = DefaultManoptProblem(TangentSpace(M, p0), sub_objective)

        # TODO: Continue here, this still needs a sub state.
        # lms = LevenbergMarquardtState(M, initial_residuals; sub_problem = sub_problem)
        #
        # Manopt.set_parameter!(lms.sub_problem, :Objective, :Penalty, 1.0)
    end
end
