using Manopt, Manifolds, Test
using RecursiveArrayTools: ArrayPartition
using LinearAlgebra: I, eigvecs, tr, Diagonal, dot

mutable struct QuasiNewtonGradientDirectionUpdate{VT <: AbstractVectorTransportMethod} <:
    AbstractQuasiNewtonDirectionUpdate
    vector_transport_method::VT
    num_times_init::Int
end
function QuasiNewtonGradientDirectionUpdate(vtm::AbstractVectorTransportMethod)
    return QuasiNewtonGradientDirectionUpdate{typeof(vtm)}(vtm, 0)
end

function (d::QuasiNewtonGradientDirectionUpdate)(mp, st)
    return get_gradient(st)
end
function (d::QuasiNewtonGradientDirectionUpdate)(r, mp, st)
    r .= get_gradient(st)
    return r
end
function Manopt.initialize_update!(d::QuasiNewtonGradientDirectionUpdate)
    d.num_times_init += 1
    return d
end

struct QuasiNewtonTestDirectionUpdate{VT <: AbstractVectorTransportMethod} <:
    AbstractQuasiNewtonDirectionUpdate
    vector_transport_method::VT
end

@testset "Riemannian quasi-Newton Methods" begin
    @testset "Show & Status" begin
        M = Euclidean(4)
        qnu = InverseBFGS()
        d = QuasiNewtonMatrixDirectionUpdate(M, qnu)
        @test startswith(
            Manopt.status_summary(d), "A quasi Newton direction update stored as a matrix"
        )
        @test Manopt.status_summary(d; context = :inline) ==
            "A quasi Newton direction update using $(qnu), stored as a matrix."
        @test Manopt.status_summary(d; context = :short) == repr(d)
        s = "QuasiNewtonMatrixDirectionUpdate(M, InverseBFGS(), DefaultOrthonormalBasis(ℝ), [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]; initial_scale = 1.0, vector_transport_method = ParallelTransport())"
        @test repr(d) == s
        @test Manopt.get_message(d) == ""
        c = QuasiNewtonCautiousDirectionUpdate(d)
        @test startswith(
            Manopt.status_summary(c), "A cautious quasi Newton direction update"
        )
        @test startswith(
            Manopt.status_summary(c; context = :inline), "A cautious direction update with θ = "
        )
        @test startswith(repr(c), "QuasiNewtonCautiousDirectionUpdate with θ = ")
    end

    @testset "Mean of 3 Matrices" begin
        # Mean of 3 matrices
        A = [18.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
        B = [0.0 0.0 0.0 0.009; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0]
        C = [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; -5.0 0.0 0.0 0.0]
        ABC = [A, B, C]
        x_solution = mean(ABC)
        f(::Euclidean, x) = 0.5 * norm(A - x)^2 + 0.5 * norm(B - x)^2 + 0.5 * norm(C - x)^2
        grad_f(::Euclidean, x) = -A - B - C + 3 * x
        costgrad(M, p) = (f(M, p), grad_f(M, p))
        M = Euclidean(4, 4)
        p = zeros(Float64, 4, 4)
        x_lrbfgs = quasi_Newton(
            M, f, grad_f, p; stopping_criterion = StopWhenGradientNormLess(10^(-6))
        )
        @test norm(x_lrbfgs - x_solution) ≈ 0 atol = 10.0^(-14)
        # with State
        lrbfgs_s = quasi_Newton(
            M, f, grad_f, p;
            stopping_criterion = StopWhenGradientNormLess(10^(-6)), return_state = true, debug = [],
        )
        # Verify that Newton update direction works also allocating
        dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
        p_star = get_solver_result(lrbfgs_s)
        D = zero_vector(M, p_star)
        lrbfgs_s.direction_update(D, dmp, lrbfgs_s)
        @test isapprox(M, p_star, D, lrbfgs_s.direction_update(dmp, lrbfgs_s))

        @test startswith(
            Manopt.status_summary(lrbfgs_s; context = :default),
            "# Solver state for `Manopt.jl`s Quasi Newton Method\n"
        )
        @test startswith(repr(lrbfgs_s), "QuasiNewtonState(; ")
        @test get_last_stepsize(dmp, lrbfgs_s, lrbfgs_s.stepsize) > 0
        @test Manopt.get_iterate(lrbfgs_s) == x_lrbfgs
        set_gradient!(lrbfgs_s, M, p, grad_f(M, p))
        @test isapprox(M, p, Manopt.get_gradient(lrbfgs_s), grad_f(M, p))
        @test Manopt.get_message(lrbfgs_s) == ""
        # with Cached Basis
        x_lrbfgs_cached = quasi_Newton(
            M, f, grad_f, p;
            stopping_criterion = StopWhenGradientNormLess(10^(-6)),
            basis = get_basis(M, p, DefaultOrthonormalBasis()),
        )
        @test isapprox(M, x_lrbfgs_cached, x_lrbfgs)
        x_lrbfgs_cached_2 = quasi_Newton(
            M, f, grad_f, p;
            stopping_criterion = StopWhenGradientNormLess(10^(-6)),
            basis = get_basis(M, p, DefaultOrthonormalBasis()), memory_size = -1,
        )
        @test isapprox(M, x_lrbfgs_cached_2, x_lrbfgs; atol = 1.0e-5)
        # with Costgrad
        mcgo = ManifoldCostGradientObjective(costgrad)

        x_lrbfgs_costgrad = quasi_Newton(
            M, mcgo, p; stopping_criterion = StopWhenGradientNormLess(10^(-6)), debug = []
        )
        @test isapprox(M, x_lrbfgs_costgrad, x_lrbfgs; atol = 1.0e-5)

        clrbfgs_s = quasi_Newton(
            M, f, grad_f, p;
            cautious_update = true, stopping_criterion = StopWhenGradientNormLess(10^(-6)),
            return_state = true, debug = [],
        )
        # Test direction passthrough
        x_clrbfgs = get_solver_result(clrbfgs_s)
        D = zero_vector(M, x_clrbfgs)
        clrbfgs_s.direction_update(D, dmp, clrbfgs_s)
        @test isapprox(M, x_clrbfgs, D, clrbfgs_s.direction_update(dmp, clrbfgs_s))

        @test norm(x_clrbfgs - x_solution) ≈ 0 atol = 10.0^(-14)

        x_rbfgs_Huang = quasi_Newton(
            M, f, grad_f, p;
            memory_size = -1,
            stepsize = WolfePowellBinaryLinesearch(
                M;
                retraction_method = ExponentialRetraction(),
                vector_transport_method = ParallelTransport(),
            ),
            stopping_criterion = StopWhenGradientNormLess(10^(-6)),
        )
        @test norm(x_rbfgs_Huang - x_solution) ≈ 0 atol = 10.0^(-14)

        for T in [InverseBFGS(), BFGS(), InverseDFP(), DFP(), InverseSR1(), SR1()]
            for c in [true, false]
                x_state = quasi_Newton(
                    M, f, grad_f, p;
                    direction_update = T, cautious_update = c, memory_size = -1,
                    stopping_criterion = StopWhenGradientNormLess(10^(-12)),
                    return_state = true, debug = [],
                )
                x_direction = get_solver_result(x_state)
                D = zero_vector(M, x_direction)
                x_state.direction_update(D, dmp, x_state)
                @test isapprox(M, x_direction, D, x_state.direction_update(dmp, x_state))
                @test norm(x_direction - x_solution) ≈ 0 atol = 10.0^(-14)
            end
        end
        tdu = QuasiNewtonTestDirectionUpdate(ParallelTransport())
        @test Manopt.initialize_update!(tdu) === tdu
    end

    @testset "Rayleigh Quotient Minimization" begin
        n = 4
        rayleigh_atol = 1.0e-7
        A = [2.0 1.0 0.0 3.0; 1.0 3.0 4.0 5.0; 0.0 4.0 3.0 2.0; 3.0 5.0 2.0 6.0]
        A = (A + A') / 2
        M = Sphere(n - 1)
        f(::Sphere, X) = X' * A * X
        grad_f(::Sphere, X) = 2 * (A * X - X * (X' * A * X))
        grad_f!(::Sphere, Y, X) = (Y .= 2 * (A * X - X * (X' * A * X)))
        x_solution = eigvecs(A)[:, 1]

        x = Matrix{Float64}(I, n, n)[n, :]
        x_lrbfgs = quasi_Newton(M, f, grad_f, x; memory_size = -1)
        @test isapprox(M, x_lrbfgs, x_solution; atol = rayleigh_atol)
        x_lrbfgs2 = copy(M, x)
        quasi_Newton!(
            M, f, grad_f!, x_lrbfgs2; evaluation = InplaceEvaluation(), memory_size = -1
        )
        @test isapprox(M, x_lrbfgs2, x_lrbfgs)

        # A simple preconditioner
        x_lrbfgs = quasi_Newton(
            M, f, grad_f, x; memory_size = -1, preconditioner = (M, p, X) -> 0.5 .* X
        )
        @test isapprox(M, x_lrbfgs, x_solution; atol = rayleigh_atol)

        # An in-place preconditioner
        x_lrbfgs = quasi_Newton(
            M, f, grad_f, x; memory_size = -1,
            preconditioner = QuasiNewtonPreconditioner(
                (M, Y, p, X) -> (Y .= 0.5 .* X); evaluation = InplaceEvaluation()
            ),
        )
        @test isapprox(M, x_lrbfgs, x_solution; atol = rayleigh_atol)

        @testset "Preconditioner with a finite memory" begin
            Me = Euclidean(2)
            fe(M, p) = 0.5 * (4 * p[1]^2 + p[2]^2)
            grad_fe(M, p) = [4 * p[1], p[2]]
            # with an empty memory the first step is the preconditioned gradient step
            q = quasi_Newton(
                Me, fe, grad_fe, [1.0, 2.0];
                memory_size = 2, preconditioner = (M, p, X) -> 0.5 .* X,
                stepsize = ConstantLength(1.0), stopping_criterion = StopAfterIteration(1),
            )
            @test q ≈ [-1.0, 1.0]
        end

        x_clrbfgs = quasi_Newton(M, f, grad_f, x; cautious_update = true)
        @test isapprox(M, x_clrbfgs, x_solution; atol = rayleigh_atol)

        for T in [
                    InverseDFP(), DFP(), Broyden(0.5), InverseBroyden(0.5),
                    Broyden(0.5, :Davidon), InverseBroyden(0.5, :InverseDavidon), InverseBFGS(), BFGS(),
                ],
                c in [true, false]
            x_direction = quasi_Newton(
                M, f, grad_f, x; direction_update = T, cautious_update = c, memory_size = -1
            )
            @test isapprox(M, x_direction, x_solution; atol = rayleigh_atol)
        end

        @testset "Byrd's nonpositive rule" begin
            x1 = quasi_Newton(M, f, grad_f, x; nonpositive_curvature_behavior = :byrd, sy_tol = 1.0e8)
            @test isapprox(M, x1, x_solution; atol = rayleigh_atol)
        end
    end

    @testset "Brocket" begin
        struct GradF
            A::Matrix{Float64}
            N::Diagonal{Float64, Vector{Float64}}
        end
        function (gradF::GradF)(::Stiefel, X::Array{Float64, 2})
            AX = gradF.A * X
            XpAX = X' * AX
            return 2 .* AX * gradF.N .- X * XpAX * gradF.N .- X * gradF.N * XpAX
        end

        n = 4
        k = 2
        M = Stiefel(n, k)
        A = [2.0 1.0 0.0 3.0; 1.0 3.0 4.0 5.0; 0.0 4.0 3.0 2.0; 3.0 5.0 2.0 6.0]
        f(::Stiefel, X) = tr((X' * A * X) * Diagonal(k:-1:1))
        grad_f = GradF(A, Diagonal(Float64.(collect(k:-1:1))))

        x = Matrix{Float64}(I, n, n)[:, 2:(k + 1)]
        x_inverseBFGSCautious = quasi_Newton(
            M, f, grad_f, x; memory_size = 8,
            vector_transport_method = ProjectionTransport(), retraction_method = QRRetraction(),
            cautious_update = true, stopping_criterion = StopWhenGradientNormLess(1.0e-6) | StopAfterIteration(100),
        )

        x_inverseBFGSHuang = quasi_Newton(
            M, f, grad_f, x; memory_size = 8,
            stepsize = WolfePowellBinaryLinesearch(
                M; retraction_method = QRRetraction(), vector_transport_method = ProjectionTransport(),
            ),
            vector_transport_method = ProjectionTransport(), retraction_method = QRRetraction(),
            cautious_update = true, stopping_criterion = StopWhenGradientNormLess(1.0e-6) | StopAfterIteration(100),
        )
        @test isapprox(M, x_inverseBFGSCautious, x_inverseBFGSHuang; atol = 2.0e-4)
    end

    @testset "Wolfe Powell line search" begin
        n = 4
        rayleigh_atol = 1.0e-8
        A = [2.0 1.0 0.0 3.0; 1.0 3.0 4.0 5.0; 0.0 4.0 3.0 2.0; 3.0 5.0 2.0 6.0]
        A = (A + A') / 2
        M = Sphere(n - 1)
        F(::Sphere, X) = X' * A * X
        grad_f(::Sphere, X) = 2 * (A * X - X * (X' * A * X))
        x_solution = abs.(eigvecs(A)[:, 1])

        x = [0.7011245948687502, -0.1726003159556036, 0.38798265967671103, -0.5728026616491424]
        x_lrbfgs = quasi_Newton(
            M, F, grad_f, x;
            basis = get_basis(M, x, DefaultOrthonormalBasis()),
            memory_size = -1,
            stopping_criterion = StopWhenGradientNormLess(1.0e-9) | StopAfterIteration(1000),
        )
        @test norm(abs.(x_lrbfgs) - x_solution) ≈ 0 atol = rayleigh_atol
    end

    @testset "update rules" begin
        # the stabilization parameter takes any real number
        @test SR1(1).r === 1.0
        @test InverseSR1(1 // 2).r === 0.5
        n = 4
        A = [2.0 1.0 0.0 3.0; 1.0 3.0 4.0 5.0; 0.0 4.0 3.0 2.0; 3.0 5.0 2.0 6.0]
        A = (A + A') / 2
        M = Sphere(n - 1)
        F(::Sphere, X) = X' * A * X
        grad_f(::Sphere, X) = 2 * (A * X - X * (X' * A * X))
        grad_f!(::Sphere, X, p) = (X .= 2 * (A * p - p * (p' * A * p)))

        p_1 = [1.0; 0.0; 0.0; 0.0]
        p_2 = [0.0; 0.0; 1.0; 0.0]

        SR1_allocating = ApproxHessianSymmetricRankOne(
            M, copy(M, p_1), grad_f; evaluation = AllocatingEvaluation()
        )

        SR1_inplace = ApproxHessianSymmetricRankOne(
            M, copy(M, p_1), grad_f!; evaluation = InplaceEvaluation()
        )

        BFGS_allocating = ApproxHessianBFGS(
            M, copy(M, p_1), grad_f; evaluation = AllocatingEvaluation()
        )

        BFGS_inplace = ApproxHessianBFGS(
            M, copy(M, p_1), grad_f!; evaluation = InplaceEvaluation()
        )

        Y = [0.0; 1.0; 0.0; 0.0]
        X_1 = SR1_allocating(M, p_1, Y)
        SR1_allocating.p_tmp .= p_2
        X_2 = SR1_allocating(M, p_1, Y)
        @test isapprox(M, p_1, X_1, X_2; atol = 1.0e-10)
        update_hessian_basis!(M, SR1_allocating, p_1)
        update_hessian_basis!(M, SR1_allocating, p_2)

        X_3 = zero_vector(M, p_1)
        X_4 = zero_vector(M, p_1)
        SR1_inplace(M, X_3, p_1, Y)
        SR1_inplace.p_tmp .= p_2
        SR1_inplace(M, X_4, p_1, Y)
        @test isapprox(M, p_1, X_3, X_4; atol = 1.0e-10)
        update_hessian_basis!(M, SR1_inplace, p_1)
        update_hessian_basis!(M, SR1_inplace, p_2)

        X_5 = BFGS_allocating(M, p_1, Y)
        X_6 = BFGS_allocating(M, p_2, Y)
        @test isapprox(M, p_1, X_5, X_6; atol = 1.0e-10)
        update_hessian_basis!(M, BFGS_allocating, p_1)
        update_hessian_basis!(M, BFGS_allocating, p_2)

        X_7 = zero_vector(M, p_1)
        X_8 = zero_vector(M, p_1)
        BFGS_inplace(M, X_7, p_1, Y)
        BFGS_inplace(M, X_8, p_2, Y)
        update_hessian_basis!(M, BFGS_inplace, p_1)
        update_hessian_basis!(M, BFGS_inplace, p_2)

        @test isapprox(M, p_1, X_7, X_8; atol = 1.0e-10)

        BFGS_allocating.grad_tmp = ones(4)
        BFGS_allocating.matrix = one(zeros(3, 3))
        Manopt.update_hessian!(M, BFGS_allocating, p_1, p_2, Y)
        test_m = [
            7.0 -1.0 3.0
            -1.0 1.1428571428571428 -0.42857142857142855
            3.0 -0.42857142857142855 2.2857142857142856
        ]
        @test isapprox(test_m, BFGS_allocating.matrix)

        update_hessian_basis!(M, BFGS_allocating, p_1)
        update_hessian_basis!(M, BFGS_allocating, p_2)
        @test isapprox(M, p_1, BFGS_allocating.grad_tmp, [0.0, 8.0, 0.0, 4.0])
    end

    @testset "A small complex example in Tutorial Mode" begin
        M = Euclidean(2; field = ℂ)
        A = [2 im; -im 2]
        fc(::Euclidean, p) = real(p' * A * p)
        grad_fc(::Euclidean, p) = 2 * A * p
        p0 = [2.0, 1 + im]
        @test_logs (:info,) Manopt.set_parameter!(:Mode, "Tutorial")
        p4 = quasi_Newton(M, fc, grad_fc, p0; stopping_criterion = StopAfterIteration(3))
        @test_logs (:info,) Manopt.set_parameter!(:Mode, "")
        @test fc(M, p4) ≤ fc(M, p0)
    end

    @testset "Boundary cases and safeguards" begin
        M = Euclidean(2)
        p = [0.0, 0.0]
        f(M, p) = sum(p .^ 2)
        grad_f(M, p) = 2 .* p
        gmp = ManifoldGradientObjective(f, grad_f)
        mp = DefaultManoptProblem(M, gmp)
        qns = QuasiNewtonState(M; p = p)
        @test qns.direction_update.initial_scale == 1.0 # default scaling active without preconditioner
        # push zeros to memory
        qns.yk = copy(p)
        qns.sk = copy(p)
        update_hessian!(qns.direction_update, mp, qns, p, 1)
        update_hessian!(qns.direction_update, mp, qns, p, 2)
        @test contains(qns.direction_update.message, "i=2,1,1")
        # get_message must surface the direction-update message (was dropped before)
        @test contains(Manopt.get_message(qns), "i=2,1,1")
        # and it also has to survive the cautious and the box wrapper
        cdu = QuasiNewtonCautiousDirectionUpdate(qns.direction_update)
        @test contains(Manopt.get_message(cdu), "i=2,1,1")
        bdu = QuasiNewtonLimitedMemoryBoxDirectionUpdate(qns.direction_update)
        @test contains(Manopt.get_message(bdu), "i=2,1,1")
        qns.direction_update(mp, qns)
        # Update (1) says at i=1 inner products are zero (2) all are zero -> gradient proposal
        @test contains(qns.direction_update.message, "gradient")
    end

    @testset "Broken direction update" begin
        M = Euclidean(2)
        p = [0.0, 1.0]
        f(M, p) = sum(p .^ 2)
        # A wrong gradient
        grad_f(M, p) = -2 .* p
        gmp = ManifoldGradientObjective(f, grad_f)
        mp = DefaultManoptProblem(M, gmp)
        qns = QuasiNewtonState(
            M; p = copy(M, p),
            direction_update = QuasiNewtonGradientDirectionUpdate(ParallelTransport()),
            nondescent_direction_behavior = :step_towards_negative_gradient,
        )
        dqns = DebugSolverState(qns, DebugMessages(:Warning, :Once))
        @test_logs (
            :warn,
            r"Computed direction is not a descent direction\. The inner product evaluated to 4\.0\. Resetting to negative gradient\.",
        ) (
            :warn,
            "Further warnings will be suppressed, use DebugMessages(:Warning, :Always) to get all warnings.",
        ) solve!(mp, dqns)

        qns = QuasiNewtonState(
            M; p = copy(M, p),
            direction_update = QuasiNewtonGradientDirectionUpdate(ParallelTransport()),
            nondescent_direction_behavior = :step_towards_negative_gradient,
        )

        @test_nowarn solve!(mp, qns)
        @test qns.direction_update.num_times_init == 1

        qns = QuasiNewtonState(
            M; p = copy(M, p),
            direction_update = QuasiNewtonGradientDirectionUpdate(ParallelTransport()),
            nondescent_direction_behavior = :reinitialize_direction_update,
        )

        @test_nowarn solve!(mp, qns)
        @test qns.direction_update.num_times_init == 1001
    end

    @testset "A Circle example" begin
        M = Circle()
        data = fill.([-π / 2, π / 4, 0.0, π / 4])
        pstar = sum([-π / 2, π / 4, 0.0, π / 4]) / length(data)
        f(M, p) = 1 / 10 * sum(distance.(Ref(M), data, Ref(p)) .^ 2)
        grad_f(M, p) = 1 / 5 * sum(-log.(Ref(M), Ref(p), data))
        p = quasi_Newton(M, f, grad_f, data[1])
        @test isapprox(M, pstar, p)
        s = quasi_Newton(M, f, grad_f, data[1]; return_state = true)
        @test get_solver_result(s)[] == p[]
    end

    @testset "Cautious Update that does only transport" begin
        M = Euclidean(2)
        p = [0.0, 1.0]
        f(M, p) = sum(p .^ 2)
        # A wrong gradient
        grad_f(M, p) = -2 .* p
        gmp = ManifoldGradientObjective(f, grad_f)
        mp = DefaultManoptProblem(M, gmp)
        # the parallel transport on `Euclidean` is isometric, so the re-test is off by default
        @test !QuasiNewtonCautiousDirectionUpdate(
            QuasiNewtonLimitedMemoryDirectionUpdate(M, p, InverseBFGS(), 2)
        ).retest
        for retest in (false, true)
            qdu = QuasiNewtonLimitedMemoryDirectionUpdate(M, p, InverseBFGS(), 2)
            qns = QuasiNewtonState(
                M; p = copy(M, p),
                direction_update = QuasiNewtonCautiousDirectionUpdate(qdu; retest = retest),
            )
            # current bound with the gradient is 2, so we choose an sk larger than that
            qns.sk = [1.0, 1.1]
            # push two dummy pairs we can transport, with ⟨s, y⟩ / ‖s‖² = 11 / 25 and 3
            push!(qdu.memory_y, [1.0, 2.0])
            push!(qdu.memory_s, [3.0, 4.0])
            push!(qdu.memory_y, [3.0, 0.0])
            push!(qdu.memory_s, [1.0, 0.0])
            # This triggers a cautious update that does not update the Hessian
            Manopt.update_hessian!(qns.direction_update, mp, qns, p, 1)
            # the stored pairs are only transported: no new pair is added and ρ is recomputed;
            # tested against the bound 2, the first pair is removed
            if retest
                @test length(qdu.memory_s) == 1
                @test qdu.memory_s[1] == [1.0, 0.0]
                @test qdu.ρ[1] ≈ 1 / 3
            else
                @test length(qdu.memory_s) == 2
                @test qdu.ρ ≈ [1 / 11, 1 / 3]
            end
            @test startswith(repr(qdu), "QuasiNewtonLimitedMemoryDirectionUpdate with memory size")
        end
    end
    @testset "Cautious update re-tests pairs after a non-isometric transport" begin
        M = Sphere(2)
        mp = DefaultManoptProblem(
            M, ManifoldGradientObjective((M, p) -> 0.0, (M, p) -> zero_vector(M, p))
        )
        p_old = [1.0, 0.0, 0.0]
        p_new = [1.0, 1.0, 0.0] ./ sqrt(2)
        # ⟨s, y⟩ / ‖s‖² is 1/2 at p_old, but 1/3 after the projection onto T_{p_new}M,
        # while ⟨s, y⟩ = 1/2 stays positive
        s_bad, y_bad = [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]
        # tangent at both points, so the projection does not change it
        s_good, y_good = [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]
        for (memory_size, add_new) in Iterators.product((2, 3), (true, false))
            qdu = QuasiNewtonLimitedMemoryDirectionUpdate(
                M, p_old, InverseBFGS(), memory_size;
                vector_transport_method = ProjectionTransport(),
                nonpositive_curvature_behavior = :byrd,
            )
            dc = QuasiNewtonCautiousDirectionUpdate(qdu; θ = x -> 0.4)
            st = QuasiNewtonState(M; p = copy(M, p_new), direction_update = dc)
            push!(qdu.memory_s, copy(s_good))
            push!(qdu.memory_y, copy(y_good))
            push!(qdu.memory_s, copy(s_bad))
            push!(qdu.memory_y, copy(y_bad))
            # the new pair has ⟨s, y⟩ / ‖s‖² = 4 or 0
            st.sk .= [0.0, 0.0, 1.0]
            st.yk .= add_new ? [0.0, 0.0, 4.0] : zero_vector(M, p_new)
            Manopt.update_hessian!(dc, mp, st, p_old, 1)
            # the first pair is kept, the second one is removed, and the new one is added
            # if it satisfies the decision rule, without removing the first one
            @test length(qdu.memory_s) == (add_new ? 2 : 1)
            @test qdu.memory_s[1] == s_good
            @test qdu.ρ[1] ≈ 1 / 2
            add_new && @test qdu.ρ[2] ≈ 1 / 4
        end
        # the projection transport is not isometric, so the re-test is on by default;
        # without it, the second pair is kept, with ⟨s, y⟩ = 1/2 after the projection
        qdu = QuasiNewtonLimitedMemoryDirectionUpdate(
            M, p_old, InverseBFGS(), 3;
            vector_transport_method = ProjectionTransport(),
            nonpositive_curvature_behavior = :byrd,
        )
        @test QuasiNewtonCautiousDirectionUpdate(qdu).retest
        dc = QuasiNewtonCautiousDirectionUpdate(qdu; θ = x -> 0.4, retest = false)
        st = QuasiNewtonState(M; p = copy(M, p_new), direction_update = dc)
        push!(qdu.memory_s, copy(s_good))
        push!(qdu.memory_y, copy(y_good))
        push!(qdu.memory_s, copy(s_bad))
        push!(qdu.memory_y, copy(y_bad))
        st.sk .= [0.0, 0.0, 1.0]
        st.yk .= zero_vector(M, p_new)
        Manopt.update_hessian!(dc, mp, st, p_old, 1)
        @test length(qdu.memory_s) == 2
        @test qdu.ρ[1:2] ≈ [1 / 2, 2]
        # the default follows the components of a product vector transport, also in `quasi_Newton`
        N = Sphere(2) × Euclidean(2)
        pN = ArrayPartition(p_old, [0.0, 0.0])
        fN(N, q) = sum(abs2, submanifold_component(N, q, 2) .- 1)
        function grad_fN(N, q)
            return ArrayPartition(
                zero_vector(Sphere(2), submanifold_component(N, q, 1)),
                2 .* (submanifold_component(N, q, 2) .- 1),
            )
        end
        for (vtm, retest) in (
                (ProductVectorTransport(ParallelTransport(), ParallelTransport()), false),
                (ProductVectorTransport(ProjectionTransport(), ParallelTransport()), true),
            )
            qdu = QuasiNewtonLimitedMemoryDirectionUpdate(
                N, pN, InverseBFGS(), 2; vector_transport_method = vtm
            )
            @test QuasiNewtonCautiousDirectionUpdate(qdu).retest == retest
            for cautious_retest in (retest, !retest)
                kw = cautious_retest == retest ? (;) : (; cautious_retest = cautious_retest)
                st = quasi_Newton(
                    N, fN, grad_fN, pN;
                    cautious_update = true, vector_transport_method = vtm,
                    stopping_criterion = StopAfterIteration(1), return_state = true, kw...,
                )
                @test Manopt.get_state(st).direction_update.retest == cautious_retest
            end
        end
        @test occursin("* retest: ", Manopt.status_summary(dc))
        # a new pair that satisfies the decision rule but not the curvature condition of
        # `:byrd` is not added either, and the oldest pair is kept
        qdu = QuasiNewtonLimitedMemoryDirectionUpdate(
            M, p_old, InverseBFGS(), 1;
            vector_transport_method = ProjectionTransport(),
            nonpositive_curvature_behavior = :byrd, sy_tol = 1.0,
        )
        dc = QuasiNewtonCautiousDirectionUpdate(qdu; θ = x -> 0.4)
        st = QuasiNewtonState(M; p = copy(M, p_new), direction_update = dc)
        push!(qdu.memory_s, copy(s_good))
        push!(qdu.memory_y, [0.0, 0.0, 0.5])
        st.sk .= [0.0, 0.0, 1.0]
        st.yk .= [0.0, 0.0, 4.0]
        Manopt.update_hessian!(dc, mp, st, p_old, 1)
        @test length(qdu.memory_s) == 1
        @test qdu.memory_y[1] == [0.0, 0.0, 0.5]
    end
    @testset "Cautious skip transports the basis" begin
        M = Sphere(2)
        mp = DefaultManoptProblem(
            M, ManifoldGradientObjective((M, p) -> 0.0, (M, p) -> 10.0 .* [0.0, 0.0, 1.0])
        )
        p_old = [1.0, 0.0, 0.0]
        p_new = [0.0, 1.0, 0.0]
        b = get_basis(M, p_old, DefaultOrthonormalBasis())
        d = QuasiNewtonMatrixDirectionUpdate(M, InverseBFGS(), b, Matrix{Float64}(I, 2, 2))
        dc = QuasiNewtonCautiousDirectionUpdate(d; θ = identity)
        st = QuasiNewtonState(M; p = copy(p_new), direction_update = dc)
        st.sk .= [0.0, 0.0, 1.0]
        st.yk .= [0.0, 0.0, 1.0]
        # the bound is 10, the ratio 1, so the update is skipped, but the basis moves along
        Manopt.update_hessian!(dc, mp, st, p_old, 2)
        @test all(is_vector(M, p_new, X) for X in d.basis.data)
    end
    @testset "Removing zero rho vectors" begin
        M = Euclidean(2)
        p = [0.0, 1.0]
        f(M, p) = sum(p .^ 2)
        # A wrong gradient
        grad_f(M, p) = -2 .* p
        gmp = ManifoldGradientObjective(f, grad_f)
        mp = DefaultManoptProblem(M, gmp)
        qdu = QuasiNewtonLimitedMemoryDirectionUpdate(M, p, InverseBFGS(), 3)
        # push three pairs; middle one has zero inner product
        push!(qdu.memory_y, [1, 0])
        push!(qdu.memory_s, [1, 0])

        push!(qdu.memory_y, [1, 0])
        push!(qdu.memory_s, [0, 1])

        push!(qdu.memory_y, [0, 2])
        push!(qdu.memory_s, [0, 2])
        qdu.ρ = [1.0, 0.0, 4.0]
        # delete the zero inner product pair and check that the removal was correct
        Manopt._drop_zero_rho_vectors!(qdu)
        @test length(qdu.memory_y) == 2
        @test length(qdu.memory_s) == 2
        @test qdu.ρ[[1, 2]] == [1.0, 4.0]
        @test qdu.memory_y[1] == [1, 0]
        @test qdu.memory_y[2] == [0, 2]
    end

    @testset "reforming_required + (start == 2)" begin
        M = Euclidean(2)
        p = [0.0, 0.0]
        f(M, p) = sum(p .^ 2)
        grad_f(M, p) = 2 .* p
        gmp = ManifoldGradientObjective(f, grad_f)
        mp = DefaultManoptProblem(M, gmp)
        ha = QuasiNewtonLimitedMemoryDirectionUpdate(M, p, InverseBFGS(), 2; nonpositive_curvature_behavior = :byrd)
        qns = QuasiNewtonState(M; p = p, nonpositive_curvature_behavior = :byrd, direction_update = ha)

        qns.yk = [1.0, 1.0]
        qns.sk = [1.0, 2.0]
        update_hessian!(qns.direction_update, mp, qns, p, 1)

        qns.yk = [2.0, 1.0]
        qns.sk = [1.0, 2.0]
        update_hessian!(qns.direction_update, mp, qns, p, 2)
        ha.memory_s[2] = [0.0, 0.0]  # force reforming_required in next step
        update_hessian!(qns.direction_update, mp, qns, p, 3)
        # test that the zeroes out pair was replaced
        @test qns.direction_update.memory_s[1] == [1.0, 2.0]
        @test qns.direction_update.memory_s[2] == [1.0, 2.0]
    end
    @testset "get_cost specialization" begin
        M = Euclidean(2)
        p = [0.0, 1.0]
        f(M, p) = sum(p .^ 2)
        grad_f(M, p) = 2 .* p
        gmp = ManifoldGradientObjective(f, grad_f)
        mp = DefaultManoptProblem(M, gmp)
        ha = QuasiNewtonLimitedMemoryDirectionUpdate(M, p, InverseBFGS(), 2; nonpositive_curvature_behavior = :byrd)
        qns = QuasiNewtonState(
            M;
            p = copy(M, p),
            direction_update = ha,
            nondescent_direction_behavior = :step_towards_negative_gradient,
            stepsize = HagerZhangLinesearch()(M),
        )
        @test get_cost(mp, qns) == f(M, get_iterate(qns))
        solve!(mp, qns)
        @test get_cost(mp, qns) == f(M, get_iterate(qns))

        @testset "get_cost with DebugSolverState" begin
            dqns = DebugSolverState(qns, DebugMessages(:Info, :Always))
            @test get_cost(mp, dqns) == f(M, get_iterate(dqns))
        end
    end
    @testset "Vanishing step at the minimizer" begin
        # the minimizer is reached before the stopping criterion fires, so the step becomes zero
        N = Euclidean(3)
        c = [0.3, 0.5, 0.2]
        fq(M, p) = 0.5 * sum(abs2, p - c)
        grad_fq(M, p) = p - c
        for memory_size in [3, -1]
            q = quasi_Newton(
                N, fq, grad_fq, zeros(3);
                memory_size = memory_size, stopping_criterion = StopAfterIteration(6),
            )
            @test isapprox(N, q, c)
        end
    end
    @testset "initial_scale acts once on the matrix update" begin
        N = Euclidean(3)
        p = zeros(3)
        d = QuasiNewtonMatrixDirectionUpdate(N, InverseBFGS(); initial_scale = 2.0)
        Manopt.initialize_update!(d)
        @test d.matrix == [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        # after the first update the secant equation holds and the scale enters linearly
        fq(M, q) = 0.5 * sum(abs2, q)
        grad_fq(M, q) = q
        dmp = DefaultManoptProblem(N, ManifoldGradientObjective(fq, grad_fq))
        qns = QuasiNewtonState(N; p = [1.0, 0.0, 0.0], direction_update = d)
        qns.sk .= [1.0, 0.0, 0.0]
        qns.yk .= [2.0, 0.0, 0.0]
        Manopt.update_hessian!(d, dmp, qns, p, 1)
        # scale ⟨s,y⟩/‖y‖² = 2 * 2/4 = 1 on the complement, s/y = 1/2 along s
        @test d.matrix ≈ [0.5 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    end
    @testset "One gradient evaluation at the start" begin
        Ne = Euclidean(2)
        fe(M, q) = sum(abs2, q)
        grad_fe(M, q) = 2q
        r = quasi_Newton(Ne, fe, grad_fe, [1.0, 1.0]; count = [:Gradient], return_objective = true, stopping_criterion = StopAfterIteration(0))
        @test get_count(r[1], :Gradient) == 1
    end
    @testset "The cautious update evaluates no additional gradients" begin
        Ne = Euclidean(2)
        fe(M, q) = sum(abs2, q) + q[1]^4
        grad_fe(M, q) = 2q + [4 * q[1]^3, 0.0]
        for (cautious_update, memory_size) in Iterators.product((false, true), (2, -1))
            r = quasi_Newton(
                Ne, fe, grad_fe, [1.0, 1.0];
                cautious_update = cautious_update, memory_size = memory_size,
                stepsize = Manopt.ConstantStepsize(Ne, 0.1),
                count = [:Gradient], return_objective = true, return_state = true,
                stopping_criterion = StopAfterIteration(5),
            )
            # one evaluation at the start and one per iteration
            @test get_count(r[1], :Gradient) == 6
        end
        # the decision rule uses the gradient norm at the point the direction was computed at,
        # which is the projected gradient norm without box constraints
        mp = DefaultManoptProblem(Ne, ManifoldGradientObjective(fe, grad_fe))
        p = [1.0, 1.0]
        dc = QuasiNewtonCautiousDirectionUpdate(
            QuasiNewtonLimitedMemoryDirectionUpdate(Ne, p, InverseBFGS(), 2); θ = identity
        )
        @test isnan(dc.projected_gradient_norm)
        st = QuasiNewtonState(Ne; p = copy(p), X = grad_fe(Ne, p), direction_update = dc)
        dc(mp, st)
        @test dc.projected_gradient_norm ≈ norm(grad_fe(Ne, p))
        # the recorded norm is used once; without it the gradient at `p_old` is evaluated
        st.sk = [1.0, 0.0]
        st.yk = [10.0, 0.0]
        update_hessian!(dc, mp, st, p, 1)
        @test isnan(dc.projected_gradient_norm)
        @test length(dc.update.memory_s) == 1
        dc.projected_gradient_norm = 20.0 # pretend a large gradient was recorded: the pair is skipped
        update_hessian!(dc, mp, st, p, 2)
        @test length(dc.update.memory_s) == 1
        update_hessian!(dc, mp, st, p, 3) # evaluates ‖grad f(p)‖ ≈ 6.3 < 10: added
        @test length(dc.update.memory_s) == 2
        # the type of the recorded norm follows the numbers stored in the wrapped update
        @test dc.projected_gradient_norm isa Float64
        p32 = Float32[1.0, 1.0]
        for u in (
                QuasiNewtonLimitedMemoryDirectionUpdate(Ne, p32, InverseBFGS(), 2; initial_scale = 1.0f0),
                QuasiNewtonMatrixDirectionUpdate(Ne, InverseBFGS(), DefaultOrthonormalBasis(), Matrix{Float32}(I, 2, 2)),
                QuasiNewtonLimitedMemoryBoxDirectionUpdate(
                    QuasiNewtonLimitedMemoryDirectionUpdate(
                        Hyperrectangle(Float32[0, 0], Float32[2, 2]), p32, InverseBFGS(), 2; initial_scale = 1.0f0
                    )
                ),
            )
            dc32 = QuasiNewtonCautiousDirectionUpdate(u)
            @test dc32.projected_gradient_norm isa Float32
            @test isnan(dc32.projected_gradient_norm)
        end
        dc32 = QuasiNewtonCautiousDirectionUpdate(
            QuasiNewtonLimitedMemoryDirectionUpdate(Ne, p32, InverseBFGS(), 2; initial_scale = 1.0f0)
        )
        st32 = QuasiNewtonState(Ne; p = copy(p32), X = Float32[2.0, 0.0], direction_update = dc32)
        dc32(mp, st32)
        @test dc32.projected_gradient_norm === 2.0f0
        # reinitializing the update discards the recorded norm
        Manopt.initialize_update!(dc32)
        @test isnan(dc32.projected_gradient_norm)
        @test dc32.projected_gradient_norm isa Float32
    end
    @testset "Byrd's rule on a complex manifold" begin
        Mc = Euclidean(2; field = ℂ)
        Ac = [2 im; -im 2]
        fc(::Euclidean, q) = real(q' * Ac * q)
        grad_fc(::Euclidean, q) = 2 * Ac * q
        pc = ComplexF64[2.0, 1 + im]
        sc = StopAfterIteration(3)
        qi = quasi_Newton(Mc, fc, grad_fc, pc; nonpositive_curvature_behavior = :ignore, stopping_criterion = sc)
        qb = quasi_Newton(Mc, fc, grad_fc, pc; nonpositive_curvature_behavior = :byrd, stopping_criterion = sc)
        @test isapprox(Mc, qi, qb)
    end
    @testset "The locking condition uses the transport of the direction update" begin
        S = Sphere(2)
        fs2(M, q) = q[3]
        grad_fs2(M, q) = project(M, q, [0.0, 0.0, 1.0])
        dmp = DefaultManoptProblem(S, ManifoldGradientObjective(fs2, grad_fs2))
        p = [1.0, 0.0, 0.0]
        d = QuasiNewtonLimitedMemoryDirectionUpdate(
            S, p, InverseBFGS(), 2; vector_transport_method = ProjectionTransport()
        )
        qns = QuasiNewtonState(
            S; p = copy(p), direction_update = d, vector_transport_method = ParallelTransport(),
            stepsize = Manopt.ConstantStepsize(S, π / 4),
        )
        initialize_solver!(dmp, qns)
        step_solver!(dmp, qns, 1)
        # a step of length π/4 projected to the new tangent space shrinks by cos(π/4)
        @test norm(S, qns.p, qns.sk) ≈ (π / 4) * cos(π / 4)
        # y_k = X/β - X_old with β = 1/cos(π/4) for the projection, β = 1 for the parallel transport
        β = (π / 4) / norm(S, qns.p, qns.sk)
        @test isapprox(S, qns.p, qns.yk, qns.X ./ β .- qns.X_old)
    end
    @testset "Float32 default Wolfe-Powell line search" begin
        M = Euclidean(2)
        p = Float32[1.0, 2.0]
        qns = QuasiNewtonState(M; p = p)
        @test qns.stepsize.candidate_point isa Vector{Float32}
        @test qns.stepsize.last_stepsize isa Float32
        @test qns.stepsize.stop_when_stepsize_less isa Float32

        f(M, q) = sum(abs2, q)
        grad_f(M, q) = 2 .* q
        state = quasi_Newton(
            M, f, grad_f, p;
            stopping_criterion = StopAfterIteration(1),
            return_state = true,
            debug = [],
        )
        @test state.stepsize.candidate_point isa Vector{Float32}
        @test state.stepsize.last_stepsize isa Float32
        @test state.stepsize.stop_when_stepsize_less isa Float32
    end
    @testset "Least squares on the complex Stiefel manifold (#656)" begin
        M = Stiefel(3, 2, ℂ)
        A = ComplexF64[1 1im 0; 0 1 1im; 1im 0 1]
        B = ComplexF64[1 0; 0 1im; 1 1]
        f(M, p) = sum(abs2, A * p - B)
        grad_f(M, p) = project(M, p, 2 * A' * (A * p - B))
        p0 = ComplexF64[1 0; 0 1; 0 0]
        p = quasi_Newton(
            M, f, grad_f, p0; memory_size = 1,
            stopping_criterion = StopWhenGradientNormLess(1.0e-9) | StopAfterIteration(50),
        )
        @test norm(M, p, grad_f(M, p)) < 1.0e-7
    end
    @testset "Matrix update on ℂ² equals the update on the realification ℝ⁴" begin
        Mc = Euclidean(2; field = ℂ)
        Mr = Euclidean(4)
        C = ComplexF64[1 -1im]
        fc(M, z) = sum((abs2.(z) .- 1) .^ 2) + sum(abs2, C * z)
        grad_fc(M, z) = 4 .* (abs2.(z) .- 1) .* z + 2 * C' * (C * z)
        toz(x) = x[1:2] + im * x[3:4]
        tox(z) = [real(z); imag(z)]
        fr(M, x) = fc(Mc, toz(x))
        grad_fr(M, x) = tox(grad_fc(Mc, toz(x)))
        z0 = ComplexF64[2.0, 1 + im]
        sc = StopAfterIteration(4)
        zc = quasi_Newton(Mc, fc, grad_fc, z0; memory_size = -1, stopping_criterion = sc)
        xr = quasi_Newton(Mr, fr, grad_fr, tox(z0); memory_size = -1, stopping_criterion = sc)
        @test isapprox(Mc, zc, toz(xr))
    end
    @testset "Line search failure behavior" begin
        M = Euclidean(2)
        f(M, p) = 100 * (p[2] - p[1]^2)^2 + (1 - p[1])^2
        grad_f(M, p) = [-400 * p[1] * (p[2] - p[1]^2) - 2 * (1 - p[1]), 200 * (p[2] - p[1]^2)]
        p0 = [-1.2, 1.0]
        sc = StopAfterIteration(200) | StopWhenGradientNormLess(1.0e-8)
        # by default step sizes do not report failures
        @test !Manopt.has_failed(Manopt.ConstantStepsize(M))
        hz = HagerZhangLinesearch(; max_function_evaluations = 3)(M)
        @test hz.last_termination === :not_searched
        @test !Manopt.has_failed(hz)
        @test_throws ArgumentError QuasiNewtonState(M; p = copy(p0), linesearch_failure_behavior = :foo)
        @testset "accept" begin
            # accepting failed line searches makes the iterates sensitive to rounding,
            # so only check the default stopping criterion of a full run
            st = quasi_Newton(
                M, f, grad_f, p0;
                stepsize = HagerZhangLinesearch(; max_function_evaluations = 3),
                linesearch_failure_behavior = :accept, return_state = true,
            )
            qns = Manopt.get_state(st)
            @test qns.linesearch_failure_behavior === :accept
            @test !any(c -> c isa StopWhenStuck, get_stopping_criteria(qns.stop))
            # and the behavior step by step: the first failed line search is accepted
            mp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            qns = QuasiNewtonState(
                M; p = copy(p0), linesearch_failure_behavior = :accept,
                stepsize = HagerZhangLinesearch(; max_function_evaluations = 3)(M),
            )
            initialize_solver!(mp, qns)
            k = 0
            while qns.linesearch_status !== :failure_accepted && k < 50
                k += 1
                step_solver!(mp, qns, k)
            end
            @test qns.linesearch_status === :failure_accepted
            @test Manopt.has_failed(qns.stepsize)
            @test qns.stepsize.last_termination === :evaluation_limit
            # the iterate moved by the last trial step α η (η is scaled by α after the step)
            @test qns.p ≈ qns.p_old + qns.η
            @test get_last_stepsize(qns.stepsize) == qns.stepsize.triples[3].t
            @test !qns.direction_update_reset
            @test occursin("last trial step was accepted", Manopt.get_message(qns))
        end
        @testset "restart" begin
            # the default stopping criterion for :restart includes StopWhenStuck
            st = quasi_Newton(
                M, f, grad_f, p0;
                stepsize = HagerZhangLinesearch(; max_function_evaluations = 3),
                linesearch_failure_behavior = :restart,
                record = [:Cost], return_state = true,
            )
            costs = [f(M, p0); get_record(st)]
            qns = Manopt.get_state(st)
            @test qns.linesearch_failure_behavior === :restart
            # rejected steps keep the cost monotone, up to the tolerance of the approximate Wolfe conditions
            @test all(diff(costs) .<= 1.0e-6 * maximum(abs, costs))
            # the second failure in a row stops
            @test qns.linesearch_status === :failure_after_restart
            @test qns.direction_update_reset
            @test any(c -> c isa StopWhenStuck && Manopt.is_active_stopping_criterion(c), get_stopping_criteria(qns.stop))
            @test occursin("got stuck", get_reason(st))
            @test occursin("just been reinitialized", Manopt.get_message(qns))
            # the cost from the line search is the one at the (unchanged) iterate
            mp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            @test get_cost(mp, qns) == f(M, get_iterate(qns))
        end
        @testset "a custom stopping criterion is kept as is" begin
            st = quasi_Newton(
                M, f, grad_f, p0;
                stepsize = HagerZhangLinesearch(; max_function_evaluations = 3),
                linesearch_failure_behavior = :restart, stopping_criterion = sc, return_state = true,
            )
            qns = Manopt.get_state(st)
            @test !any(c -> c isa StopWhenStuck, get_stopping_criteria(qns.stop))
            # without it the solver keeps retrying from the same iterate
            @test qns.linesearch_status === :failure_after_restart
            @test get_count(st, :Iterations) == 200
        end
        @testset "same iterates without failures" begin
            res = [
                quasi_Newton(
                    M, f, grad_f, p0;
                    stepsize = HagerZhangLinesearch(), linesearch_failure_behavior = b, stopping_criterion = sc,
                ) for b in (:accept, :restart)
            ]
            @test res[1] == res[2]
        end
        @testset "a single restart" begin
            mp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            qns = QuasiNewtonState(
                M; p = copy(p0), linesearch_failure_behavior = :restart,
                stepsize = HagerZhangLinesearch(; max_function_evaluations = 3)(M),
            )
            initialize_solver!(mp, qns)
            # gather curvature information until a line search fails
            k = 0
            while qns.linesearch_status !== :failure_restarted && k < 50
                k += 1
                q = copy(M, qns.p)
                step_solver!(mp, qns, k)
                (qns.linesearch_status === :failure_restarted) && (@test qns.p == q)
            end
            @test qns.linesearch_status === :failure_restarted
            @test qns.direction_update_reset
            @test isempty(qns.direction_update.memory_s)
            @test occursin("step was rejected", Manopt.get_message(qns))
            c = StopWhenStuck()
            @test !c(mp, qns, k)
            @test !Manopt.is_stuck(qns)
            # states that do not report line search failures
            @test !Manopt.is_stuck(GradientDescentState(M; p = copy(p0)))
            # a terminal failure, also seen through a decorated state
            qns.linesearch_status = :failure_after_restart
            dqns = DebugSolverState(qns, DebugMessages(:Info, :Always))
            @test Manopt.is_stuck(dqns)
            @test c(mp, dqns, k)
            @test occursin("could not recover", get_reason(c))
            @test endswith(Manopt.status_summary(c; context = :inline), " reached")
            @test !c(mp, qns, 0) # reset on init
            @test get_reason(c) == ""
            qns.linesearch_status = :failure_restarted
            @test get_reason(c) == ""
            @test repr(c) == "StopWhenStuck()"
            @test !Manopt.indicates_convergence(c)
            @test Manopt.status_summary(c; context = :short) == repr(c)
            @test endswith(Manopt.status_summary(c; context = :inline), "not reached")
            @test startswith(Manopt.status_summary(c), "A stopping criterion to stop when the solver is stuck")
            @test occursin("line search failure:     restart", Manopt.status_summary(qns))
            @test occursin("linesearch_failure_behavior = restart", repr(qns))
        end
    end
end
