using Manopt, Manifolds, Test
using LinearAlgebra: I, eigvecs, tr, Diagonal

mutable struct QuasiNewtonGradientDirectionUpdate{VT<:AbstractVectorTransportMethod} <:
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

struct QuasiNewtonTestDirectionUpdate{VT<:AbstractVectorTransportMethod} <:
       AbstractQuasiNewtonDirectionUpdate
    vector_transport_method::VT
end

@testset "Riemannian quasi-Newton Methods" begin
    @testset "Show & Status" begin
        M = Euclidean(4)
        qnu = InverseBFGS()
        d = QuasiNewtonMatrixDirectionUpdate(M, qnu)
        @test Manopt.status_summary(d) ==
            "$(qnu) with initial scaling 1.0 and vector transport method ParallelTransport()."
        s = "QuasiNewtonMatrixDirectionUpdate(DefaultOrthonormalBasis(ℝ), [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0], 1.0, InverseBFGS(), ParallelTransport())\n"
        @test repr(d) == s
        @test Manopt.get_message(d) == ""
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
            M, f, grad_f, p; stopping_criterion=StopWhenGradientNormLess(10^(-6))
        )
        @test norm(x_lrbfgs - x_solution) ≈ 0 atol = 10.0^(-14)
        # with State
        lrbfgs_s = quasi_Newton(
            M,
            f,
            grad_f,
            p;
            stopping_criterion=StopWhenGradientNormLess(10^(-6)),
            return_state=true,
            debug=[],
        )
        # Verify that Newton update direction works also allocating
        dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
        p_star = get_solver_result(lrbfgs_s)
        D = zero_vector(M, p_star)
        lrbfgs_s.direction_update(D, dmp, lrbfgs_s)
        @test isapprox(M, p_star, D, lrbfgs_s.direction_update(dmp, lrbfgs_s))

        @test startswith(
            repr(lrbfgs_s), "# Solver state for `Manopt.jl`s Quasi Newton Method\n"
        )
        @test get_last_stepsize(dmp, lrbfgs_s, lrbfgs_s.stepsize) > 0
        @test Manopt.get_iterate(lrbfgs_s) == x_lrbfgs
        set_gradient!(lrbfgs_s, M, p, grad_f(M, p))
        @test isapprox(M, p, Manopt.get_gradient(lrbfgs_s), grad_f(M, p))
        @test Manopt.get_message(lrbfgs_s) == ""
        # with Cached Basis
        x_lrbfgs_cached = quasi_Newton(
            M,
            f,
            grad_f,
            p;
            stopping_criterion=StopWhenGradientNormLess(10^(-6)),
            basis=get_basis(M, p, DefaultOrthonormalBasis()),
        )
        @test isapprox(M, x_lrbfgs_cached, x_lrbfgs)

        x_lrbfgs_cached_2 = quasi_Newton(
            M,
            f,
            grad_f,
            p;
            stopping_criterion=StopWhenGradientNormLess(10^(-6)),
            basis=get_basis(M, p, DefaultOrthonormalBasis()),
            memory_size=-1,
        )
        @test isapprox(M, x_lrbfgs_cached_2, x_lrbfgs; atol=1e-5)

        # with Costgrad
        mcgo = ManifoldCostGradientObjective(costgrad)

        x_lrbfgs_costgrad = quasi_Newton(
            M, mcgo, p; stopping_criterion=StopWhenGradientNormLess(10^(-6)), debug=[]
        )
        @test isapprox(M, x_lrbfgs_costgrad, x_lrbfgs; atol=1e-5)

        clrbfgs_s = quasi_Newton(
            M,
            f,
            grad_f,
            p;
            cautious_update=true,
            stopping_criterion=StopWhenGradientNormLess(10^(-6)),
            return_state=true,
            debug=[],
        )
        # Test direction passthrough
        x_clrbfgs = get_solver_result(clrbfgs_s)
        D = zero_vector(M, x_clrbfgs)
        clrbfgs_s.direction_update(D, dmp, clrbfgs_s)
        @test isapprox(M, x_clrbfgs, D, clrbfgs_s.direction_update(dmp, clrbfgs_s))

        @test norm(x_clrbfgs - x_solution) ≈ 0 atol = 10.0^(-14)

        x_rbfgs_Huang = quasi_Newton(
            M,
            f,
            grad_f,
            p;
            memory_size=-1,
            stepsize=WolfePowellBinaryLinesearch(
                M;
                retraction_method=ExponentialRetraction(),
                vector_transport_method=ParallelTransport(),
            ),
            stopping_criterion=StopWhenGradientNormLess(10^(-6)),
        )
        @test norm(x_rbfgs_Huang - x_solution) ≈ 0 atol = 10.0^(-14)

        for T in [InverseBFGS(), BFGS(), InverseDFP(), DFP(), InverseSR1(), SR1()]
            for c in [true, false]
                x_state = quasi_Newton(
                    M,
                    f,
                    grad_f,
                    p;
                    direction_update=T,
                    cautious_update=c,
                    memory_size=-1,
                    stopping_criterion=StopWhenGradientNormLess(10^(-12)),
                    return_state=true,
                    debug=[],
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
        rayleigh_atol = 1e-7
        A = [2.0 1.0 0.0 3.0; 1.0 3.0 4.0 5.0; 0.0 4.0 3.0 2.0; 3.0 5.0 2.0 6.0]
        A = (A + A') / 2
        M = Sphere(n - 1)
        f(::Sphere, X) = X' * A * X
        grad_f(::Sphere, X) = 2 * (A * X - X * (X' * A * X))
        grad_f!(::Sphere, Y, X) = (Y .= 2 * (A * X - X * (X' * A * X)))
        x_solution = eigvecs(A)[:, 1]

        x = Matrix{Float64}(I, n, n)[n, :]
        x_lrbfgs = quasi_Newton(M, f, grad_f, x; memory_size=-1)
        @test isapprox(M, x_lrbfgs, x_solution; atol=rayleigh_atol)
        x_lrbfgs2 = copy(M, x)
        quasi_Newton!(
            M, f, grad_f!, x_lrbfgs2; evaluation=InplaceEvaluation(), memory_size=-1
        )
        @test isapprox(M, x_lrbfgs2, x_lrbfgs)

        # A simple preconditioner
        x_lrbfgs = quasi_Newton(
            M, f, grad_f, x; memory_size=-1, preconditioner=(M, p, X) -> 0.5 .* X
        )
        @test isapprox(M, x_lrbfgs, x_solution; atol=rayleigh_atol)

        # An in-place preconditioner
        x_lrbfgs = quasi_Newton(
            M,
            f,
            grad_f,
            x;
            memory_size=-1,
            preconditioner=QuasiNewtonPreconditioner(
                (M, Y, p, X) -> (Y .= 0.5 .* X); evaluation=InplaceEvaluation()
            ),
        )
        @test isapprox(M, x_lrbfgs, x_solution; atol=rayleigh_atol)

        x_clrbfgs = quasi_Newton(M, f, grad_f, x; cautious_update=true)
        @test isapprox(M, x_clrbfgs, x_solution; atol=rayleigh_atol)

        x_cached_lrbfgs = quasi_Newton(M, f, grad_f, x; memory_size=-1)
        @test isapprox(M, x_cached_lrbfgs, x_solution; atol=rayleigh_atol)

        for T in [
                InverseDFP(),
                DFP(),
                Broyden(0.5),
                InverseBroyden(0.5),
                Broyden(0.5, :Davidon),
                Broyden(0.5, :InverseDavidon),
                InverseBFGS(),
                BFGS(),
            ],
            c in [true, false]

            x_direction = quasi_Newton(
                M, f, grad_f, x; direction_update=T, cautious_update=c, memory_size=-1
            )
            @test isapprox(M, x_direction, x_solution; atol=rayleigh_atol)
        end
    end

    @testset "Brocket" begin
        struct GradF
            A::Matrix{Float64}
            N::Diagonal{Float64,Vector{Float64}}
        end
        function (gradF::GradF)(::Stiefel, X::Array{Float64,2})
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
            M,
            f,
            grad_f,
            x;
            memory_size=8,
            vector_transport_method=ProjectionTransport(),
            retraction_method=QRRetraction(),
            cautious_update=true,
            stopping_criterion=StopWhenGradientNormLess(1e-6),
        )

        x_inverseBFGSHuang = quasi_Newton(
            M,
            f,
            grad_f,
            x;
            memory_size=8,
            stepsize=WolfePowellBinaryLinesearch(
                M;
                retraction_method=QRRetraction(),
                vector_transport_method=ProjectionTransport(),
            ),
            vector_transport_method=ProjectionTransport(),
            retraction_method=QRRetraction(),
            cautious_update=true,
            stopping_criterion=StopWhenGradientNormLess(1e-6),
        )
        @test isapprox(M, x_inverseBFGSCautious, x_inverseBFGSHuang; atol=2e-4)
    end

    @testset "Wolfe Powell linesearch" begin
        n = 4
        rayleigh_atol = 1e-8
        A = [2.0 1.0 0.0 3.0; 1.0 3.0 4.0 5.0; 0.0 4.0 3.0 2.0; 3.0 5.0 2.0 6.0]
        A = (A + A') / 2
        M = Sphere(n - 1)
        F(::Sphere, X) = X' * A * X
        grad_f(::Sphere, X) = 2 * (A * X - X * (X' * A * X))
        x_solution = abs.(eigvecs(A)[:, 1])

        x = [
            0.7011245948687502
            -0.1726003159556036
            0.38798265967671103
            -0.5728026616491424
        ]
        x_lrbfgs = quasi_Newton(
            M,
            F,
            grad_f,
            x;
            basis=get_basis(M, x, DefaultOrthonormalBasis()),
            memory_size=-1,
            stopping_criterion=StopWhenGradientNormLess(1e-9),
        )
        @test norm(abs.(x_lrbfgs) - x_solution) ≈ 0 atol = rayleigh_atol
    end

    @testset "update rules" begin
        n = 4
        A = [2.0 1.0 0.0 3.0; 1.0 3.0 4.0 5.0; 0.0 4.0 3.0 2.0; 3.0 5.0 2.0 6.0]
        A = (A + A') / 2
        M = Sphere(n - 1)
        F(::Sphere, X) = X' * A * X
        grad_f(::Sphere, X) = 2 * (A * X - X * (X' * A * X))
        grad_f!(::Sphere, X, p) = (X .= 2 * (A * X - X * (X' * A * X)))

        p_1 = [1.0; 0.0; 0.0; 0.0]
        p_2 = [0.0; 0.0; 1.0; 0.0]

        SR1_allocating = ApproxHessianSymmetricRankOne(
            M, copy(M, p_1), grad_f; evaluation=AllocatingEvaluation()
        )

        SR1_inplace = ApproxHessianSymmetricRankOne(
            M, copy(M, p_1), grad_f!; evaluation=InplaceEvaluation()
        )

        BFGS_allocating = ApproxHessianBFGS(
            M, copy(M, p_1), grad_f; evaluation=AllocatingEvaluation()
        )

        BFGS_inplace = ApproxHessianBFGS(
            M, copy(M, p_1), grad_f!; evaluation=InplaceEvaluation()
        )

        Y = [0.0; 1.0; 0.0; 0.0]
        X_1 = SR1_allocating(M, p_1, Y)
        SR1_allocating.p_tmp .= p_2
        X_2 = SR1_allocating(M, p_1, Y)
        @test isapprox(M, p_1, X_1, X_2; atol=1e-10)
        update_hessian_basis!(M, SR1_allocating, p_1)
        update_hessian_basis!(M, SR1_allocating, p_2)

        X_3 = zero_vector(M, p_1)
        X_4 = zero_vector(M, p_1)
        SR1_inplace(M, X_3, p_1, Y)
        SR1_inplace.p_tmp .= p_2
        SR1_inplace(M, X_4, p_1, Y)
        @test isapprox(M, p_1, X_3, X_4; atol=1e-10)
        update_hessian_basis!(M, SR1_inplace, p_1)
        update_hessian_basis!(M, SR1_inplace, p_2)

        X_5 = BFGS_allocating(M, p_1, Y)
        X_6 = BFGS_allocating(M, p_2, Y)
        @test isapprox(M, p_1, X_5, X_6; atol=1e-10)
        update_hessian_basis!(M, BFGS_allocating, p_1)
        update_hessian_basis!(M, BFGS_allocating, p_2)

        X_7 = zero_vector(M, p_1)
        X_8 = zero_vector(M, p_1)
        BFGS_inplace(M, X_7, p_1, Y)
        BFGS_inplace(M, X_8, p_2, Y)
        update_hessian_basis!(M, BFGS_inplace, p_1)
        update_hessian_basis!(M, BFGS_inplace, p_2)

        @test isapprox(M, p_1, X_3, X_4; atol=1e-10)

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
        M = Euclidean(2; field=ℂ)
        A = [2 im; -im 2]
        fc(::Euclidean, p) = real(p' * A * p)
        grad_fc(::Euclidean, p) = 2 * A * p
        p0 = [2.0, 1 + im]
        @test_logs (:info,) Manopt.set_parameter!(:Mode, "Tutorial")
        p4 = quasi_Newton(M, fc, grad_fc, p0; stopoing_criterion=StopAfterIteration(3))
        @test_logs (:info,) Manopt.set_parameter!(:Mode, "")
        @test fc(M, p4) ≤ fc(M, p0)
    end

    @testset "Boundary cases and safeguards" begin
        M = Euclidean(2)
        p = [0.0, 0.0]
        f(M, p) = sum(p .^ 2)
        grad_f(M, p) = 2 * sum(p)
        gmp = ManifoldGradientObjective(f, grad_f)
        mp = DefaultManoptProblem(M, gmp)
        qns = QuasiNewtonState(M; p=p)
        # push zeros to memory
        push!(qns.direction_update.memory_s, copy(p))
        push!(qns.direction_update.memory_s, copy(p))
        push!(qns.direction_update.memory_y, copy(p))
        push!(qns.direction_update.memory_y, copy(p))
        qns.direction_update(mp, qns)
        # Update (1) says at i=1 inner products are zero (2) all are zero -> gradient proposal
        @test contains(qns.direction_update.message, "i=1,2")
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
            M;
            p=copy(M, p),
            direction_update=QuasiNewtonGradientDirectionUpdate(ParallelTransport()),
            nondescent_direction_behavior=:step_towards_negative_gradient,
        )
        dqns = DebugSolverState(qns, DebugMessages(:Warning, :Once))
        @test_logs (
            :warn,
            "Computed direction is not a descent direction. The inner product evaluated to 1.0. Resetting to negative gradient.",
        ) (
            :warn,
            "Further warnings will be suppressed, use DebugMessages(:Warning, :Always) to get all warnings.",
        ) solve!(mp, dqns)

        qns = QuasiNewtonState(
            M;
            p=copy(M, p),
            direction_update=QuasiNewtonGradientDirectionUpdate(ParallelTransport()),
            nondescent_direction_behavior=:step_towards_negative_gradient,
        )

        @test_nowarn solve!(mp, qns)
        @test qns.direction_update.num_times_init == 1

        qns = QuasiNewtonState(
            M;
            p=copy(M, p),
            direction_update=QuasiNewtonGradientDirectionUpdate(ParallelTransport()),
            nondescent_direction_behavior=:reinitialize_direction_update,
        )

        @test_nowarn solve!(mp, qns)
        @test qns.direction_update.num_times_init == 1001
    end

    @testset "A Circle example" begin
        M = Circle()
        data = [-π / 2, π / 4, 0.0, π / 4]
        pstar = sum([-π / 2, π / 4, 0.0, π / 4]) / length(data)
        f(M, p) = 1 / 10 * sum(distance.(Ref(M), data, Ref(p)) .^ 2)
        grad_f(M, p) = 1 / 5 * sum(-log.(Ref(M), Ref(p), data))
        p = quasi_Newton(M, f, grad_f, data[1])
        @test isapprox(M, pstar, p)
        s = quasi_Newton(M, f, grad_f, data[1]; return_state=true)
        @test get_solver_result(s)[] == p
    end
end
