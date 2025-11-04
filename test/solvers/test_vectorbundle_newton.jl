s = joinpath(@__DIR__, "..", "ManoptTestSuite.jl")
!(s in LOAD_PATH) && (push!(LOAD_PATH, s))


using Manopt, Manifolds, ManifoldsBase, ManoptTestSuite, Test, LinearAlgebra
using LinearAlgebra: eigvals
@testset "Vectorbundle Newton" begin
    @testset "Vector bundle Newton runs – Rayleigh quotient minimization" begin
        N = 10
        matrix = zeros(N, N)
        for i in 1:N
            matrix[i, i] = i
        end

        M = Sphere(size(matrix, 1) - 1)
        f(::Sphere, p) = p' * matrix * p

        f_prime(p) = (2.0 * matrix * p)'
        f_second_derivative(p) = 2.0 * matrix

        struct NewtonEquation{F, T, NM, Nrhs}
            f_prime::F
            f_second_prime::T
            A::NM
            b::Nrhs
        end

        function NewtonEquation(M, f_pr, f_sp)
            A = zeros(N + 1, N + 1)
            b = zeros(N + 1)
            return NewtonEquation{typeof(f_pr), typeof(f_sp), typeof(A), typeof(b)}(f_pr, f_sp, A, b)
        end

        function (ne::NewtonEquation)(M, VB, p)
            ne.A .= hcat(vcat(ne.f_second_prime(p) - ne.f_prime(p) * p * Matrix{Float64}(I, N, N), p'), vcat(p, 0))
            ne.b .= vcat(ne.f_prime(p)', 0)
            return
        end

        function solve_augmented_system(problem, newtonstate)
            return ((problem.newton_equation.A) \ (-problem.newton_equation.b))[1:(end - 1)]
        end
        function solve_augmented_system!(problem, X, newtonstate)
            X .= ((problem.newton_equation.A) \ (-problem.newton_equation.b))[1:(end - 1)]
            return X
        end
        y0 = zeros(N)
        y0[2] = 1.0
        y0[3] = 1.0
        y0[5] = 1.0
        y0 = 1 / norm(y0) * y0

        NE = NewtonEquation(M, f_prime, f_second_derivative)

        alg_kwargs = (;
            stopping_criterion = (StopAfterIteration(15) | StopWhenChangeLess(M, 1.0e-11)),
            retraction_method = ProjectionRetraction(),
            stepsize = ConstantLength(M, 1.0),
        )
        y1 = Manopt.vectorbundle_newton(
            M, TangentBundle(M), NE, y0; sub_problem = solve_augmented_system, alg_kwargs...
        )

        @test any(isapprox(f(M, y1), λ; atol = 2.0 * 1.0e-2) for λ in eigvals(matrix))

        y2 = Manopt.vectorbundle_newton(
            M, TangentBundle(M), NE, y0;
            sub_problem = solve_augmented_system!, sub_state = InplaceEvaluation(),
            alg_kwargs...
        )
        @test y1 == y2


        y3 = copy(M, y0) # avoid working inplace of y0
        Manopt.vectorbundle_newton!(
            M, TangentBundle(M), NE, y3; sub_problem = solve_augmented_system,
            alg_kwargs...
        )
        @test y1 == y3
        y4 = copy(M, y0) # avoid working inplace of y0
        Manopt.vectorbundle_newton!(
            M, TangentBundle(M), NE, y4; sub_problem = solve_augmented_system,
            alg_kwargs...
        )
        @test y1 == y4
    end

    @testset "Affine covariant stepsize" begin
        N = 10
        matrix = zeros(N, N)
        for i in 1:N
            matrix[i, i] = i
        end

        M = Sphere(size(matrix, 1) - 1)
        f(::Sphere, p) = p' * matrix * p

        f_prime(p) = (2.0 * matrix * p)'
        f_second_derivative(p) = 2.0 * matrix

        struct NewtonEquation2{F, T, NM, Nrhs}
            f_prime::F
            f_second_prime::T
            A::NM
            b::Nrhs
        end

        function NewtonEquation2(M, f_pr, f_sp)
            A = zeros(N + 1, N + 1)
            b = zeros(N + 1)
            return NewtonEquation2{typeof(f_pr), typeof(f_sp), typeof(A), typeof(b)}(f_pr, f_sp, A, b)
        end

        function (ne::NewtonEquation2)(M, VB, p)
            ne.A .= hcat(vcat(ne.f_second_prime(p) - ne.f_prime(p) * p * Matrix{Float64}(I, N, N), p'), vcat(p, 0))
            ne.b .= vcat(ne.f_prime(p)', 0)
            return p
        end

        #  needed: function that returns the (transported) right hand side for the simplified Newton step
        function (ne::NewtonEquation2)(M, VB, p, p_trial)
            return vcat(vector_transport_to(M, p, ne.f_prime(p_trial)', p_trial, ProjectionTransport()), 0)
        end

        function solve_augmented_system(problem, newtonstate)
            return ((problem.newton_equation.A) \ (-problem.newton_equation.b))[1:(end - 1)]
        end

        y0 = zeros(N)
        y0[2] = 1.0
        y0[3] = 1.0
        y0[5] = 1.0
        y0 = 1 / norm(y0) * y0

        NE = NewtonEquation2(M, f_prime, f_second_derivative)

        st = Manopt.vectorbundle_newton(
            M, TangentBundle(M), NE, y0; sub_problem = solve_augmented_system,
            stopping_criterion = (StopAfterIteration(15) | StopWhenChangeLess(M, 1.0e-11)),
            retraction_method = ProjectionRetraction(),
            stepsize = Manopt.AffineCovariantStepsize(M, θ_des = 0.1),
            return_state = true,
        )
        y1 = get_iterate(st)
        @test any(isapprox(f(M, y1), λ; atol = 2.0 * 1.0e-2) for λ in eigvals(matrix))
        st_str = repr(st)
        @test occursin("Vector bundle Newton method", st_str)
        # we stopped since the change was small enough
        @test occursin("* |Δp| < 1.0e-11: reached", st_str)
        @test occursin("AffineCovariantStepsize", st_str)
        acs = st.stepsize
        @test get_initial_stepsize(acs) == acs.α
        @test get_last_stepsize(acs) > 0.0
        @test default_stepsize(M, VectorBundleNewtonState) isa Manopt.ConstantStepsize
    end
end
