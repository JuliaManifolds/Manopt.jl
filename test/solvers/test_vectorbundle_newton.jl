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
            res = (problem.newton_equation.A) \ (-problem.newton_equation.b)
            return res[1:N]
        end

        y0 = 1 / sqrt(N) * ones(N)

        NE = NewtonEquation(M, f_prime, f_second_derivative)

        res = Manopt.vectorbundle_newton(
            M, TangentBundle(M), NE, y0; sub_problem = solve_augmented_system,
            stopping_criterion = (StopAfterIteration(40) | StopWhenChangeLess(M, 1.0e-11)),
            retraction_method = ProjectionRetraction(),
            stepsize = ConstantLength(M, 1.0)
        )

        @test any(isapprox(f(M, res), λ; atol = 2.0 * 1.0e-2) for λ in eigvals(matrix))
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

        #  needed: function that returns the (transported) right hand side for the simplified Newton step
        function (ne::NewtonEquation)(M, VB, p, p_trial)
            return vcat(vector_transport_to(M, p, ne.f_prime(p_trial)', p_trial, ProjectionTransport()), 0)
        end

        function solve_augmented_system(problem, newtonstate)
            res = (problem.newton_equation.A) \ (-problem.newton_equation.b)
            return res[1:N]
        end

        y0 = 1 / sqrt(N) * ones(N)

        NE = NewtonEquation(M, f_prime, f_second_derivative)

        st_res = Manopt.vectorbundle_newton(
            M, TangentBundle(M), NE, y0; sub_problem = solve_augmented_system,
            stopping_criterion = (StopAfterIteration(40) | StopWhenChangeLess(M, 1.0e-11)),
            retraction_method = ProjectionRetraction(),
            stepsize = Manopt.AffineCovariantStepsize(M, θ_des = 0.1),
            record = [:Iterate, :Change],
            return_state = true
        )

        res = get_solver_result(st_res)
       @test any(isapprox(f(M, res), λ; atol = 2.0 * 1.0e-2) for λ in eigvals(matrix))
    end
end
