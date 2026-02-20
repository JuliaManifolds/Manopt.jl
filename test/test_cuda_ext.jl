using Manopt, Manifolds, ManifoldsBase, Test
using LinearAlgebra

@testset "GPU solver tests" begin
    cuda_loaded = false
    try
        using CUDA
        cuda_loaded = CUDA.functional()
    catch
        cuda_loaded = false
    end

    if cuda_loaded
        @eval using CUDA

        @testset "GD + ConstantLength on Euclidean" begin
            M = Euclidean(3)
            target_cpu = [1.0, 2.0, 3.0]
            target = CuArray(target_cpu)
            f(M, p) = sum((p .- target) .^ 2) / 2
            grad_f(M, p) = p .- target
            p0 = CuArray(zeros(3))

            result = gradient_descent(
                M, f, grad_f, p0;
                stopping_criterion=StopAfterIteration(200),
                stepsize=ConstantLength(0.1),
            )
            @test result isa CuArray{Float64}
            @test isapprox(Array(result), target_cpu; atol=1e-6)
        end

        @testset "GD + ArmijoLinesearch on Euclidean" begin
            M = Euclidean(3)
            target_cpu = [1.0, 2.0, 3.0]
            target = CuArray(target_cpu)
            f(M, p) = sum((p .- target) .^ 2) / 2
            grad_f(M, p) = p .- target
            p0 = CuArray(zeros(3))

            result = gradient_descent(
                M, f, grad_f, p0;
                stopping_criterion=StopAfterIteration(50),
            )
            @test result isa CuArray{Float64}
            @test isapprox(Array(result), target_cpu; atol=1e-5)
        end

        @testset "GD + ConstantLength Float32" begin
            T = Float32
            M = Euclidean(3)
            target_cpu = T[1.0, 2.0, 3.0]
            target = CuArray(target_cpu)
            f(M, p) = sum((p .- target) .^ 2) / T(2)
            grad_f(M, p) = p .- target
            p0 = CUDA.zeros(T, 3)

            result = gradient_descent(
                M, f, grad_f, p0;
                stopping_criterion=StopAfterIteration(200),
                stepsize=ConstantLength(T(0.1)),
            )
            @test result isa CuArray{Float32}
            @test isapprox(Array(result), target_cpu; atol=T(1e-3))
        end

        @testset "GD on matrix-valued Euclidean" begin
            M = Euclidean(3, 3)
            target_cpu = randn(3, 3)
            target = CuArray(target_cpu)
            f(M, p) = sum((p .- target) .^ 2) / 2
            grad_f(M, p) = p .- target
            p0 = CuArray(zeros(3, 3))

            result = gradient_descent(
                M, f, grad_f, p0;
                stopping_criterion=StopAfterIteration(200),
                stepsize=ConstantLength(0.1),
            )
            @test result isa CuArray{Float64,2}
            @test size(result) == (3, 3)
            @test isapprox(Array(result), target_cpu; atol=1e-6)
        end

        @testset "Conjugate GD on Euclidean" begin
            M = Euclidean(5)
            target_cpu = randn(5)
            target = CuArray(target_cpu)
            f(M, p) = sum((p .- target) .^ 2) / 2
            grad_f(M, p) = p .- target
            p0 = CuArray(zeros(5))

            result = conjugate_gradient_descent(
                M, f, grad_f, p0;
                stopping_criterion=StopAfterIteration(50),
                stepsize=ConstantLength(0.1),
            )
            @test result isa CuArray{Float64}
            @test isapprox(Array(result), target_cpu; atol=1e-3)
        end

        @testset "GD on Sphere" begin
            M = Sphere(2)
            a_cpu = [1.0, 2.0, 3.0]
            known_solution = a_cpu / norm(a_cpu)
            a = CuArray(a_cpu)
            f(M, p) = sum((p .- a) .^ 2) / 2
            grad_f(M, p) = project(M, p, p .- a)

            s_cpu = [0.0, 0.0, 1.0]
            p0 = CuArray(s_cpu)

            result = gradient_descent(
                M, f, grad_f, p0;
                stopping_criterion=StopAfterIteration(100),
                stepsize=ConstantLength(0.1),
            )
            @test result isa CuArray{Float64}
            @test isapprox(norm(Array(result)), 1.0; atol=1e-10)
            @test isapprox(Array(result), known_solution; atol=1e-4)
        end

        @testset "GD + recording on Euclidean" begin
            M = Euclidean(3)
            target = CuArray([1.0, 2.0, 3.0])
            f(M, p) = sum((p .- target) .^ 2) / 2
            grad_f(M, p) = p .- target
            p0 = CuArray(zeros(3))

            result = gradient_descent(
                M, f, grad_f, p0;
                stopping_criterion=StopAfterIteration(20),
                stepsize=ConstantLength(0.1),
                record=[:Cost],
                return_state=true,
            )
            rec = get_record(result)
            @test length(rec) == 20
            p_final = get_solver_result(result)
            @test p_final isa CuArray{Float64}
        end

        @testset "CPU vs GPU equivalence" begin
            M = Euclidean(5)
            target_cpu = randn(5)
            target_gpu = CuArray(target_cpu)

            f_cpu(M, p) = sum((p .- target_cpu) .^ 2) / 2
            grad_f_cpu(M, p) = p .- target_cpu
            f_gpu(M, p) = sum((p .- target_gpu) .^ 2) / 2
            grad_f_gpu(M, p) = p .- target_gpu

            p0_cpu = zeros(5)
            p0_gpu = CuArray(zeros(5))

            result_cpu = gradient_descent(
                M, f_cpu, grad_f_cpu, p0_cpu;
                stopping_criterion=StopAfterIteration(100),
                stepsize=ConstantLength(0.1),
            )
            result_gpu = gradient_descent(
                M, f_gpu, grad_f_gpu, p0_gpu;
                stopping_criterion=StopAfterIteration(100),
                stepsize=ConstantLength(0.1),
            )
            @test isapprox(Array(result_gpu), result_cpu; atol=1e-10)
        end
    else
        @info "CUDA not functional, skipping GPU solver tests"
    end
end
