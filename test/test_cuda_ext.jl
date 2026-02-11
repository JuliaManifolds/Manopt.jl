using Manopt, Manifolds, ManifoldsBase, Test
using CUDA

@testset "ManoptCUDAExt" begin
    if CUDA.functional()
        @testset "GPU allocate" begin
            M = Euclidean(10)
            p_gpu = CUDA.zeros(10)
            q = ManifoldsBase.allocate(M, p_gpu)
            @test q isa CuArray
            @test size(q) == (10,)

            q2 = ManifoldsBase.allocate(M, p_gpu, Float32)
            @test q2 isa CuArray{Float32}
            @test size(q2) == (10,)
        end

        @testset "Gradient descent with ConstantLength on Euclidean" begin
            # Simple quadratic on R^n.
            # Euclidean uses broadcasting-based retract/inner/norm, so this
            # tests the basic GPU solver path without linesearch workspace.
            M = Euclidean(10)
            target = randn(10)
            target_gpu = CuArray(target)

            f_cpu(M, p) = 0.5 * sum((p .- target) .^ 2)
            grad_f_cpu(M, p) = p .- target

            f_gpu(M, p) = 0.5 * sum((p .- target_gpu) .^ 2)
            grad_f_gpu(M, p) = p .- target_gpu

            p0_cpu = zeros(10)
            result_cpu = gradient_descent(
                M, f_cpu, grad_f_cpu, p0_cpu;
                stopping_criterion=StopAfterIteration(100),
                stepsize=ConstantLength(0.1),
            )

            p0_gpu = CuArray(zeros(10))
            result_gpu = gradient_descent(
                M, f_gpu, grad_f_gpu, p0_gpu;
                stopping_criterion=StopAfterIteration(100),
                stepsize=ConstantLength(0.1),
            )

            @test result_gpu isa CuArray
            @test isapprox(Array(result_gpu), result_cpu; atol=1e-10)
        end

        @testset "Gradient descent with ArmijoLinesearch on Euclidean" begin
            # Tests the candidate_point adaptation path.
            # ArmijoLinesearchStepsize is the default stepsize for gradient_descent.
            M = Euclidean(10)
            target = randn(10)
            target_gpu = CuArray(target)

            f_cpu(M, p) = 0.5 * sum((p .- target) .^ 2)
            grad_f_cpu(M, p) = p .- target

            f_gpu(M, p) = 0.5 * sum((p .- target_gpu) .^ 2)
            grad_f_gpu(M, p) = p .- target_gpu

            p0_cpu = zeros(10)
            result_cpu = gradient_descent(
                M, f_cpu, grad_f_cpu, p0_cpu;
                stopping_criterion=StopAfterIteration(50),
            )
            # Default stepsize is ArmijoLinesearchStepsize

            p0_gpu = CuArray(zeros(10))
            result_gpu = gradient_descent(
                M, f_gpu, grad_f_gpu, p0_gpu;
                stopping_criterion=StopAfterIteration(50),
            )

            @test result_gpu isa CuArray
            @test isapprox(Array(result_gpu), result_cpu; atol=1e-10)
        end
    else
        @info "CUDA not functional, skipping ManoptCUDAExt GPU tests"
    end
end
