using Manopt, Manifolds, ManifoldsBase, Test

@testset "ManoptCUDAExt" begin
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
            M = Euclidean(10)
            target = CuArray(randn(10))
            f(M, p) = 0.5 * sum((p .- target) .^ 2)
            grad_f(M, p) = p .- target
            p0 = CUDA.zeros(10)

            result = gradient_descent(
                M, f, grad_f, p0;
                stopping_criterion=StopAfterIteration(200),
                stepsize=ConstantLength(0.1),
            )
            @test result isa CuArray
            @test isapprox(Array(result), Array(target); atol=1e-6)
        end

        @testset "GD + ArmijoLinesearch on Euclidean" begin
            # Tests the linesearch_backtrack! CPU/GPU mismatch fix.
            # ArmijoLinesearchStepsize pre-allocates candidate_point as CPU Array;
            # the extension intercepts linesearch_backtrack! to use a GPU buffer.
            M = Euclidean(10)
            target = CuArray(randn(10))
            f(M, p) = 0.5 * sum((p .- target) .^ 2)
            grad_f(M, p) = p .- target
            p0 = CUDA.zeros(10)

            result = gradient_descent(
                M, f, grad_f, p0;
                stopping_criterion=StopAfterIteration(50),
            )
            @test result isa CuArray
            @test isapprox(Array(result), Array(target); atol=1e-6)
        end

        @testset "GD + record on Euclidean" begin
            M = Euclidean(5)
            target = CuArray(randn(5))
            f(M, p) = 0.5 * sum((p .- target) .^ 2)
            grad_f(M, p) = p .- target
            p0 = CUDA.zeros(5)

            result = gradient_descent(
                M, f, grad_f, p0;
                stopping_criterion=StopAfterIteration(20),
                stepsize=ConstantLength(0.1),
                record=[:Cost],
                return_state=true,
            )
            costs = get_record(result, :Cost)
            @test length(costs) == 20
            # Cost should decrease monotonically
            @test all(diff(costs) .<= 0)
            p_final = get_solver_result(result)
            @test p_final isa CuArray
        end

        @testset "GD with Float32 on Euclidean" begin
            M = Euclidean(8)
            target = CuArray(randn(Float32, 8))
            f(M, p) = Float32(0.5) * sum((p .- target) .^ 2)
            grad_f(M, p) = p .- target
            p0 = CUDA.zeros(Float32, 8)

            result = gradient_descent(
                M, f, grad_f, p0;
                stopping_criterion=StopAfterIteration(200),
                stepsize=ConstantLength(Float32(0.1)),
            )
            @test result isa CuArray{Float32}
            @test isapprox(Array(result), Array(target); atol=1e-3)
        end

        @testset "GD on matrix-valued Euclidean" begin
            M = Euclidean(3, 3)
            target = CuArray(randn(3, 3))
            f(M, p) = 0.5 * sum((p .- target) .^ 2)
            grad_f(M, p) = p .- target
            p0 = CUDA.zeros(3, 3)

            result = gradient_descent(
                M, f, grad_f, p0;
                stopping_criterion=StopAfterIteration(200),
                stepsize=ConstantLength(0.1),
            )
            @test result isa CuArray{Float64,2}
            @test size(result) == (3, 3)
            @test isapprox(Array(result), Array(target); atol=1e-6)
        end

        @testset "Conjugate GD on Euclidean" begin
            M = Euclidean(10)
            target = CuArray(randn(10))
            f(M, p) = 0.5 * sum((p .- target) .^ 2)
            grad_f(M, p) = p .- target
            p0 = CUDA.zeros(10)

            result = conjugate_gradient_descent(
                M, f, grad_f, p0;
                stopping_criterion=StopAfterIteration(50),
                stepsize=ConstantLength(0.1),
            )
            @test result isa CuArray
            @test isapprox(Array(result), Array(target); atol=1e-4)
        end

        @testset "GD on Sphere" begin
            M = Sphere(9)
            # Target point on sphere
            t = randn(10)
            t ./= norm(t)
            target = CuArray(t)

            # Minimize geodesic distance squared
            f(M, p) = 0.5 * sum((p .- target) .^ 2)
            grad_f(M, p) = project(M, p, p .- target)

            # Start from a different point
            s = randn(10)
            s ./= norm(s)
            p0 = CuArray(s)

            result = gradient_descent(
                M, f, grad_f, p0;
                stopping_criterion=StopAfterIteration(100),
                stepsize=ConstantLength(0.1),
            )
            @test result isa CuArray
            # Should be on the sphere
            @test isapprox(norm(Array(result)), 1.0; atol=1e-10)
        end
    else
        @info "CUDA not functional, skipping ManoptCUDAExt GPU tests"
    end
end
