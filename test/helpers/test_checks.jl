using Manifolds, Manopt, Plots, Test
# don't show plots actually
default(; show=false, reuse=true)

@testset "Numerical Check functions" begin
    @testset "Test Gradient checks" begin
        n = 10
        M = Sphere(n)
        q = zeros(n + 1)
        q[1] = 1.0
        p = zeros(11)
        p[1:4] .= 1 / sqrt(4)
        X = log(M, p, q)

        f(M, p) = 1 / 2 * distance(M, p, q)^2
        grad_f(M, p) = -log(M, p, q)

        @test check_gradient(M, f, grad_f, p, X)

        grad_fb(M, p) = -0.5 * log(M, p, q)
        @test_throws ErrorException check_gradient(M, f, grad_fb, p, X; error=:error)
        @test !check_gradient(M, f, grad_fb, p, X)

        check_gradient(M, f, grad_f, p, X; plot=true)

        #test window size error
        @test_throws ErrorException Manopt.find_best_slope_window(zeros(2), zeros(2), 20)
        @test_throws ErrorException Manopt.find_best_slope_window(
            zeros(2), zeros(2), [2, 20]
        )
        #Check complex Sphere as well
        M2 = Sphere(n, ℂ)
        check_gradient(M2, f, grad_f, p, X)
        # Linear Euclidean function -> exact
        M2 = Euclidean(1)
        f2(M, p) = 3 * p[1]
        grad_f2(M, p) = [3]
        p2 = [1.0]
        X2 = [2.0]
        # true due to exactness.
        check_gradient(M2, f2, grad_f2, p2, X2;)
    end
    @testset "Hessian Checks" begin
        M3 = Euclidean(2)
        A = [2.0 1.0; 1.0 2.0]
        f3(M, p) = 0.5 * p' * A * p
        grad_f3(M, p) = A * p
        Hess_f3(M, p, X) = A * X
        p3 = [1.0, 2.0]
        X3 = [1.0, 0.0]
        #just run all defaults with true and even the gradient descent
        @test check_Hessian(M3, f3, grad_f3, Hess_f3, p3, X3; mode=:CriticalPoint)
        # Euclidean and completely exact

        # gradient not correct
        grad_f3f(M, p) = 2 .* A * p
        @test !check_Hessian(M3, f3, grad_f3f, Hess_f3, p3, X3)

        # Hessian not linear
        Hess_f3f1(M, p, X) = X .^ 2
        @test !check_Hessian(M3, f3, grad_f3, Hess_f3f1, p3, X3)

        # Hessian not symmetric
        Hess_f3f2(M, p, X) = [0.0 1.0; 0.0 0.0] * X
        @test !check_Hessian(M3, f3, grad_f3, Hess_f3f2, p3, X3, [0.0, 1.0])

        # Sphere
        M4 = Sphere(2)
        A2 = [2.0 1.0 0.0; 1.0 2.0 1.0; 0.0 1.0 2.0]
        f4(::Sphere, p) = p' * A2 * p
        grad_f4(::Sphere, p) = 2 * (A2 * p - p * p' * A2 * p)
        function Hess_f4(::Sphere, p, X)
            return 2 *
                   (A2 * X - p * p' * A2 * X - X * p' * A2 * p - p * p' * X * p' * A2 * p)
        end
        p4 = [1.0, 0.0, 0.0]
        X4 = [0.0, 1.0, 1.0]

        #Verify the `check` for a bit smaller scale due to rounding errors
        @test check_Hessian(M4, f4, grad_f4, Hess_f4, p4, X4; limits=(-5.0, 0.0))

        #Hessian not tangent
        Hess_f4f1(::Sphere, p, X) = p
        @test !check_Hessian(M4, f4, grad_f4, Hess_f4f1, p4, X4)
    end
end
