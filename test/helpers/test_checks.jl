using Manifolds, Manopt, Plots, Test
# don't show plots actually
default(; show=false, reuse=true)

@testset "Test Gradient checks" begin
    M = Sphere(10)
    q = zeros(11)
    q[1] = 1.0
    p = zeros(11)
    p[1:4] .= 1 / sqrt(4)
    r = log(M, p, q)

    f(M, p) = 1 / 2 * distance(M, p, q)^2
    grad_f(M, p) = -log(M, p, q)

    @test check_gradient(M, f, grad_f, p, r)

    grad_f2(M, p) = -0.5 * log(M, p, q)
    @test_throws ErrorException check_gradient(M, f, grad_f2, p, r; throw_error=true)
    @test !check_gradient(M, f, grad_f2, p, r)

    check_gradient(M, f, grad_f, p, r; plot=true)

    #test windowsize error
    @test_throws ErrorException Manopt.find_best_slope_window(zeros(2), zeros(2), 20)
    @test_throws ErrorException Manopt.find_best_slope_window(zeros(2), zeros(2), [2, 20])
end
