using Manopt
using Manifolds, ManifoldDiff, Random
using ManifoldDiff: grad_distance

Random.seed!(42)
m = 30
M = Sphere(m)
n = 800
σ = π / 8
p = zeros(Float64, m + 1)
p[2] = 1.0
data = [exp(M, p, σ * rand(M; vector_at = p)) for i in 1:n];

f(M, p) = sum(1 / (2 * n) * distance.(Ref(M), Ref(p), data) .^ 2)
grad_f(M, p) = sum(1 / n * grad_distance.(Ref(M), data, Ref(p)))

sc = StopWhenGradientNormLess(5.0e-9)
p0 = zeros(Float64, m + 1); p0[1] = 1 / sqrt(2); p0[2] = 1 / sqrt(2)
m1 = gradient_descent(M, f, grad_f, p0; stopping_criterion = sc);

gradient_descent(M, f, grad_f, p0; stopping_criterion = sc)
