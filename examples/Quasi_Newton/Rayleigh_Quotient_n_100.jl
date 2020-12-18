A_n_100 = randn(100, 100)
A_n_100 = (A_n_100 + A_n_100') / 2
M_n_100 = Sphere(99)
F_n_100(X::Array{Float64,1}) = X' * A_n_100 * X
∇F_n_100(X::Array{Float64,1}) = 2 * (A_n_100 * X - X * X' * A_n_100 * X)
sample_times_n_100 = []

for i in 1:10
    x = random_point(M_n_100)
    bench_n_100 = @benchmark quasi_Newton(
        M_n_100,
        F_n_100,
        ∇F_n_100,
        $x;
        memory_size=-1,
        stopping_criterion=StopWhenGradientNormLess(
            norm(M_n_100, $x, ∇F_n_100($x)) * 10^(-6)
        ),
    ) seconds = 600 samples = 10 evals = 1
    append!(sample_times_n_100, bench_n_100.times)
end

result_n_100 = sum(sample_times_n_100) / length(sample_times_n_100) / 1000000000
