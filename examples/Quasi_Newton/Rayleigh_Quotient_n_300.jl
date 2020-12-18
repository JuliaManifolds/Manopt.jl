A_n_300 = randn(300, 300)
A_n_300 = (A_n_300 + A_n_300') / 2
M_n_300 = Sphere(299)
F_n_300(X::Array{Float64,1}) = X' * A_n_300 * X
∇F_n_300(X::Array{Float64,1}) = 2 * (A_n_300 * X - X * X' * A_n_300 * X)
sample_times_n_300 = []

for i in 1:10
    x = random_point(M_n_300)
    bench_n_300 = @benchmark quasi_Newton(
        M_n_300,
        F_n_300,
        ∇F_n_300,
        $x;
        memory_size=-1,
        stopping_criterion=StopWhenGradientNormLess(
            norm(M_n_300, $x, ∇F_n_300($x)) * 10^(-6)
        ),
    ) seconds = 600 samples = 10 evals = 1
    append!(sample_times_n_300, bench_n_300.times)
end

result_n_300 = sum(sample_times_n_300) / length(sample_times_n_300) / 1000000000
