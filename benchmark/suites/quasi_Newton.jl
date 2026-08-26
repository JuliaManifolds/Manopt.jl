if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate()
end
isdefined(@__MODULE__, :RiemannianMean) ||
    include(joinpath(dirname(@__DIR__), "problems", "riemannian_mean.jl"))

"""
    QuasiNewtonSuite

Benchmarks of `quasi_Newton` on the `RiemannianMean` problem, which compared to
`gradient_descent` additionally cover the limited memory direction update.
"""
module QuasiNewtonSuite

    using BenchmarkTools
    using Manopt
    using ..RiemannianMean: M, f, grad_f, p0, sc

    const SUITE = BenchmarkGroup()

    for memory in (4, 20)
        SUITE["riemannian-mean/lbfgs-memory-$(memory)"] = @benchmarkable(
            quasi_Newton($M, $f, $grad_f, q; memory_size = $memory, stopping_criterion = $sc),
            setup = (q = copy($p0)), evals = 1,
        )
    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    using BenchmarkTools
    display(median(run(QuasiNewtonSuite.SUITE; verbose = true)))
    println()
end
