#
# Benchmarks for `gradient_descent`.
#
# Run `julia benchmark/suites/gradient_descent.jl` for these benchmarks alone, or
# `julia benchmark/benchmarks.jl` for the whole suite.
#
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate()
end
isdefined(@__MODULE__, :RiemannianMean) ||
    include(joinpath(dirname(@__DIR__), "problems", "riemannian_mean.jl"))

module GradientDescentSuite

    using BenchmarkTools
    using Manopt
    using ..RiemannianMean: M, f, grad_f, grad_f!, p0, sc

    const SUITE = BenchmarkGroup()

    SUITE["riemannian-mean/allocating"] = @benchmarkable(
        gradient_descent($M, $f, $grad_f, q; stopping_criterion = $sc),
        setup = (q = copy($p0)), evals = 1,
    )
    SUITE["riemannian-mean/inplace"] = @benchmarkable(
        gradient_descent!(
            $M, $f, $grad_f!, q;
            evaluation = InplaceEvaluation(), stopping_criterion = $sc,
        ),
        setup = (q = copy($p0)), evals = 1,
    )
    SUITE["riemannian-mean/cached"] = @benchmarkable(
        gradient_descent($M, $f, $grad_f, q; stopping_criterion = $sc, cache = :Simple),
        setup = (q = copy($p0)), evals = 1,
    )

end

if abspath(PROGRAM_FILE) == @__FILE__
    using BenchmarkTools
    display(median(run(GradientDescentSuite.SUITE; verbose = true)))
    println()
end
