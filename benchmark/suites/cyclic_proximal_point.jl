#
# Benchmarks for `cyclic_proximal_point`.
#
# `evals = 1`, since with more than one evaluation per sample the in-place
# variant would restart from an already converged point.
#
# Run `julia benchmark/suites/cyclic_proximal_point.jl` for these benchmarks
# alone, or `julia benchmark/benchmarks.jl` for the whole suite.
#
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate()
end
isdefined(@__MODULE__, :RiemannianMedian) ||
    include(joinpath(dirname(@__DIR__), "problems", "riemannian_median.jl"))

module CyclicProximalPointSuite

    using BenchmarkTools
    using Manopt
    using ..RiemannianMedian: M, f, proxes, proxes!, p0, sc

    const SUITE = BenchmarkGroup()

    SUITE["riemannian-median/allocating"] = @benchmarkable(
        cyclic_proximal_point($M, $f, $proxes, q; stopping_criterion = $sc),
        setup = (q = copy($M, $p0)), evals = 1,
    )
    SUITE["riemannian-median/inplace"] = @benchmarkable(
        cyclic_proximal_point!(
            $M, $f, $proxes!, q;
            evaluation = InplaceEvaluation(), stopping_criterion = $sc,
        ),
        setup = (q = copy($M, $p0)), evals = 1,
    )

end

if abspath(PROGRAM_FILE) == @__FILE__
    using BenchmarkTools
    display(median(run(CyclicProximalPointSuite.SUITE; verbose = true)))
    println()
end
