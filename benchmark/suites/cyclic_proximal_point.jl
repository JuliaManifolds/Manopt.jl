if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(dirname(@__DIR__))
    Pkg.instantiate()
end
isdefined(@__MODULE__, :RiemannianMedian) ||
    include(joinpath(dirname(@__DIR__), "problems", "riemannian_median.jl"))

"""
    CyclicProximalPointSuite

Benchmarks of `cyclic_proximal_point` on the `RiemannianMedian` problem.
"""
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
