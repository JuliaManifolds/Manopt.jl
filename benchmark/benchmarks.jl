#
# Benchmark suite for `Manopt.jl`, see `benchmark/Readme.md` for how to run it.
#
# This is the entry point both for local runs and for AirspeedVelocity, whose
# runner requires the benchmark group to be called `SUITE`.
#
# `problems/` defines the problems to run a solver on, `suites/` contains the
# benchmarks themselves, one file per solver, each of which can also be run on
# its own.
#
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

using BenchmarkTools

const SUITE = BenchmarkGroup()

# The problems the benchmarks run on, each in a module of its own, so that the
# suites below can share them.
include(joinpath(@__DIR__, "problems", "riemannian_mean.jl"))
include(joinpath(@__DIR__, "problems", "riemannian_median.jl"))

# One `include` and one mount per suite. The two are deliberately kept as
# separate statements: on Julia 1.12 including a file and accessing the module
# it defines within the same statement warns about the binding's world age.
include(joinpath(@__DIR__, "suites", "gradient_descent.jl"))
SUITE["gradient_descent"] = GradientDescentSuite.SUITE

include(joinpath(@__DIR__, "suites", "quasi_Newton.jl"))
SUITE["quasi_Newton"] = QuasiNewtonSuite.SUITE

include(joinpath(@__DIR__, "suites", "cyclic_proximal_point.jl"))
SUITE["cyclic_proximal_point"] = CyclicProximalPointSuite.SUITE

# Run everything when this file is the script being run, but not when it is
# included, be that from the REPL or by AirspeedVelocity.
if abspath(PROGRAM_FILE) == @__FILE__
    display(median(run(SUITE; verbose = true)))
    println()
end
