#!/usr/bin/env julia
#
#

if "--help" ∈ ARGS
    println(
        """
        benchmark/benchmarks.jl

        Run the benchmarks for `Manopt.jl` with optional arguments

        Arguments
        * `--help`              - print this help and exit without rendering the documentation
        """
    )
    exit(0)
end

# If the Benchmark environment is not the active one: Activate and instantiate it it
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

using BenchmarkTools

const manopt_suite = BenchmarkGroup()

manopt_suite["gradient_descent"] = @benchmarkable include("gradient_descent.jl")

tune!(manopt_suite)
