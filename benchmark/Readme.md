# Benchmarks for `Manopt.jl`

Written with [BenchmarkTools.jl](https://juliaci.github.io/BenchmarkTools.jl/stable/),
compared between two revisions with [AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl).
`problems/` are modules containing a problem each, `suites/` contains the benchmarks, one file per
solver, and `benchmarks.jl` collects them all in `SUITE`.

## Running

```
julia benchmark/benchmarks.jl                      # everything
julia benchmark/suites/gradient_descent.jl         # one solver
```

Including `benchmarks.jl` in the REPL provides `SUITE`, so a single benchmark can
be run with `run(SUITE["gradient_descent"]["riemannian-mean/allocating"])`.

## Comparing two revisions

With `benchpkg` from `AirspeedVelocity.jl`, where `dirty` is the working copy and
`--filter` selects by a part of the `group/name` key:

```
benchpkg Manopt --rev=master,dirty --bench-on=dirty
benchpkgtable Manopt --rev=master,dirty --ratio
```

## Adding a benchmark

Copy `suites/gradient_descent.jl`, adapt it to the new solver and mount it in
`benchmarks.jl`.
You can start defining the problem to benchmark in the same module. If you aim to reuse a problem definition, consider moving it to its own module in `problems/`.
