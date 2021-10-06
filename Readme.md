# Manopt.jl

Optimization on Manifolds.

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://manoptjl.org/stable)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![CI](https://github.com/JuliaManifolds/Manopt.jl/workflows/CI/badge.svg)](https://github.com/JuliaManifolds/Manopt.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![codecov](https://codecov.io/gh/JuliaManifolds/Manopt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaManifolds/Manopt.jl)
[![DOI](https://zenodo.org/badge/74746729.svg)](https://zenodo.org/badge/latestdoi/74746729)
[![status](https://joss.theoj.org/papers/803fd5cc2034643ea1476d45f9df669b/status.svg)](https://joss.theoj.org/papers/803fd5cc2034643ea1476d45f9df669b)

For a function f that maps from a [Riemannian manifold](https://en.wikipedia.org/wiki/Riemannian_manifold)
ℳ onto the real line, we aim to solve

> Find the minimizer x on ℳ, i.e. the (or a) point where f attains its minimum.

`Manopt.jl` provides a framework for optimization on manifolds.
Based on [Manopt](https://manopt.org) and
[MVIRT](https://ronnybergmann.net/mvirt/), both implemented in Matlab,
this toolbox aims to provide an easy access to optimization methods on manifolds
for [Julia](https://julialang.org), including example data and visualization methods.

## Getting started

In Julia you can get started by just typing

```julia
] add Manopt
```

then checkout the [Get Started: Optimize!](https://manoptjl.org/stable/tutorials/MeanAndMedian.html) tutorial or the
[examples](https://github.com/JuliaManifolds/Manopt.jl/tree/master/examples)
in this repository.
