# Manopt.jl

Optimization on Manifolds.


[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://manoptjl.org/stable)
[![Build Status](https://travis-ci.com/JuliaManifolds/Manopt.jl.svg?branch=master)](https://travis-ci.com/JuliaManifolds/Manopt.jl)
[![codecov](https://codecov.io/gh/JuliaManifolds/Manopt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaManifolds/Manopt.jl)
[![DOI](https://zenodo.org/badge/74746729.svg)](https://zenodo.org/badge/latestdoi/74746729)

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
in this repository, where you might want to adapt the `resultsFolder` string.
You can also read the [documentation](https://www.manoptjl.org/stable).
