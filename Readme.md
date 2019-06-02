# Manopt.jl

Optimization on Manifolds.


[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://kellertuer.github.io/Manopt.jl/stable)
[![Build Status](https://travis-ci.com/kellertuer/Manopt.jl.svg?branch=master)](https://travis-ci.com/kellertuer/Manopt.jl)
[![codecov](https://codecov.io/gh/kellertuer/Manopt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kellertuer/Manopt.jl)

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

then checkout the [Getting Started: Optimize!](https://kellertuer.github.io/Manopt.jl/stable/tutorials/MeanAndMedian/) tutorial or the 
[examples](https://github.com/kellertuer/Manopt.jl/tree/master/src/examples)
in this repository, where you might want to adapt the `resultsFolder` string.
You can also read the [documentation](https://kellertuer.github.io/Manopt.jl/stable).

