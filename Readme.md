# Manopt.jl

Optimization on Manifolds.

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://manoptjl.org/stable)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![CI](https://github.com/JuliaManifolds/Manopt.jl/workflows/CI/badge.svg)](https://github.com/JuliaManifolds/Manopt.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![codecov](https://codecov.io/gh/JuliaManifolds/Manopt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaManifolds/Manopt.jl)
[![DOI](https://zenodo.org/badge/74746729.svg)](https://zenodo.org/badge/latestdoi/74746729)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.03866/status.svg)](https://doi.org/10.21105/joss.03866)

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

## Citation

If you use `Manopt.jl`in your work, please cite the following

```biblatex
@article{Bergmann2022,
    Author    = {Ronny Bergmann},
    Doi       = {10.21105/joss.03866},
    Journal   = {Journal of Open Source Software},
    Number    = {70},
    Pages     = {3866},
    Publisher = {The Open Journal},
    Title     = {Manopt.jl: Optimization on Manifolds in {J}ulia},
    Volume    = {7},
    Year      = {2022},
}
```

To refer to a certain version or the source code in general we recommend to cite for example

```biblatex
@software{manoptjl-zenodo-mostrecent,
    Author = {Ronny Bergmann},
    Copyright = {MIT License},
    Doi = {10.5281/zenodo.4290905},
    Publisher = {Zenodo},
    Title = {Manopt.jl},
    Year = {2022},
}
```

for the most recent version or a corresponding version specific DOI, see [the list of all versions](https://zenodo.org/search?page=1&size=20&q=conceptrecid:%224290905%22&sort=-version&all_versions=True).
Note that both citations are in [BibLaTeX](https://ctan.org/pkg/biblatex) format.
