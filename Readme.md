# Manopt.jl

Optimization Algorithm on Riemannian Manifolds.

[![](https://img.shields.io/badge/docs-stable-blue?logo=Julia&logoColor=white)](https://manoptjl.org/stable)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![CI](https://github.com/JuliaManifolds/Manopt.jl/workflows/CI/badge.svg)](https://github.com/JuliaManifolds/Manopt.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![codecov](https://codecov.io/gh/JuliaManifolds/Manopt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaManifolds/Manopt.jl)
[![DOI](https://zenodo.org/badge/74746729.svg)](https://zenodo.org/badge/latestdoi/74746729)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.03866/status.svg)](https://doi.org/10.21105/joss.03866)

For a function $f: ℳ → ℝ$  that maps from a [Riemannian manifold](https://en.wikipedia.org/wiki/Riemannian_manifold)
ℳ to the real line, we aim to solve

> Find the minimizer p on ℳ, i.e. the (or a) point where f attains its minimum.

`Manopt.jl` provides

* A framework to implement arbitrary optimization algorithms on Riemannian Manifolds
* A library of optimization algorithms on Riemannian manifolds
* an easy-to-use interface for (debug) output and recording values during an algorithm run.
* several tools to investigate the algorithms, gradients, and optimality criteria

## Getting started

In Julia you can get started by just typing

```julia
using Pkg; Pkg.add("Manopt");
```

and then checkout the [Get Started: Optimize!](https://manoptjl.org/stable/tutorials/Optimize!/) tutorial.

## Related packages

Manopt.jl is based on [`ManifoldsBase.jl`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/),
hence the algorithms can be used with _any_ manifold following this interface for defining
a Riemannian manifold.

The following packages are related to `Manopt.jl`

* [`Manifolds.jl`](https://juliamanifolds.github.io/Manifolds.jl/stable/) – a library of manifolds implemented using [`ManifoldsBase.jl`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/) :octocat: [GitHub repository](https://github.com/JuliaManifolds/Manifolds.jl)
* [`ManifoldsDiff.jl`](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/) – a package to use (Euclidean) AD tools on manifolds, that also provides several differentials and gradients. :octocat: [GitHub repository](https://github.com/JuliaManifolds/ManifoldDiff.jl)
* [`JuMP.jl`](https://jump.dev/) can be used as interface to solve an optimization problem with Manopt. See [usage examples](https://manoptjl.org/stable/extensions/). :octocat: [GitHub repository](https://github.com/jump-dev/JuMP.jl)

## Citation

If you use `Manopt.jl` in your work, please cite the following

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

If you are also using [`Manifolds.jl`](https://juliamanifolds.github.io/Manifolds.jl/stable/) please consider to cite

```
@article{AxenBaranBergmannRzecki:2023,
    AUTHOR     = {Seth D. Axen and Mateusz Baran and Ronny Bergmann and Krzysztof Rzecki},
    DOI        = {10.1145/3618296},
    EPRINT     = {2021.08777},
    EPRINTTYPE = {arXiv},
    JOURNAL    = {AMS Transactions on Mathematical Software},
    NOTE       = {accepted for publication},
    TITLE      = {Manifolds.jl: An Extensible {J}ulia Framework for Data Analysis on Manifolds},
    YEAR       = {2023}
}
```

as well.
Note that all citations are in [BibLaTeX](https://ctan.org/pkg/biblatex) format.

## Further and Similar Packages & Links

`Manopt.jl` belongs to the Manopt family:

*  [manopt.org](https://www.manopt.org) – The Matlab version of Manopt, see also their :octocat: [GitHub repository](https://github.com/NicolasBoumal/manopt)
* [pymanopt.org](https://www.pymanopt.org/) – The Python version of Manopt – providing also several AD backends, see also their :octocat: [GitHub repository](https://github.com/pymanopt/pymanopt)

but there are also more packages providing tools on manifolds:

* [Jax Geometry](https://bitbucket.org/stefansommer/jaxgeometry/src/main/) (Python/Jax) for differential geometry and stochastic dynamics with deep learning
* [Geomstats](https://geomstats.github.io) (Python with several backends) focusing on statistics and machine learning :octocat: [GitHub repository](https://github.com/geomstats/geomstats)
* [Geoopt](https://geoopt.readthedocs.io/en/latest/) (Python & PyTorch) – Riemannian ADAM & SGD. :octocat: [GitHub repository](https://github.com/geoopt/geoopt)
* [McTorch](https://github.com/mctorch/mctorch) (Python & PyToch) – Riemannian SGD, Adagrad, ASA & CG.
* [ROPTLIB](https://www.math.fsu.edu/~whuang2/papers/ROPTLIB.htm) (C++) a Riemannian OPTimization LIBrary :octocat: [GitHub repository](https://github.com/whuang08/ROPTLIB)
* [TF Riemopt](https://github.com/master/tensorflow-riemopt) (Python & TensorFlow) Riemannian optimization using TensorFlow

Did you use `Manopt.jl` somewhere? Let us know! We'd love to collect those here as well.