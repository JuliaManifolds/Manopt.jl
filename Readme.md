<div align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/JuliaManifolds/Manopt.jl/master/docs/src/assets/logo-text-readme-dark.png">
      <img alt="Manifolds.jl logo with text on the side" src="https://raw.githubusercontent.com/JuliaManifolds/Manopt.jl/master/docs/src/assets/logo-text-readme.png">
    </picture>
</div>

## Optimization Algorithm on Riemannian Manifolds.

[![](https://img.shields.io/badge/docs-stable-blue?logo=Julia&logoColor=white)](https://manoptjl.org/stable)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)
[![CI](https://github.com/JuliaManifolds/Manopt.jl/workflows/CI/badge.svg)](https://github.com/JuliaManifolds/Manopt.jl/actions?query=workflow%3ACI+branch%3Amaster)
[![codecov](https://codecov.io/gh/JuliaManifolds/Manopt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaManifolds/Manopt.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

[![DOI](https://zenodo.org/badge/74746729.svg)](https://zenodo.org/badge/latestdoi/74746729)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.03866/status.svg)](https://doi.org/10.21105/joss.03866)

For a function $f: ℳ → ℝ$  that maps from a [Riemannian manifold](https://en.wikipedia.org/wiki/Riemannian_manifold)
ℳ to the real line, this package aims to solve

> Find the minimizer p on ℳ, that is, the (or a) point where f attains its minimum.

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

and then checkout the [🏔️ Get started with Manopt.jl](https://manoptjl.org/stable/tutorials/getstarted/) tutorial.

You can also watch an introduction given at JuliaCon 2022

[<img src ="https://img.youtube.com/vi/thbekfsyhCE/maxresdefault.jpg" style="width:66%; align:center;">](https://youtu.be/thbekfsyhCE)

or look at the slides from the presentation [here](https://ronnybergmann.net/talks/2022-JuliaCon-Manoptjl-extended.pdf).


## Related packages

Manopt.jl is based on [`ManifoldsBase.jl`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/),
hence the algorithms can be used with _any_ manifold following this interface for defining
a Riemannian manifold.

The following packages are related to `Manopt.jl`

* [`Manifolds.jl`](https://juliamanifolds.github.io/Manifolds.jl/stable/): a library of manifolds implemented using [`ManifoldsBase.jl`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/) :octocat: [GitHub repository](https://github.com/JuliaManifolds/Manifolds.jl)
* [`ManifoldsDiff.jl`](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/): a package to use (Euclidean) AD tools on manifolds, that also provides several differentials and gradients. :octocat: [GitHub repository](https://github.com/JuliaManifolds/ManifoldDiff.jl)
* [`JuMP.jl`](https://jump.dev/): can be used as interface to solve an optimization problem with Manopt. See [usage examples](https://manoptjl.org/stable/extensions/). :octocat: [GitHub repository](https://github.com/jump-dev/JuMP.jl)

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

To refer to a certain version or the source code in general please cite for example

```biblatex
@software{manoptjl-zenodo-mostrecent,
    Author    = {Ronny Bergmann},
    Copyright = {MIT License},
    Doi       = {10.5281/zenodo.4290905},
    Publisher = {Zenodo},
    Title     = {Manopt.jl},
    Year      = {2024},
}
```

for the most recent version or a corresponding version specific DOI, see [the list of all versions](https://zenodo.org/search?page=1&size=20&q=conceptrecid:%224290905%22&sort=-version&all_versions=True).

If you are also using [`Manifolds.jl`](https://juliamanifolds.github.io/Manifolds.jl/stable/) please consider to cite

```biblatex
@article{AxenBaranBergmannRzecki:2023,
    AUTHOR    = {Axen, Seth D. and Baran, Mateusz and Bergmann, Ronny and Rzecki, Krzysztof},
    ARTICLENO = {33},
    DOI       = {10.1145/3618296},
    JOURNAL   = {ACM Transactions on Mathematical Software},
    MONTH     = {dec},
    NUMBER    = {4},
    TITLE     = {Manifolds.jl: An Extensible Julia Framework for Data Analysis on Manifolds},
    VOLUME    = {49},
    YEAR      = {2023}
}
```

as well.
Note that all citations are in [BibLaTeX](https://ctan.org/pkg/biblatex) format.

`Manopt.jl` belongs to the Manopt family:

* [www.manopt.org](https://www.manopt.org): the MATLAB version of Manopt, see also their :octocat: [GitHub repository](https://github.com/NicolasBoumal/manopt)
* [www.pymanopt.org](https://www.pymanopt.org): the Python version of Manopt—providing also several AD backends, see also their :octocat: [GitHub repository](https://github.com/pymanopt/pymanopt)

Did you use `Manopt.jl` somewhere? Let us know! We'd love to collect those here as well.
