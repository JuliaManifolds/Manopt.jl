# Welcome to Manopt.jl

```@meta
CurrentModule = Manopt
```

```@docs
Manopt.Manopt
```

For a function ``f:\mathcal M → ℝ`` defined on a [Riemannian manifold](https://en.wikipedia.org/wiki/Riemannian_manifold) ``\mathcal M`` algorithms in this package aim to solve

```math
\operatorname*{argmin}_{p ∈ \mathcal M} f(p),
```

or in other words: find the point ``p`` on the manifold, where ``f`` reaches its minimal function value.

`Manopt.jl` provides a framework for optimization on manifolds as well as a Library of optimization algorithms in [Julia](https://julialang.org).
It belongs to the “Manopt family”, which includes [Manopt](https://manopt.org) (Matlab) and [pymanopt.org](https://www.pymanopt.org/) (Python).

If you want to delve right into `Manopt.jl` read the
[🏔️ Get started with Manopt.jl](tutorials/getstarted.md) tutorial.

`Manopt.jl` makes it easy to use an algorithm for your favourite
manifold as well as a manifold for your favourite algorithm. It already provides
many manifolds and algorithms, which can easily be enhanced, for example to
[record](@ref sec-record) certain data or
[debug output](@ref sec-debug) throughout iterations.

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

To refer to a certain version or the source code in general cite for example

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
    TITLE     = {Manifolds.Jl: An Extensible Julia Framework for Data Analysis on Manifolds},
    VOLUME    = {49},
    YEAR      = {2023}
}
```

Note that both citations are in [BibLaTeX](https://ctan.org/pkg/biblatex) format.

## Main features

### Optimization algorithms (solvers)

For every optimization algorithm, a [solver](solvers/index.md) is implemented based on a [`AbstractManoptProblem`](@ref) that describes the problem to solve and its [`AbstractManoptSolverState`](@ref) that set up the solver, and stores values that are required between or for the next iteration.
Together they form a [plan](plans/index.md).

## Manifolds

This project is build upon [ManifoldsBase.jl](@extref ManifoldsBase :doc:`index`), a generic interface to implement manifolds. Certain functions are extended for specific manifolds from [Manifolds.jl](https://juliamanifolds.github.io/Manifolds.jl/stable/), but all other manifolds from that package can be used here, too.

The notation in the documentation aims to follow the same [notation](https://juliamanifolds.github.io/Manifolds.jl/stable/misc/notation.html) from these packages.

### Visualization

To visualize and interpret results, `Manopt.jl` aims to provide both easy plot functions as well as [exports](helpers/exports.md). Furthermore a system to get [debug](plans/debug.md) during the iterations of an algorithms as well as [record](plans/record.md) capabilities, for example to record a specified tuple of values per iteration, most prominently [`RecordCost`](@ref) and
[`RecordIterate`](@ref). Take a look at the [🏔️ Get started with Manopt.jl](tutorials/getstarted.md) tutorial on how to easily activate this.

## Literature

If you want to get started with manifolds, one book is [doCarmo:1992](@cite),
and if you want do directly dive into optimization on manifolds, good references are
[AbsilMahonySepulchre:2008](@cite) and [Boumal:2023](@cite),
which are both available online for free

```@bibliography
Pages = ["index.md"]
Canonical=false
```