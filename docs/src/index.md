# Welcome to Manopt.jl

```@meta
CurrentModule = Manopt
```

```@docs
Manopt.Manopt
```

For a function $f:\mathcal M → ℝ$ defined on a [Riemannian manifold](https://en.wikipedia.org/wiki/Riemannian_manifold) $\mathcal M$ we aim to solve

```math
\operatorname*{argmin}_{p ∈ \mathcal M} f(p),
```

or in other words: find the point $x$ on the manifold, where $f$ reaches its minimal function value.

`Manopt.jl` provides a framework for optimization on manifolds as well as a Library of optimization algorithms in [Julia](https://julialang.org).
It belongs to the “Manopt family”, which includes [Manopt](https://manopt.org) (Matlab) and [pymanopt.org](https://www.pymanopt.org/) (Python).

If you want to delve right into `Manopt.jl` check out the
[Get started: Optimize!](tutorials/Optimize!.md) tutorial.

`Manopt.jl` makes it easy to use an algorithm for your favourite
manifold as well as a manifold for your favourite algorithm. It already provides
many manifolds and algorithms, which can easily be enhanced, for example to
[record](@ref RecordSection) certain data or
[debug output](@ref DebugSection) throughout iterations.

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

## Main Features

### Optimization Algorithms (Solvers)

For every optimization algorithm, a [solver](@ref SolversSection) is implemented based on a [`AbstractManoptProblem`](@ref) that describes the problem to solve and its [`AbstractManoptSolverState`](@ref) that set up the solver, store interims values. Together they
form a [plan](@ref planSection).

## Manifolds

This project is build upon [ManifoldsBase.jl](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html), a generic interface to implement manifolds. Certain functions are extended for specific manifolds from [Manifolds.jl](https://juliamanifolds.github.io/Manifolds.jl/stable/), but all other manifolds from that package can be used here, too.

The notation in the documentation aims to follow the same [notation](https://juliamanifolds.github.io/Manifolds.jl/stable/notation.html) from these packages.

### Functions on Manifolds

Several functions are available, implemented on an arbitrary manifold, [cost functions](@ref CostFunctions), [differentials](@ref DifferentialFunctions) and their [adjoints](@ref adjointDifferentialFunctions), and [gradients](@ref GradientFunctions) as well as [proximal maps](@ref proximalMapFunctions).

### Visualization

To visualize and interpret results, `Manopt.jl` aims to provide both easy plot functions as well as [exports](@ref Exports). Furthermore a system to get [debug](@ref DebugSection) during the iterations of an algorithms as well as [record](@ref RecordSection) capabilities, i.e. to record a specified tuple of values per iteration, most prominently [`RecordCost`](@ref) and
[`RecordIterate`](@ref). Take a look at the [Get Started: Optimize!](tutorials/Optimize!.md) tutorial on how to easily activate this.

## Literature

If you want to get started with manifolds, one book is [[do Carmo, 1992](#doCarmo1992)],
and if you want do directly dive into optimization on manifolds, my favorites references are
[[Absil, Mahony, Sepulchre, 2008](#AbsilMahonySepulchre2008)] and [[Boumal 2023](#Boumal2023)],
which are both available online for free

```@raw html
<ul>
<li id="AbsilMahonySepulchre2008">
    [<a>Absil, Mahony, Sepulchre, 2008</a>]
    P.-A. Absil, R. Mahony and R. Sepulchre,
    <emph>Optimization Algorithms on Matrix Manifolds</emph>,
    Princeton University Press, 2008,
    doi: <a href="https://doi.org/10.1515/9781400830244">10.1515/9781400830244</a>,
    <a href="http://press.princeton.edu/chapters/absil/">open access</a>.
</li>
<li id="Boumal2023">
    [<a>Boumal, 2022</a>]
    N. Boumal,
    <emph>An introduction to optimization on smooth manifolds</emph>,
    to appear with Cambridge University Press, 2023,
    <a href="https://www.nicolasboumal.net/book/index.html">open access</a>.
</li>
<li id="doCarmo1992">
    [<a>doCarmo, 1992</a>]
    M. P. do Carmo,
    <emph>Riemannian Geometry</emph>,
    Birkhäuser Boston, 1992,
    ISBN: 0-8176-3490-8.
</li>
</ul>
```
