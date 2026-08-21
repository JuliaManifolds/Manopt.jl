```@raw html
---
layout: home

hero:
  name: Manopt.jl
  text: Optimization on Riemannian manifolds
  tagline: Efficient algorithms for minimizing a function on a Riemannian manifold
  actions:
    - theme: brand
      text: Get started
      link: tutorials/getstarted/index.html
    - theme: alt
      text: Available Solvers
      link: /solvers/index.html
    - theme: alt
      text: Developer Guide
      link: /base/index.html
  image:
    src: /logo.png            # primary image (light themes)
    alt: Manopt.jl         # accessibility text

features:
  - icon:
        light: /logo-manifoldsbase.png
        dark: /logo-manifoldsbase-dark.png
        alt: ManifoldsBase.jl
        wrap: true
    title: Generic Implementations
    details: Algorithms are implemented on a general Riemannian manifold using the interface from ManifoldsBase.jl.
    link: https://juliamanifolds.github.io/ManifoldsBase.jl/stable/
  - icon: 🧩
    title: Composable
    details: All components are designed in a modular fashion, for example problems, objectives, and solver states, but also stopping criteria and step sizes. They can be reused and combined easily. Points and tangent vectors can be represented using arbitrary Julia types.
  - icon: ⚡️
    title: Efficient
    details: All methods use in-place evaluations for both the objective and the manifold. There is also an advanced, but easy to use caching system further promoting efficiency.
  - icon: 📚
    title: Well-documented and -tested
    details: All algorithms are documented – both their theoretical foundation and all their options and keywords. The theoretical background also refers to further literature.
  - icon: 🛠️
    title: Customizable
    details: All solvers bring recording, debug and callback capabilities, the objectives can keep track of internal function calls.
  - icon:
        light: /logo-manifolds.png
        dark: /logo-manifolds-dark.png
        alt: Manifolds.jl
        wrap: true
    title: Use with Manifolds.jl
    details: If you have a certain manifold in mind, check out whether it is already provided in Manifolds.jl or LieGroups.jl.
    link: https://juliamanifolds.github.io/Manifolds.jl/stable/
---
```

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

or in other words: find the point ``p`` on the manifold ``\mathcal M``, where ``f`` reaches its minimal function value.

`Manopt.jl` provides a framework for optimization on manifolds as well as a library of optimization algorithms in [Julia](https://julialang.org).
It belongs to the **Manopt family**, which includes [Manopt](https://manopt.org) (Matlab) and [pymanopt.org](https://www.pymanopt.org/) (Python), both aiming to provide the same framework
in the flavour of the corresponding language.

## Get Started

To get started with `Manopt.jl`, start [Julia](https://julialang.org) and type

```julia-repl
] add Manopt
```

to install the package. Then you can dive directly into optimization on manifolds, following the
[🏔️ Get started with Manopt.jl](tutorials/getstarted.md) tutorial.

`Manopt.jl` makes it easy to use an algorithm for your favourite
manifold as well as a manifold for your favourite algorithm. It already provides
many manifolds and algorithms, which can easily be enhanced, for example to
[record](@ref sec-record) certain data or
[debug output](@ref sec-debug) throughout iterations.

If you use `Manopt.jl` in your work, please cite the following:

> _Bergmann, R._ (2022).
> **Manopt.jl: Optimization on Manifolds in Julia**,
> Journal of Open Source Software, 7(70), 3866.
>
> doi: [10.21105/joss.03866](https://doi.org/10.21105/joss.03866)

```@raw html
<details><summary><code>Bergmann:2022</code> (BibLaTeX) </summary>
```
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
```@raw html
</details>
```

To refer to a certain version or the source code in general, cite for example

> _Bergmann, R._ (2026). **Manopt.jl**, Zenodo.
>
> doi: [10.5281/zenodo.4290905](https://doi.org/10.5281/zenodo.4290905)

```@raw html
<details><summary><code>Manoptjl-zenodo-mostrecent</code> (BibLaTeX) </summary>
```
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
```@raw html
</details>
```

for the most recent version or a corresponding version specific DOI, see [the list of all versions](https://zenodo.org/search?page=1&size=20&q=conceptrecid:%224290905%22&sort=-version&all_versions=True).

If you are also using [`Manifolds.jl`](https://juliamanifolds.github.io/Manifolds.jl/stable/), please consider citing

> _Axen, S. D., Baran, M., Bergmann, R., Rzecki, K._ (2023).
> **Manifolds.jl: An Extensible Julia Framework for Data Analysis on Manifolds**,
> ACM Transactions on Mathematical Software, Volume 49, Issue 4, Article No. 33, pages 1–23.
>
> doi: [10.1145/3618296](https://doi.org/10.1145/3618296),
> arXiv: [2106.08777](https://arxiv.org/abs/2106.08777)

```@raw html
<details><summary><code>AxenBaranBergmannRzecki:2023</code> (BibLaTeX) </summary>
```
```biblatex
@article{AxenBaranBergmannRzecki:2023,
    AUTHOR    = {Axen, Seth D. and Baran, Mateusz and Bergmann, Ronny and Rzecki, Krzysztof},
    ARTICLENO = {33},
    DOI       = {10.1145/3618296},
    JOURNAL   = {ACM Transactions on Mathematical Software},
    MONTH     = {dec},
    NUMBER    = {4},
    TITLE     = {{Manifolds.jl}: An Extensible {J}ulia Framework for Data Analysis on Manifolds},
    VOLUME    = {49},
    YEAR      = {2023}
}
```
```@raw html
</details>
```

## Main features

### Optimization algorithms (solvers)

For every optimization algorithm, a [solver](solvers/index.md) is implemented based on an [`AbstractManoptProblem`](@ref) that describes the problem to solve and its [`AbstractManoptSolverState`](@ref) that sets up the solver, and stores values that are required between or for the next iteration.

### Manifolds

This project is built upon [ManifoldsBase.jl](@extref ManifoldsBase :doc:`index`), a generic interface to implement manifolds. Certain functions are extended for specific manifolds from [Manifolds.jl](@extref Manifolds :std:doc:`index`), but all other manifolds from that package can be used here, too.

The notation in the documentation aims to follow the [notation](@extref Manifolds :std:doc:`misc/notation`) of these packages.

### Algorithm exploration

To visualize and interpret results, `Manopt.jl` provides a system to get [debug output](base/state/debug.md) during the iterations of an algorithm as well as [record](base/state/record.md) capabilities, for example to record a specified tuple of values per iteration, most prominently [`RecordCost`](@ref) and
[`RecordIterate`](@ref). Take a look at the [🏔️ Get started with Manopt.jl](tutorials/getstarted.md) tutorial on how to easily activate this.

## Literature

If you want to get started with manifolds, a recommended reference is the book [doCarmo:1992](@cite),
and if you want to directly dive into optimization on manifolds, good references are
[AbsilMahonySepulchre:2008](@cite) and [Boumal:2023](@cite),
which are both available online for free.

```@bibliography
Pages = ["index.md"]
Canonical = false
```
