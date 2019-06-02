# [The Manifold $\operatorname{SO}(n)$ of rotations](@id SOn)

The manifold $\mathcal M = \mathrm{SO}(n)$ of orthogonal matrices with
determinant $+1$ in $\mathbb R^{n\times n}$, i.e.

$\mathrm{SO}(n) = \bigl\{R \in \mathbb{R}^{n\times n} \big| RR^{\mathrm{T}} =
R^{\mathrm{T}}R = \mathrm{I}_n, \operatorname{det}(R) = 1 \bigr\}$

The $\mathrm{SO}(n)$ is a subgroup of the orthogonal group $\mathrm{O}(n)$
and also known as the special orthogonal group or the set of rotations group.

## Applications

The manifold of rotations appears for example in
[Electron Backscatter diffraction](https://en.wikipedia.org/wiki/Electron_backscatter_diffraction) (EBSD), where orientations (modulo a symmetry group) are measured. For more details on symmetry groups, see for
example the [MTEX](http://mtex-toolbox.github.io/) toolbox, where several image
processing methods are implemented on $\mathrm{SO}(3)$-valued data, taking also
the symmetries in the crystal orientations into account. 

A paper concerned with discrete regression on $\mathrm{SO}(n)$ is

> Boumal, N; Absil, P.–A.: _A discrete Recgression Method on Manifolds and its Application to Data on $\mathrm{SO}(n)$,_
> IFAC Proceedings Volume 44, Issue 1, pp 2284–2289. doi: [10.3182/20110828-6-IT-1002.00542](https://doi.org/10.3182/20110828-6-IT-1002.00542)

which also includes the formulae for several functions implemented for this manifold within `Manopt.jl`. 

## Types

The manifold posesses the following instances of the abstract types
[`Manifold`](@ref), [`MPoint`](@ref), and [`TVector`](@ref).

```@autodocs
Modules = [Manopt]
Pages = ["manifolds/Rotations.jl"]
Order = [:type]
```

## Functions

```@autodocs
Modules = [Manopt]
Pages = ["manifolds/Rotations.jl"]
Order = [:function]
```
