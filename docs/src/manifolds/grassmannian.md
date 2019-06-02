# [The Grassmannian manifold $\mathrm{Gr}(k,n)$ embedded in $\mathbb R^{n\times k}$](@id GrassmannianManifold)

The manifold $\mathcal M = \mathrm{Gr}(k,n)$ of the set of k-dimensional
subspaces in $\mathbb{K}^{n}$. This set can be written as

```math
\mathrm{Gr}(k,n) = \bigl\{ \operatorname{span}(x)
\ \big|\ x \in \mathbb{K}^{n\times k}, {\bar x}^{\mathrm{T}}x = \mathrm{I}_k \bigr\}.
```

Here we consider the vector spaces $\mathbb{K}^{k}$, where $\mathbb{K}$ is equal
to $\mathbb{C}$ or $\mathbb{R}$, which are subspaces of $\mathbb{K}^{n}$ with
$n\geq k$. Thus all the manifolds produced are compact and smooth. The manifold
is named after [Hermann Günther Graßmann](https://en.wikipedia.org/wiki/Hermann_Grassmann) (1809-1877).
The Grassmannian manifold $\mathcal M = \mathrm{Gr}(k,n)$ possesses the following
instances of the abstract types [`Manifold`](@ref), [`MPoint`](@ref),
and [`TVector`](@ref).

```@autodocs
Modules = [Manopt]
Pages = ["manifolds/Grassmannian.jl"]
Order = [:type]
```

## Functions

```@autodocs
Modules = [Manopt]
Pages = ["manifolds/Grassmannian.jl"]
Order = [:function]
```
