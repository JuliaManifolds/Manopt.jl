# [The Stiefel manifold $\mathrm{St}(k,n)$ embedded in $\mathbb K^{n\times k}$](@id StiefelManifold)

The manifold $\mathcal M = \mathrm{St}(k,n)$ is an embedded submanifold of
$\mathbb{K}^{n×k}$, wich represents all orthonormal k-frames in
$\mathbb{K}^{n}$. The set can be written as

$\mathrm{St}(k,n) = \bigl\{ x \in \mathbb{K}^{n\times k} \big| {\bar x}^{\mathrm{T}}x = I_k \bigl\}$


The Stiefel manifold $\mathrm{St}(k,n)$ can be thought of as a set of $n×k$
matrices by writing a $k$-frame as a matrix of $k$ column vectors in
$\mathbb{K}^{n}$. It is named after the mathematician [Eduard Stiefel](https://de.wikipedia.org/wiki/Eduard_Stiefel) (1909–1978).
The Stiefel manifold $\mathcal M = \mathrm{St}(k,n)$ posesses the following instances of the
abstract types [`Manifold`](@ref), [`MPoint`](@ref), and [`TVector`](@ref).

```@autodocs
Modules = [Manopt]
Pages = ["manifolds/Stiefel.jl"]
Order = [:type]
```

## Functions

```@autodocs
Modules = [Manopt]
Pages = ["manifolds/Stiefel.jl"]
Order = [:function]
```
