# [Special Manifolds build upon one or more Riemannian manifolds](@id CombinedManifolds)

```@meta
CurrentModule = Manopt
```

## [Tangent bundle](@id SubSecTangentBundle)

The tangent bundle $T\mathcal M$ of a manifold $\mathcal M$ consists of all tuples
$(x,\xi) \in T\mathcal M$, where $\xi\in T_x\mathcal M$, $x\in \mathcal M$, where
the metric is inherited component wise and for the exponential and logarithmic map,
the second component requires a [`parallelTransport`](@ref).

### Tangent Bundle Types

```@autodocs
Modules = [Manopt]
Pages = ["manifolds/TangentBundle.jl"]
Order = [:type]
```

### Tangent Bundle Functions

```@autodocs
Modules = [Manopt]
Pages = ["manifolds/TangentBundle.jl"]
Order = [:function]
```

## Power Manifold

The product manifold $\mathcal M^n$, where $n\in\mathbb N^k$ represents
arrays that are manifold-valued, for example, if $n$ is a number ($k=1$)
we obtain a manifold-valued signal $f\in\mathcal M^n$.
Many operations are performed element wise, while for example the distance
on the power manifold is the $\ell^2$ norm of the element wise distances.

### Power Manifold Types

```@autodocs
Modules = [Manopt]
Pages = ["manifolds/Power.jl"]
Order = [:type]
```

### Power Manifold Functions

```@autodocs
Modules = [Manopt]
Pages = ["manifolds/Power.jl"]
Order = [:function]
```

## Product Manifold

A little more general is the product manifold, where
$\mathcal M = \mathcal N_1\times\cdots\times\mathcal N_n$, $n\in\mathbb N^k$
is a product of manifolds, i.e. for a value $f\in\mathcal M$ we have that
$f_i\in\mathcal N_i$, where $i$ might be a multi-index.

### Product Manifold Types

```@autodocs
Modules = [Manopt]
Pages = ["manifolds/Product.jl"]
Order = [:type]
```

### Product Manifold Functions

```@autodocs
Modules = [Manopt]
Pages = ["manifolds/Product.jl"]
Order = [:function]
```

## Graph Manifold

The Graph manifold provides methods for two often interacting manifolds on
a given graph $\mathcal G = (\mathcal V,\mathcal E)$: A vertex graph manifold,
$\mathcal M^{\lvert \mathcal V\rvert}$ and an edge manifold $\mathcal N^{\lvert \mathcal E\rvert}$
for two [`Manifold`](@ref)s $\mathcal M$ and $\mathcal N$. For example $\mathcal N$
might be the tangent bundle of $\mathcal M$.

### Types

```@autodocs
Modules = [Manopt]
Pages = ["manifolds/Graph.jl"]
Order = [:type]
```

### Functions

```@autodocs
Modules = [Manopt]
Pages = ["manifolds/Graph.jl"]
Order = [:function]
```
