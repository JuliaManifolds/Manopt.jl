```@meta
CurrentModule = Manopt
```
# [Special Manifolds build upon one or more Riemannian manifolds](@id CombinedManifolds)

## Tangent bundle
The tangent bundle $T\mathcal M$ of a manifold $\mathcal M$ consists of all tuples
$(x,\xi) \in T\mathcal M$, where $\xi\inT_x\mathcal M$, $x\in \mathcal M$, where
the metric is inherited componentwise and for the exponential and logarithmic map,
the second component requires a `parallelTransport`.

### Types
```@autodocs
Modules = [Manopt]
Pages = ["manifolds/TangentBundle.jl"]
Order = [:type]
```
### Functions
```@autodocs
Modules = [Manopt]
Pages = ["manifolds/TangentBundle.jl"]
Order = [:function]
```

## Power manifold
The product manifold $\mathcal M^n$, where $n\in\mathbb N^k$ represents
arrays that are manifold-valued, for example, if $n$ is a number ($k=1$)
we obtain a manifold-valued signal $f\in\mathcal M^n$.
Many operations, for example `jacobiFIelds` are performed elementwise,
while the basic operations, like `distance` is of course not.
### Types
```@autodocs
Modules = [Manopt]
Pages = ["manifolds/Power.jl"]
Order = [:type]
```
### Functions
```@autodocs
Modules = [Manopt]
Pages = ["manifolds/Power.jl"]
Order = [:function]
```

## Product manifold
A little more general is the product manifold, where
$\mathcal M = \mathcal N_1\times\cdots\times\mathcal N_n$, $n\in\mathbb N^k$
is a product of manifolds, i.e. for a value $f\in\mathcal M$ we have that
$f_i\in\mathcal N_i$, where $i$ might be a multi-index.
Many operations, for example `jacobiFIelds` are performed elementwise,
while the basic operations, like `distance` is of course not.
### Types
```@autodocs
Modules = [Manopt]
Pages = ["manifolds/Product.jl"]
Order = [:type]
```
### Functions
```@autodocs
Modules = [Manopt]
Pages = ["manifolds/Product.jl"]
Order = [:function]
```

## Graph Manifold
The Graph manifold provides methods for two often interacting manifolds on
a given graph $\mathcal G = (\mathcal V,\mathcal E)$: A vertex graph manifold,
$\mathcal M^{\lvert \mathcal V\rvert}$ and an edge manifold $\mathcal N^{\lvert \mathcal E\rvert}$
for two [`Manifold`]s $\mathcal M$ and $\mathcal N$. For example $\mathcal N$
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
