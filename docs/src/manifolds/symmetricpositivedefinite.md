# [The Symmetric Positive Definite $n\times n$ Matrices](@id SymmetricPositiveDefiniteManifold)

The manifold of symmetric positive definite matrices
$\mathcal P(3) = \bigl\{ A \in \mathbb R^{n\times n}\ \big|\ 
A = A^{\mathrm{T}} \text{ and }
x^{\mathrm{T}}Ax > 0 \text{ for } 0\neq x \in\mathbb R^n \bigr\}$
posesses the following instances of the abstract types
[`Manifold`](@ref), [`MPoint`](@ref), and [`TVector`](@ref).

```@autodocs
Modules = [Manopt]
Pages = ["manifolds/SymmetricPositiveDefinite.jl"]
Order = [:type]
```

While there are two Riemannian metrics available, this one focuses on the
affine metric. The Log-Euclidean Metric needs at least a new tangent vector
type inheriting from `<: TVector`.

Note that saving the points on the manifold in this format is a little bit rendundant,
it' enough to save the upper triangular matrix. For ease of computations this
is – for now – adapted from Matlab.

# Functions

```@autodocs
Modules = [Manopt]
Pages = ["manifolds/SymmetricPositiveDefinite.jl"]
Order = [:function]
```
