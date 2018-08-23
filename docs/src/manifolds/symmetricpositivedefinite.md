# [The symmetric positive definite $n\times n$ matrices](@id SymmmetricPositiveDefiniteManifold)
The manifold of symmetric positive definite matrices
$\mathcal P(3) = \bigl\{ A \in \mathbb R^{n\times n} \big|
A = A^{\mathrm{T}} \text{ and }
x^{\mathrm{T}}Ax > 0 \text{ for } 0\neq x \in\mathbb R^n \bigr\}$
posesses the following instances of the abstract types
[`Manifold`](@ref), [`MPoint`](@ref), and [`TVector`](@ref).
```@autodocs
Modules = [Manopt]
Pages = ["manifolds/SymmetricPositiveDefinite.jl"]
Order = [:type]
```
Note that saving the points on the manifold in this format is a little bit rendundant,
it' enough to save the upper triangular matrix. For ease of computations this
is – for now – adapted from Matlab.

# Specific Functions
```@autodocs
Modules = [Manopt]
Pages = ["manifolds/SymmetricPositiveDefinite.jl"]
Order = [:function]
```
