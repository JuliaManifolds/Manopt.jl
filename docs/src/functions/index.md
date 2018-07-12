# Functions
There are several functions required within optimization, most prominently
`costFunctions` and [gradients](@ref GradientFunctions). This package includes
several cost functions and corresponding gradients, but also corresponding
[proximal maps](@ref proximalMapFunctions) for variational methods
manifold-valued data. Most of these functions require the evaluation of
`Differential`s or their `AdjointDifferential`s as well as `JacobiFields` (for
symmetric manifolds).

These are collected in the Functions Library of `Manopt.jl`.
