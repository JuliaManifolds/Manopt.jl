# Functions

There are several functions required within optimization, most prominently
[costFunctions](@ref CostFunctions) and [gradients](@ref GradientFunctions). This package includes
several cost functions and corresponding gradients, but also corresponding
[proximal maps](@ref proximalMapFunctions) for variational methods
manifold-valued data. Most of these functions require the evaluation of
[Differential](@ref DifferentialFunctions)s or their [adjoints](@ref adjointDifferentialFunctions)s.