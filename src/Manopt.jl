"""
    Manopt.jl
A package to perform Optimization methods on manifold in Julia uncluding
high dimensional power manifolds to tackle manifold-valued image processing.

See `Readme.md` more details, and `examples/` for examples that can just be
`include`d.
"""
module Manopt
  using SimpleTraits
  using Markdown
# Manifolds
  include("manifolds/Manifold.jl") #base type
  # Traits (properties/decorators)
  include("manifolds/traits/EmbeddedManifold.jl")
  include("manifolds/traits/LieGroup.jl")
  include("manifolds/traits/MatrixManifold.jl")
  # specific manifolds
  include("manifolds/Circle.jl")
  include("manifolds/Euclidean.jl")
  include("manifolds/Graph.jl")
  include("manifolds/Hyperbolic.jl")
	include("manifolds/SymmetricPositiveDefinite.jl")
  include("manifolds/Power.jl")
  include("manifolds/Product.jl")
  include("manifolds/Sphere.jl")
  # Functions
  include("functions/adjointDifferentials.jl")
  include("functions/costFunctions.jl")
  include("functions/differentials.jl")
  include("functions/gradients.jl")
  include("functions/jacobiFields.jl")
  include("functions/proximalMaps.jl")
  # ...corresponding plans consisting of problems and options
  include("plans/plan.jl")
  # ...solvers
  include("solvers/conjugateGradientDescent.jl")
  include("solvers/cyclicProximalPoint.jl")
  include("solvers/DouglasRachford.jl")
  include("solvers/steepestDescent.jl")
  include("solvers/subGradientMethod.jl")
  include("solvers/trustRegion.jl")
  # algorithms
  include("algorithms/basicAlgorithms.jl")
  include("algorithms/lineSearch.jl")
  # Plots
  include("plots/SpherePlots.jl")
  # helpers
  include("helpers/debugFunctions.jl")
  include("helpers/errorMeasures.jl")
  include("helpers/imageHelpers.jl")
  include("helpers/stepSizeFunctions.jl")
  # Exports
  include("helpers/exports/Asymptote.jl")
  # data
  include("data/artificialDataFunctions.jl")
  include("data/S2Lemniscate.jl")
  include("data/signals.jl")
end
