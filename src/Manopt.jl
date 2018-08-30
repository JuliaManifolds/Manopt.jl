"""
    Manopt.jl
A package to perform Optimization methods on manifold in Julia uncluding
high dimensional power manifolds to tacke manifold-valued image processing.

See `Readme.md` for an exaustive list of features and `examples/` for several
examples that can just be `include`d.
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
  include("plans/problem.jl")
  include("plans/options.jl")
  # ...solvers
  include("solvers/conjugateGradientDescent.jl")
  include("solvers/cyclicProximalPoint.jl")
  include("solvers/DouglasRachford.jl")
  include("solvers/steepestDescent.jl")
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
  # data
  include("data/artificialDataFunctions.jl")
  include("data/signals.jl")
end
