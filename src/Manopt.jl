"""
`Manopt.jl` – Optimization on Manifolds in Julia.
"""
module Manopt
  using SimpleTraits
  using Markdown
# Manifolds
  include("manifolds/Manifold.jl") #base type
  # errors for false combinations of types or nonimplemented cases
  include("manifolds/defaults/manifoldFallbacks.jl")
  # Extended Vector decorations
  include("manifolds/defaults/extendedData.jl")

  # Traits (properties/decorators)
  include("manifolds/traits/EmbeddedManifold.jl")
  include("manifolds/traits/LieGroup.jl")
  include("manifolds/traits/MatrixManifold.jl")
  # specific manifolds
  include("manifolds/Circle.jl")
  include("manifolds/Euclidean.jl")
  include("manifolds/Graph.jl")
  include("manifolds/Grassmannian.jl")
  include("manifolds/Hyperbolic.jl")
  include("manifolds/Rotations.jl")
  include("manifolds/Sphere.jl")
  include("manifolds/Stiefel.jl")
  include("manifolds/SymmetricPositiveDefinite.jl")
  include("manifolds/Symmetric.jl")
  include("manifolds/TangentBundle.jl")
  # meta
  include("manifolds/Power.jl")
  include("manifolds/Product.jl")
  # ...corresponding plans consisting of problems and options
  include("plans/plan.jl")
  # Functions
  include("functions/adjointDifferentials.jl")
  include("functions/costFunctions.jl")
  include("functions/differentials.jl")
  include("functions/gradients.jl")
  include("functions/jacobiFields.jl")
  include("functions/proximalMaps.jl")
  # ...solvers (1) general framework
  include("solvers/solver.jl")
  # ...solvers (2) specific solvers
  include("solvers/cyclicProximalPoint.jl")
  include("solvers/DouglasRachford.jl")
  include("solvers/steepestDescent.jl")
  include("solvers/subGradientMethod.jl")
  # extended metasolvers
  include("solvers/debugSolver.jl")
  include("solvers/recordSolver.jl")
  # algorithms
  include("algorithms/basicAlgorithms.jl")
  # Plots
  include("plots/SpherePlots.jl")
  # helpers
  include("helpers/errorMeasures.jl")
  include("helpers/imageHelpers.jl")
  # Exports
  include("helpers/exports/Asymptote.jl")
  # data
  include("data/artificialDataFunctions.jl")
end
