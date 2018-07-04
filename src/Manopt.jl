"""
    Manopt.jl
  A package to perform Optimization methods on manifold in Julia uncluding
	high dimensional power manifolds to tacke manifold-valued image processing.

  See Readme.md for an exaustive list of features and examples/ for several
  examples that can just be `include`d.
"""
module Manopt
# Manifolds
  include("manifolds/Manifold.jl") #base type
  # matrix manifold – common functions
  include("manifolds/MatrixManifold.jl")
  # specific manifolds
  include("manifolds/Circle.jl")
	include("manifolds/Euclidean.jl")
	include("manifolds/SymmetricPositiveDefinite.jl")
  include("manifolds/PowerManifold.jl")
  include("manifolds/ProductManifold.jl")
  include("manifolds/Sphere.jl")
  # ...corresponding plans consisting of problems and options
  include("plans/problem.jl")
  include("plans/options.jl")
  # ...solvers
  include("solvers/steepestDescent.jl")
  # algorithms
  include("algorithms/basicAlgorithms.jl")
  include("algorithms/lineSearch.jl")
  include("algorithms/proximalMaps.jl")
  # Plots
  include("plots/SpherePlots.jl")
  # helpers
  include("helpers/imageHelpers.jl")
  include("helpers/debugFunctions.jl")
  # data
  include("data/artificialDataFunctions.jl")
end
