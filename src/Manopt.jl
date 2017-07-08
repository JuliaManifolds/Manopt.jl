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
  include("manifolds/Sphere.jl")
# algorithms
	include("algorithms/simpleAlgorithms.jl")
	include("algorithms/proximalMaps.jl")
# helpers
  include("helpers/imageHelpers.jl")
# data
  include("artificialDataFunctions.jl")
end
