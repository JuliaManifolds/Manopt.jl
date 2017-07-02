"""
    ManifoldValuedImageProcessing.jl
  A package to perform Image processing methods on images and data whose values
  lie on a given manifold.

  See Readme.md for an exaustive list of features and examples/ for several
  examples that can just je `include`d.
"""
module ManifoldValuedImageProcessing
# Manifolds
  include("manifolds/Manifold.jl") #base type
  # matrix manifold – common functions
  include("manifolds/MatrixManifold.jl")
  # specific manifolds
  include("manifolds/Circle.jl")
  include("manifolds/Euclidean.jl")
  include("manifolds/Sphere.jl")
# algorithms
  include("CPPAlgorithms.jl")
# helpers
  include("helpers/imageHelpers.jl")
# data
  include("artificialDataFunctions.jl")
end
