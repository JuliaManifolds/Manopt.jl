"""
    ManifoldValuedImageProcessing.jl
  A package to perform Image processing methods on images and data whose values
  lie on a given manifold.

  See Readme.md for an exaustive list of features and examples/ for several
  examples that can just je `include`d.
"""
module ManifoldValuedImageProcessing
# Manifolds
  include("Manifold.jl") #base type
  include("MatrixManifold.jl")
  include("Sphere.jl")
  include("Circle.jl")
# algorithms
  include("CPPAlgorithms.jl")
# data
  include("artificialDataFunctions.jl")
end
