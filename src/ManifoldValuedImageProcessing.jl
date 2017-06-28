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
  include("MATRIXManifold.jl") #base type
  include("Sn.jl")
  include("S1.jl")
# algorithms
  include("CPPAlgorithms.jl")
# data
  include("artificialDataFunctions.jl")
end
