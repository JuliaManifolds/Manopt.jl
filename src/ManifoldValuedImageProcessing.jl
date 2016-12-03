module ManifoldValuedImageProcessing
# Manifolds
  include("Manifold.jl") #base type
  include("Sn.jl")
  include("S1.jl")
# algorithms
  include("CPPAlgorithms.jl")
end
