using ManifoldValuedImageProcessing
using Base.Test
tests = ["testSn","testProximalMaps"]

for t in tests
  include("$(t).jl")
end
