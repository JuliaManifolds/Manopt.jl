using ManifoldValuedImageProcessing
using Base.Test
tests = ["testSn","testProximalMaps"]

@testset "ManifoldValuedImageProcessing Tests" begin
  for t in tests
    include("$(t).jl")
  end
end
