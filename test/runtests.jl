using Manopt
using Test
tests = ["testSn","testSPD",
          "testGradients",
          "testGraphConstruction",
          "testGradDesc",
          "testProximalMaps"
          ]
@testset "Manopt.jl Tests" begin
  for t in tests
    include("$(t).jl")
  end
end
