using Manopt
using Test
@testset "Manopt.jl Tests" begin
    include("testSn.jl")
    include("testSO.jl")
    include("testSPD.jl")
    include("testGradients.jl")
    include("testGraphConstruction.jl")
    include("testGradDesc.jl")
    include("testProximalMaps.jl")
end
