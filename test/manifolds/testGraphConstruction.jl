@testset "image graph construction" begin
  # these points are null, but we only need the size for the indices tests
  img = Array{SnPoint}(undef,2,3)
  K = constructImageGraph(img,"firstOrderDifference")
  @test K==[(1,2),(3,4),(5,6),(1,3),(2,4),(3,5),(4,6)]
  K2 = constructImageGraph(img,"secondOrderDifference")
  @test K2 == [(1,3,5),(2,4,6)]
  @test_throws ErrorException constructImageGraph(img,"K") # any other yields an error
end
