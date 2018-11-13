#using Test
#@testset "The Second Order Terms and their gradient" begin
  using Colors
  using Manopt
  M = Sphere(2)
  p1 = MPointE(SnPoint([0., 1., 0.])) # right
  p2 = MPointE(SnPoint([0., 0., 1.])) # North pole
  p3 = MPointE(SnPoint([1., 0., 0.]))
  pM = midPoint(M,p1,p3)
  gradSO = gradTV2(M,(p1,p2,p3))
  a = [getBase(p) for p in [p1,p2,p3,pM] ]
  renderAsymptote("test.asy", asyExportS2; points=[ a, ], tVectors = [[gradSO...],],
    colors=Dict( :points => [RGBA(.5,0,0,1),], :tvectors => [RGBA(0.,.7,.3,1),] ),
    dotSize=3.0)
#end
