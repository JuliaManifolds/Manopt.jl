@testset "Combined Manifolds" begin
using Random: seed!

M1 = Euclidean(3)
x1 = RnPoint([1.,2.,0.])
y1 = RnPoint([2.,3.,0.])
ξ1 = log(M1,x1,y1)
η1 = RnTVector([1.,0.,0.])

M2 = Sphere(2)
x2 = SnPoint([1.,0.,0.])
y2 = SnPoint([0.,0.,1.])
ξ2 = log(M2,x2,y2)
η2 = SnTVector(1/sqrt(2)*[0., 1.,1.])

#
# Prod
M = Product([M1, M2])
x = ProdPoint([x1,x2])
y = ProdPoint([y1,y2])
ξ = ProdTVector([ξ1,ξ2])
η = ProdTVector([η1, η2])

@test getValue(x) == [x1,x2]
@test getValue(ξ) == [ξ1, ξ2]

@test distance(M,x,y) == sqrt(distance(M1,x1,y1)^2 + distance(M2,x2,y2).^2)
@test dot(M,x,ξ,η) == dot(M1,x1,ξ1,η1) + dot(M2,x2,ξ2,η2)
@test distance( M, exp(M,x,ξ), ProdPoint([exp(M1,x1,ξ1),exp(M2,x2,ξ2)]) ) == 0
@test norm( M, x, log(M,x,y) - ProdTVector([log(M1,x1,y1), log(M2,x2,y2)]) ) ≈ 0 
@test manifoldDimension(x) == 5
@test manifoldDimension(M) == 5
@test norm(M,y, parallelTransport(M,x,y,ξ) - ProdTVector([
    parallelTransport(M1,x1,y1,ξ1), parallelTransport(M2,x2,y2,ξ2) ]) ) ≈ 0 

@test typeofMPoint(ξ) == ProdPoint{Array{MPoint,1}}
@test typeofTVector(x) == ProdTVector{Array{TVector,1}}

@test validateMPoint(M, randomMPoint(M) )
@test validateTVector(M,x, randomTVector(M,x) )

@test typicalDistance(M) ≈ sqrt( 2* (typicalDistance(M1).^2 + typicalDistance(M2).^2) ) 

@test zeroTVector(M,x) == ProdTVector([zeroTVector(M1,x1), zeroTVector(M2,x2)]) 

@test "$M" == "The Product Manifold of [ $(M1), $(M2) ]"
@test "$x" == "Prod[ $(x1), $(x2) ]"
@test "$ξ" == "ProdT[ $(ξ1), $(ξ2) ]"
#
# Power

#
# Tangent Bundle
end