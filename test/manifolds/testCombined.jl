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
@test typeofMPoint(ProdTVector{Array{TVector,1}}) == ProdPoint{Array{MPoint,1}}
@test typeofTVector(ProdPoint{Array{MPoint,1}}) == ProdTVector{Array{TVector,1}}
@test validateMPoint(M, randomMPoint(M) )
@test validateTVector(M,x, randomTVector(M,x) )

@test_throws DomainError validateMPoint(Product([M1]),x)
@test_throws DomainError validateTVector(M,x,ProdTVector([ξ1]))

@test typicalDistance(M) ≈ sqrt( 2* (typicalDistance(M1).^2 + typicalDistance(M2).^2) ) 

@test zeroTVector(M,x) == ProdTVector([zeroTVector(M1,x1), zeroTVector(M2,x2)]) 

@test "$M" == "The Product Manifold of [ $(M1), $(M2) ]"
@test "$x" == "Prod[ $(x1), $(x2) ]"
@test "$ξ" == "ProdT[ $(ξ1), $(ξ2) ]"
#
# Power
M = Power(M1,2)
@test M.powerSize == (2,)
@test Power(M1,(3,2)).powerSize == (3,2)

x = PowPoint([x1,x1])
y = PowPoint([y1,y1])
ξ = PowTVector([ξ1,ξ1])
η = PowTVector([η1,η1])
#
@test size(x) == size(x.value)
@test distance(M, copy(x), x) == 0
@test norm(M,x, ξ - copy(ξ)) == 0
@test ndims(x) == ndims(x.value)
@test getValue(x) == [x1,x1]
@test getValue(ξ) == [ξ1,ξ1]
#
x[2] = RnPoint([1., 2., 0.])
@test getValue(x[2]) == [1., 2., 0.]
ξ[2] = RnTVector([1., 2., 0.])
@test getValue(ξ[2]) == [1., 2., 0.]
@test PowPoint([x1 x1; x1 x1])[1,2] == x1
x[2] = x1
ξ[2] = ξ1
xG = PowPoint([x1 x1;y1 y1])
@test xG[1:2,2].value == [x1,y1]
@test ndims(xG) == 2
ξG = PowTVector([ξ1 ξ1;η1 η1])
@test ξG[1:2,2].value == [ξ1,η1]
@test ndims(ξG) == 2
@test size(repeat(x, outer=[2,1])) == (4,1)
@test size(repeat(x, 2,1)) == (4,1)
@test size(repeat(x, inner=[1,2])) == (2,2)
@test size(repeat(ξ, outer=[2,1])) == (4,1)
@test size(repeat(ξ, 2,1)) == (4,1)
@test size(repeat(ξ, inner=[1,2])) == (2,2)
#
@test getValue.(getValue( adjointJacobiField(M, x, y, 0.5, ξ))) == getValue.(
        adjointJacobiField.(Ref(M1), [x1,x1], [y1,y1], 0.5, [ξ1,ξ1]) )
@test getValue.(getValue( jacobiField(M, x, y, 0.5, ξ))) == 
        getValue.(jacobiField.(Ref(M1), [x1,x1], [y1,y1], 0.5, [ξ1,ξ1]) )
#
@test distance(M,x,y) ≈ sqrt(2)*distance(M1,x1,y1)
@test dot(M,x,ξ,η) ≈ 2*dot(M1,x1,ξ1,η1)
#
@test getValue(exp(M, x, ξ)) == [exp(M1, x1, ξ1), exp(M1, x1, ξ1)]
@test getValue(log(M, x, y)) == [log(M1, x1, y1), log(M1, x1, y1)]
#
@test manifoldDimension(x) == 2*manifoldDimension(x1)
@test manifoldDimension(M) == 2*manifoldDimension(M1)
@test norm(M,x,ξ) ≈ sqrt(2)*norm(M1,x1,ξ1)
@test ndims(x) == 1
@test ndims(ξ) == 1
@test getValue( parallelTransport(M,x,y,ξ) ) == [parallelTransport(M1, x1, y1, ξ1),parallelTransport(M1, x1, y1, ξ1)]
@test validateMPoint(M,randomMPoint(M))
@test validateTVector(M,x,randomTVector(M,x))
@test validateTVector(M,x,randomTVector(M,x,:Gaussian))

@test typeofMPoint(ξ) == PowPoint{RnPoint{Float64},1}
@test typeofTVector(x) == PowTVector{RnTVector{Float64},1}
@test typeofTVector(PowPoint{RnPoint{Float64},1}) == PowTVector{RnTVector{Float64},1}
@test typeofMPoint(PowTVector{RnTVector{Float64},1}) == PowPoint{RnPoint{Float64},1}

@test_throws DomainError validateMPoint(Power(M,3),x)
@test_throws DomainError validateTVector(M,x,PowTVector([ξ1]))

@test typicalDistance(M) ≈ sqrt( 2 ) * typicalDistance(M1)

@test zeroTVector(M,x) == PowTVector([zeroTVector(M1,x1), zeroTVector(M1,x1)]) 

a0 = RnTVector([0.,0.,0.]); a1 = RnTVector([1.,0.,0.]);
a2 = RnTVector([0.,1.,0.]); a3 = RnTVector([0.,0.,1.])
ΞT = [PowTVector([a1,a0]), PowTVector([a2,a0]), PowTVector([a3,a0]),
    PowTVector([a0,a1]), PowTVector([a0,a2]), PowTVector([a0,a3]) ]
κT = zeros(6)
Ξ,κ = tangentONB(M,x,y)

@test sum( abs.(κ-κT)) == 0
@test sum( norm.(Ref(M), Ref(x),  ΞT .- Ξ ) ) == 0
# show
@test "$x" == "Pow[Rn([1.0, 2.0, 0.0]), Rn([1.0, 2.0, 0.0])]"
@test "$ξ" == "PowT[RnT([1.0, 1.0, 0.0]), RnT([1.0, 1.0, 0.0])]"
@test "$M" == "The Power Manifold of The 3-dimensional Euclidean space of size (2,)."
#
# Tangent Bundle
M = TangentBundle(M1)
@test getBase(M) == M1
x = TBPoint(x1,ξ1)
y = TBPoint(y1,η1)
xT = TBPoint( (x1,ξ1) )
@test x == xT
@test getValue(x) == (x1,ξ1)
xE = MPointE(x)
@test getBase(x) == x1
@test getTangent(x) == ξ1
@test getBase(xE) == MPointE(x1)
@test getTangent(xE) == TVectorE(ξ1,x1)

ξ = TBTVector(ξ1,η1)
ξT = TBTVector( (ξ1,η1) )
@test ξ == ξT
@test getValue(ξ) == (ξ1,η1)
ξE = TVectorE(ξ,x)
@test getBase(ξ) == ξ1
@test getTangent(ξ) == η1
@test getBase(ξE) == TVectorE(ξ1,x1)
@test getTangent(ξE) == TVectorE(η1,x1)

@test 2*ξ == TBTVector(2*ξ1, 2*η1)
@test ξ+ξ == TBTVector(ξ1+ξ1,η1+η1)
@test ξ-ξ == TBTVector(ξ1-ξ1,η1-η1)
@test -ξ == TBTVector(-ξ1,-η1)
@test +ξ == TBTVector(ξ1,η1)

@test distance(M,x,y) ≈ sqrt(3)
@test norm(M,x,ξ) ≈ sqrt(dot(M,x,ξ,ξ))
@test norm(M,x,ξ) ≈ sqrt(norm(M1,x1,ξ1)^2 + norm(M1,x1,η1))
@test exp(M,x,ξ) == TBPoint(exp(M1,x1,ξ1), ξ1+η1)
@test log(M,x,y) ==  TBTVector(log(M1,x1,y1),ξ1-η1)
@test manifoldDimension(M) == 2*manifoldDimension(M1)
@test manifoldDimension(x) == 2*manifoldDimension(x1)
@test parallelTransport(M,x,y,ξ) == ξ
@test validateMPoint(M,randomMPoint(M))
@test validateTVector(M,x,randomTVector(M,x))

@test typeofMPoint(ξ) == TBPoint{RnPoint{Float64},RnTVector{Float64}}
@test typeofTVector(x) == TBTVector{RnTVector{Float64}}

Ξ, κ = tangentONB(M,x,y)
@test κ == zeros(6)
@test Ξ == [TBTVector(a1,a0), TBTVector(a2,a0), TBTVector(a3,a0), TBTVector(a0,a1), TBTVector(a0,a2), TBTVector(a0,a3)]

@test typicalDistance(M) == sqrt( typicalDistance(M1)^2 + manifoldDimension(M1))
@test zeroTVector(M,x) == TBTVector(a0,a0)

@test "$M" == "The Tangent bundle of <$(M.manifold)>"
@test "$x" == "TB(Rn([1.0, 2.0, 0.0]), RnT([1.0, 1.0, 0.0]))"
@test "$ξ" == "TBT(RnT([1.0, 1.0, 0.0]), RnT([1.0, 0.0, 0.0]))"
end