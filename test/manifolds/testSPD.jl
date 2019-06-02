@testset "Test the 2-by-2 symmetric positive definite matrices" begin
using Random: seed!
# explicitly compute an easy exp
M = SymmetricPositiveDefinite(2)
x = SPDPoint([1.0 0.0; 0.0 1.0])
ξ = SPDTVector([1.0 0.0; 0.0 0.0])
η = SPDTVector([1.0 0.0; 0.0 1.0])
y = exp(M,x,ξ)
@test norm( getValue(y) - [ℯ 0;0 1.0]) ≈ 0 atol=10.0^(-16)
# check that with base the base must mach.
z = SPDPoint([1.0 0.0; 0.0 0.1])
ξE = TVectorE(ξ,x);
@test_throws DomainError exp(M,z,ξE)
# check that log is the inverse of exp and stores the base point correctly
# if that's activated
xT = MPointE(x);
ξT2 = log(M,x,y)
@test norm( getValue(ξ) - getValue(ξT2) ) ≈ 0 atol=10.0^(-16)
@test distance(M, getBasePoint(ξE), x) ≈ 0 atol=10.0^(-16)
# test norm
@test norm(M,x,ξ) ≈ distance(M,x,y) atol=10.0^(-16)
# test parallel transport
@test norm(getValue( parallelTransport(M,y,x,log(M,y,x)) ) + getValue(ξ) ) ≈ 0 atol=3*10.0^(-16)
@test parallelTransport(M,x,x,ξ) == ξ
# random
seed!(42)
@test validateMPoint(M,randomMPoint(M))
@test validateTVector(M,x, randomTVector(M,x))
@test validateTVector(M,x, randomTVector(M,x, :Gaussian))
@test validateTVector(M,x, randomTVector(M,x, :Rician))
#
# Test tangent ONB
#
n = 4
M2 = SymmetricPositiveDefinite(n)
x2 = SPDPoint(one(zeros(n,n)))
ξ2 = SPDTVector(getValue(x2))
y2 = exp(M2,x2,ξ2)
Ξ,κ = tangentONB(M2,x2,ξ2)
# test orthogonality
@test all([ dot(M2,x2,Ξ[i],Ξ[j]) for i=1:length(Ξ) for j=i+1:length(Ξ) ] .== 0)
# test normality
@test all( [1-dot(M2,x2,Ξ[i],Ξ[i]) for i=1:length(Ξ)] .< 3*10^(-16) )
Ξ2,κ2 = tangentONB(M2,x2,y2)
@test sum( norm.(Ref(M2),Ref(x2), Ξ .- Ξ2) ) ≈ 0
@test sum(κ - κ2) ≈ 0
#
# test types
@test typeofTVector(x) == SPDTVector{Float64}
@test typeofTVector(SPDPoint{Float64}) == SPDTVector{Float64}
@test typeofMPoint(ξ) == SPDPoint{Float64}
@test typeofMPoint(SPDTVector{Float64}) == SPDPoint{Float64}
#
@test typicalDistance(M) ≈ sqrt(3)
#
@test_throws DomainError validateMPoint(M2,x)
@test_throws DomainError validateMPoint(M, SPDPoint([0. 1.;-1. 0.])) 
@test_throws DomainError validateMPoint(M, SPDPoint([0. 0.;0. 0.]))
@test_throws DomainError validateTVector(M,x,ξ2)
@test_throws DomainError validateTVector(M,x,SPDTVector([0. 1.; -2. 0.]) )
#
@test norm(getValue(zeroTVector(M,x))) == 0
@test validateTVector(M,x,zeroTVector(M,x))
# show
@test "$M" == "The Manifold of 2-by-2 symmetric positive definite matrices"
@test "$x" == "SPD($(getValue(x)))"
@test "$ξ" == "SPDT($(getValue(ξ)))"
# Test Matrix trait
@test getValue(x+y) == getValue(x)+getValue(y)
@test getValue(x-y) == getValue(x)-getValue(y)
@test getValue(x*y) == getValue(x)*getValue(y)
@test getValue(ξ+η) == getValue(ξ)+getValue(η) 
@test getValue(ξ-η) == getValue(ξ)-getValue(η) 
@test getValue(ξ*η) == getValue(ξ)*getValue(η)
@test transpose(x) == SPDPoint(Matrix(transpose(getValue(x))))
@test transpose(ξ) == SPDTVector(Matrix(transpose(getValue(ξ))))
end