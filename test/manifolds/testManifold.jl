@testset "General Manifold functions and fallbacks" begin
M = Euclidean(3)
x = RnPoint([1.,0.,0.])
y = RnPoint([0.,1.,0.])
ξ = RnTVector([1.,0.,1.])
η = RnTVector([0.,1.,1.])
@test copy(x) == x
@test copy(ξ) == ξ
@test getValue(2*ξ) == 2*getValue(ξ)
@test getValue(ξ*2) == getValue(ξ)*2
@test getValue(2/RnTVector([1.,1.])) == 2 ./ [1.,1.]
@test getValue(ξ/2) == getValue(ξ)/2
@test getValue(ξ+η) == getValue(ξ) + getValue(η)
@test getValue(ξ-η) == getValue(ξ) - getValue(η)
@test getValue(+η) == getValue(η)
@test getValue(-η) == -getValue(η)
@test x==x
@test ξ==ξ
@test geodesic(M,x,y,3) == RnPoint.([[1.,0.,0.],[.5,.5,0.],[0.,1.,0.]])
@test geodesic(M,x,y,[0.,.5,1.]) == RnPoint.([[1.,0.,0.],[.5,.5,0.],[0.,1.,0.]])
@test reflection(M,x,y) == RnPoint(2*getValue(x)-getValue(y))
#
# Test Fallbacks – check that errors are thrown if the manifold does not fit
struct TestManifold <: Manifold end
struct TestMPoint <: MPoint end
struct TestTVector <: TVector end
M = TestManifold()
x = TestMPoint()
y = TestMPoint()
ξ = TestTVector()
η = TestTVector() 
@test_throws DomainError distance(M, x, y)
@test_throws DomainError dot(M, x, ξ, η)
@test_throws ErrorException dot(Euclidean(3), ξ, η)
@test_throws DomainError exp(M, x, ξ)
@test_throws DomainError log(M, x, y)
@test_throws DomainError parallelTransport(M, x, y, ξ)
@test_throws DomainError tangentONB(M, x, ξ)
@test_throws DomainError tangentONB(M, x, y)
@test_logs (:warn,"""No valitadion for a TestMPoint on TestManifold available. Continuing without
  validation. To turn this warning off, either deactivate the validate flag
  in (one of) your extended MPoints or implement a corresponding validation.""") validateMPoint(M,x)
@test_logs (:warn,"""No valitadion for a TestMPoint and a TestTVector on TestManifold available.
  Continuing without validation. To turn this warning off, either deactivate
  the validate flag in (one of) your extended TVectors or MPoints or
  implement a corresponding validation""") validateTVector(M,x,ξ)
@test_throws DomainError zeroTVector(M,x)
#
#
@test_throws DomainError manifoldDimension(M)
@test_throws DomainError manifoldDimension(x)
@test_throws DomainError getValue(x)
@test_throws DomainError getValue(ξ)
#
@test_throws ErrorException typeofMPoint(ξ)
@test_throws ErrorException typeofTVector(x)
@test_throws ErrorException typicalDistance(M)
end