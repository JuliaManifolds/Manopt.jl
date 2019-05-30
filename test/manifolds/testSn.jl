@testset "The 2-Sphere Sphere(2)" begin
x = SnPoint([1.,0.,0.])
y = SnPoint([0.,1.,0.])
z = SnPoint([0.,0.,1.])
M = Sphere(2)
ξ = log(M,x,y)
y2 = exp(M,x,ξ)
ν = log(M,x,z)
# Test unary operator
@test getValue( (-ν) ) == getValue(-ν)
@test opposite(M,x) == SnPoint(-getValue(x))
@test norm( getValue(y) - getValue(y2) ) ≈ 0 atol=10.0^(-16)
@test norm(M,x,ξ) ≈ norm( getValue(ξ) ) atol=10.0^(-16)
@test distance(M,x,y) ≈ norm(M,x,ξ) atol=10.0^(-16)
@test dot(M,x,ξ,ν) ≈ 0 atol = 10.0^(-16)
@test norm( getValue( mean(M,[x,y,z]) ) - 1/sqrt(3)*ones(3)) ≈ 0 atol=10.0^(-7)
@test norm( getValue( mean(M,[x,y]) ) - [1/sqrt(2),1/sqrt(2),0] ) ≈ 0 atol=10.0^(-15)
@test manifoldDimension(M) == manifoldDimension(x)
# Test randoms
@test validateMPoint(M,randomMPoint(M))
@test validateTVector(M,x,randomTVector(M,x))
# Test extended
xT = MPointE(x); yT = MPointE(y); zT = MPointE(z);
@test_throws DomainError dot(M,xT,log(M,xT,zT),log(M,yT,zT) )
@test dot(M,x,log(M,x,z),log(M,x,y) ) ≈ 0 atol=10.0^(-15)
# Tst ONB
@test tangentONB(M,x,y) == ( [SnTVector([0.,1.,0.]), SnTVector([0., 0., 1.])], [0.,1.] )
#
@test_throws ErrorException validateMPoint(M,SnPoint([1.,0.]))
@test_throws ErrorException validateMPoint(M,SnPoint([2.,0.,0.])) 
@test_throws ErrorException validateTVector(M,x,SnTVector([1., 0.]))
@test_throws ErrorException validateTVector(M,x,log(M,z,x))

#check that PT(q->p, log_qp) = -log_pq (unitary minus already checked)
@test parallelTransport(M,y,x,log(M,y,x)) == -ξ
@test typeofTVector(typeof(x)) == SnTVector{Float64}
@test typeofMPoint(typeof(ξ)) == SnPoint{Float64}
@test typeofTVector(x) == SnTVector{Float64}
@test typeofMPoint(ξ) == SnPoint{Float64}
@test typicalDistance(M) == π
@test "$M" == "The 2-Sphere"
@test "$x" == "Sn([1.0, 0.0, 0.0])"
@test "$ξ" == "SnT([0.0, 1.5708, 0.0])"
#
#
#
@test_throws ErrorException x⊗y
@test_throws ErrorException x⊗y

end
