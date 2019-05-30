@testset "The SO(3)" begin
  import LinearAlgebra: det
  import Base: zeros
  import Random
  x = SOPoint([1. 0. 0.; 0. 1. 0.; 0. 0. 1.])
  xanti = SOPoint([1. 0. 0.; 0. -1. 0.; 0. 0. -1.])
  y = SOPoint([1 0 0; 0 cos(pi/5) -sin(pi/5); 0 sin(pi/5) cos(pi/5)])
  z = SOPoint([1 0 0; 0 cos(pi/3) -sin(pi/3); 0 sin(pi/3) cos(pi/3)])
  M = Rotations(3)
  ω = SOTVector(0.2*[0 1 2; -1 0 3; -2 -3 0])
  η = randomTVector(M,x)
  ξ = log(M, x ,y)
  μ = log(M, x, z)
  y2 = exp(M, x, ξ)
  z2 = exp(M, x, μ)
  x2 = addNoise(M,x,0.5)
  x3 = addNoise(M,x,500)
  w = randomMPoint(M)
  s = rand(Float64)
  Random.seed!(1)

  # Test unary operator
  # Test Dimension
  @test manifoldDimension(x)==3
  @test manifoldDimension(M)==3
  # Test Minus operator
  @test getValue( (-η) ) == getValue(-η)
  # Test Cauchy-Schwarz-Inequation
  @test dot(M,x,ω,η)^2 <= norm(M,x,ω)^2*norm(M,x,η)^2
  @test dot(M,x,ω,η) <= norm(M,x,ω)*norm(M,x,η)
  # Test dot
  @test dot(M,x,ω,η) ≈ dot(M,x,η,ω) atol = 10.0^(-16)
  @test dot(M,x,ω,η+ξ) ≈ dot(M,x,ω,η) + dot(M,x,ω,ξ) atol = 10.0^(-15)
  @test dot(M,x,ω,s*η) ≈ s*dot(M,x,ω,η) atol = 10.0^(-14)
  # Test norm
  @test norm(M,x,ω)^2 ≈ dot(M,x,ω,ω) atol = 10.0^(-14)
  @test norm(M,x,ω) ≈ sqrt(dot(M,x,ω,ω)) atol = 10.0^(-16)
  # Test distance
  @test distance(M,x,x) ≈ 0 atol = 10.0^(-16)
  @test distance(M,x,xanti) ≈ π atol=10.0^(-16) # antipodal ponts
  @test distance(M,x,z) ≈ norm(M,x,μ) atol=10.0^(-16)
  # Test Exponential and logarithm for both usual and antipodal points
  @test norm(getValue(log(M, x, y2)) - getValue(ξ)) ≈ 0 atol = 10.0^(-16)
  @test norm(getValue(log(M, x, z2)) - getValue(μ)) ≈ 0 atol = 10.0^(-16)
  @test norm(getValue(exp(M,x,log(M,x,y))) - getValue(y)) ≈ 0 atol = 10.0^(-15)
  @test norm(getValue(log(M,x,exp(M,x,ω))) - getValue(ω)) ≈ 0 atol = 10.0^(-15)
  @test_throws ErrorException log(M,x,xanti)
  # Test randomMPoint and randomTVector
  N = Rotations(1)
  @test norm(getValue(randomMPoint(N)) - ones(1,1)) ≈ 0 atol = 10.0^(-16)
  @test det(getValue(w)) ≈ 1 atol = 10.0^(-9)
  @test norm(transpose(getValue(w))*getValue(w) - one(getValue(w))) ≈ 0 atol = 10.0^(-14)
  @test norm(getValue(η) + transpose(getValue(η))) ≈ 0 atol = 10.0^(-16)
  @test getValue(randomTVector(Rotations(1), SOPoint(ones(1,1)))) == zeros(1,1)
  # Test addNoise
  @test det(getValue(x2)) ≈ 1 atol = 10.0^(-9)
  @test norm(transpose(getValue(x2))*getValue(x2) - one(getValue(x))) ≈ 0 atol=10.0^(-15)
  @test det(getValue(x3)) ≈ 1 atol = 10.0^(-9)
  @test norm(transpose(getValue(x3))*getValue(x3) - one(getValue(x))) ≈ 0 atol=10.0^(-12)
  #Test retraction
  @test norm(getValue(inverseRetractionQR(M,x,retractionQR(M,x,ω))) - getValue(ω)) ≈ 0 atol = 10.0^(-14)
  @test norm(getValue(inverseRetraction(M,x,retraction(M,x,ω))) - getValue(ω)) ≈ 0 atol = 10.0^(-14)
  @test norm(getValue(inverseRetractionPolar(M,x,retractionPolar(M,x,ω))) - getValue(ω)) ≈ 0 atol = 10.0^(-14)
  @test norm(transpose(getValue(retractionQR(M,x,ω))) * getValue(retractionQR(M,x,ω)) - one(getValue(x))) ≈ 0 atol = 10.0^(-15)
  @test norm(transpose(getValue(retraction(M,x,ω))) * getValue(retraction(M,x,ω)) - one(getValue(x))) ≈ 0 atol = 10.0^(-15)
  @test norm(transpose(getValue(retractionPolar(M,x,ω))) * getValue(retractionPolar(M,x,ω)) - one(getValue(x))) ≈ 0 atol = 10.0^(-15)
  #Test parallelTransport
  @test norm(getValue(parallelTransport(M,x,y,ξ)) - getValue(ξ)) ≈ 0 atol = 10.0^(-16)
  #Test zeroTVector
  @test norm(getValue(zeroTVector(M,x))) ≈ 0 atol = 10.0^(-16)
  #Test Lie Group capabilities
  @test distance(M, x⊗y, SOPoint( transpose(getValue(x))*getValue(y)) ) ≈ 0 atol = 10.0^(-16)
  #Test validateMPoint and validateTVector
  @test validateMPoint(M,w) == true
  @test validateTVector(M,x,η) == true
  ynot1 = SOPoint(0.5*[1 0 0; 0 cos(pi/5) -sin(pi/5); 0 sin(pi/5) cos(pi/5)])
  ynot2 = SOPoint([1 2 3; 0 2 1; 0 0 0.5])
  @test_throws ErrorException validateMPoint(M,ynot1)
  @test_throws ErrorException validateMPoint(M,ynot2)
  ωnot = SOTVector(0.2*[1 1 2; -1 1 3; -2 -3 1])
  @test_throws ErrorException validateTVector(M,x,ωnot)
end
