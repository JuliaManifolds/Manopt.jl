@testset "The SO(3)" begin
  import LinearAlgebra: det
  import Base: zeros
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
  #@test_throws ErrorException dot(M,y,ξ,μ)
  # Test norm
  @test norm(M,x,ω)^2 ≈ dot(M,x,ω,ω) atol = 10.0^(-14)
  @test norm(M,x,ω) ≈ sqrt(dot(M,x,ω,ω)) atol = 10.0^(-16)
  # Test distance
  @test distance(M,x,x) ≈ 0 atol = 10.0^(-16)
  @test distance(M,x,xanti) ≈ π atol=10.0^(-16) # antipodal ponts
  @test distance(M,x,z) ≈ norm(M,x,μ) atol=10.0^(-16)
  # Test Exponential and logarithm for both usual and antipodal points
  @test getValue(log(M, x, y2)) ≈ getValue(ξ) atol = 10.0^(-16)
  @test getValue(log(M, x, z2)) ≈ getValue(μ) atol = 10.0^(-16)
  @test getValue(exp(M,x,log(M,x,y))) - getValue(y) ≈ zero(getValue(y)) atol = 10.0^(-15)
  @test getValue(log(M,x,exp(M,x,ω))) - getValue(ω) ≈ zero(getValue(y)) atol = 10.0^(-15)
  # Test randomMPoint and randomTVector
  @test det(getValue(w)) ≈ 1 atol = 10.0^(-9)
  @test transpose(getValue(w))*getValue(w) - one(getValue(w)) ≈ zero(getValue(w)) atol = 10.0^(-14)
  @test getValue(η) + transpose(getValue(η)) ≈ zero(getValue(η)) atol = 10.0^(-16)
  # Test addNoise
  @test det(getValue(x2)) ≈ 1 atol = 10.0^(-9)
  @test transpose(getValue(x2))*getValue(x2) ≈ one(getValue(x)) atol=10.0^(-15)
  @test det(getValue(x3)) ≈ 1 atol = 10.0^(-9)
  @test transpose(getValue(x3))*getValue(x3) - one(getValue(x)) ≈ zero(getValue(x)) atol=10.0^(-12)
  #Test retraction
  @test getValue(inverseRetractionQR(M,x,retractionQR(M,x,ω))) - getValue(ω) ≈ zero(getValue(ω)) atol = 10.0^(-14)
  @test getValue(inverseRetractionPolar(M,x,retractionPolar(M,x,ω))) - getValue(ω) ≈ zero(getValue(ω)) atol = 10.0^(-14)
  @test transpose(getValue(retractionQR(M,x,ω))) * getValue(retractionQR(M,x,ω)) - one(getValue(x)) ≈ zero(getValue(x)) atol = 10.0^(-15)
  @test transpose(getValue(retractionPolar(M,x,ω))) * getValue(retractionPolar(M,x,ω)) - one(getValue(x)) ≈ zero(getValue(x)) atol = 10.0^(-15)
end
