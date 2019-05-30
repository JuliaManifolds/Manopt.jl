@testset "The Gr(4,6)" begin
  import Base: zeros, one
  import Random: seed!
  seed!(42); #set seed -> at least always the same random numbers.
  M = Grassmannian(4, 6)
  x = GrPoint([1. 0. 0. 0.; 0. 1. 0. 0.; 0. 0. 1. 0.; 0. 0. 0. 1.; 0. 0. 0. 0.; 0. 0. 0. 0.])
  yanti = GrPoint([1. 0. 0. 0.; 0. 0. 0. 0.; 0. 1. 0. 0.; 0. 0. 1. 0.; 0. 0. 0. 1.; 0. 0. 0. 0.])
  y = randomMPoint(M)
  z = randomMPoint(M)
  η = randomTVector(M,x)
  ξ = log(M, x ,y)
  μ = log(M, x, z)
  y2 = exp(M, x, ξ)
  z2 = exp(M, x, μ)
  x2 = addNoise(M,x,0.5)
  x3 = addNoise(M,x,500.)
  w = randomMPoint(M)
  s = rand(Float64)
  μ2 = log(M,x,w)
  w2 = exp(M,x,μ2)

  # Test Grassmannian
  @test_throws ErrorException Grassmannian(6, 4)
  # Test Dimension
  @test manifoldDimension(x)==8
  @test manifoldDimension(M)==8
  # Test Minus operator
  @test getValue( (-η) ) == getValue(-η)
  # Test Cauchy-Schwarz-Inequation
  @test dot(M,x,ξ,η)^2 <= norm(M,x,ξ)^2*norm(M,x,η)^2
  @test dot(M,x,ξ,η) <= norm(M,x,ξ)*norm(M,x,η)
  # Test dot
  @test dot(M,x,ξ,η) ≈ dot(M,x,η,ξ) atol = 10. ^(-16)
  @test dot(M,x,μ,η+ξ) ≈ dot(M,x,μ,η) + dot(M,x,μ,ξ) atol = 10. ^(-15)
  @test dot(M,x,μ,s*η) ≈ s*dot(M,x,μ,η) atol = 10. ^(-15)
  # Test norm
  @test norm(M,x,μ)^2 ≈ dot(M,x,μ,μ) atol = 10. ^(-12)
  @test norm(M,x,μ) - sqrt(dot(M,x,μ,μ)) ≈ 0 atol = eps()
  # Test distance
  @test distance(M,x,x) ≈ 0 atol = 10. ^(-7)
  @test distance(M,x,z) - norm(M,x,μ) ≈ 0 atol = 10. ^(-7)
  @test distance(M,x,yanti) ≈ pi/2 atol = 10. ^(-16)
  # Test randomMPoint and randomTVector
  @test norm( getValue(x)'*getValue(η) )≈ 0 atol = 10. ^(-6)
  # Test addNoise
  @test norm( getValue(x2)'*getValue(x2) - one(getValue(x2)'*getValue(x2)) ) ≈ 0 atol = 10. ^(-15)
  @test norm( getValue(x3)'*getValue(x3) - one(getValue(x3)'*getValue(x3)) ) ≈ 0 atol = 10. ^(-15)
  # Test Exponential and logarithm
  @test norm( getValue(log(M, x, y2)) - getValue(ξ)) ≈ 0 atol = 10. ^(-5)
  @test norm( getValue(log(M, x, w2)) - getValue(μ2) ) ≈ 0 atol =  10. ^(-13)
  @test norm( getValue(log(M, x, y)) - getValue(ξ) ) ≈ 0 atol = 10.0^(-16)
  @test norm( getValue(log(M, x, z)) - getValue(μ) ) ≈ 0 atol = 10.0^(-16)
  @test distance(M, exp(M,x,log(M,x,y)) , y ) ≈ 0 atol = 5*10. ^(-8)
  @test norm( getValue(log(M,x,exp(M,x,ξ))) - getValue(ξ) ) ≈ 0 atol = 10. ^(-15)
  @test_throws ErrorException log(M,x,yanti)
  #Test retraction
  @test norm( getValue(inverseRetraction(M,x,retraction(M,x,ξ))) - getValue(ξ)) ≈ 0 atol = 10.0^(-14)
  @test norm( transpose(getValue(retraction(M,x,ξ))) * getValue(retraction(M,x,ξ)) - one(transpose(getValue(x))*getValue(x)) ) ≈ 0 atol = 10.0^(-14)
  # Test parallelTransport
  @test norm(getValue(parallelTransport(M,x,z,η)) - projection(M,z,getValue(η))) ≈ 0 atol = 10.0^(-16)
  # Test zeroTVector
  @test norm(M,x,zeroTVector(M,x)) ≈ 0 atol = 10.0^(-16)
  # Test validateMPoint and validateTVector
  ynot1 = GrPoint([1. 0. 0. 0.; 0. 1. 0. 0.; 0. 0. 1. 0.; 0. 0. 0. 1.; 0. 0. 0. 0.; 0. 0. 0. 0.;  0. 0. 0. 0.])
  ynot2 = GrPoint([1. 0. 0. 0. 0.; 0. 1. 0. 0. 0.; 0. 0. 1. 0. 0.; 0. 0. 0. 1. 0.; 0. 0. 0. 0. 0.; 0. 0. 0. 0. 0.])
  ynot3 = GrPoint([1. 0. 0. 0.; 0. 1. 0. 0.; 0. 0. 1. 0.; 0. 0. 0. 0.; 0. 0. 0. 0.; 0. 0. 0. 0.])
  ynot4 = GrPoint([1. 0. 0. 0.; 0. 2. 0. 0.; 0. 0. 4. 0.; 0. 0. 0. 8.; 0. 0. 0. 0.; 0. 0. 0. 0.])
  ξnot1 = GrTVector([1. 0. 0. 0.; 0. 1. 0. 0.; 0. 0. 1. 0.; 0. 0. 0. 1.; 0. 0. 0. 0.; 0. 0. 0. 0.;  0. 0. 0. 0.])
  ξnot2 = GrTVector([1. 0. 0. 0. 0.; 0. 1. 0. 0. 0.; 0. 0. 1. 0. 0.; 0. 0. 0. 1. 0.; 0. 0. 0. 0. 0.; 0. 0. 0. 0. 0.])
  ξnot3 = GrTVector([ 1. 2. 3. 0.; 4. 5. 6. 0.; 7. 8. 9. 0.; 0. 0. 0. 0.; 1. 0. 0. 0.; 0. 1. 0. 0.])
  @test_throws ErrorException validateMPoint(M,ynot1)
  @test_throws ErrorException validateMPoint(M,ynot2)
  @test_throws ErrorException validateMPoint(M,ynot3)
  @test_throws ErrorException validateMPoint(M,ynot4)
  @test_throws ErrorException validateTVector(M,x,ξnot1)
  @test_throws ErrorException validateTVector(M,x,ξnot2)
  @test_throws ErrorException validateTVector(M,x,ξnot3)


  #
  # Complex manifold
  #
  N = Grassmannian{Complex{Float64}}(4,6)
  xcompl = randomMPoint(N)
  ycompl = randomMPoint(N)
  zcompl = randomMPoint(N)
  ηcompl = randomTVector(N, xcompl)
  ωcompl = randomTVector(N, xcompl)
  ξcompl = randomTVector(N, xcompl)
  νcompl = randomTVector(N,xcompl)
  wcompl = randomMPoint(N)
  x2compl = addNoise(N,xcompl,0.5)
  x3compl = addNoise(N,xcompl,500.0)

  # Test Minus operator
  @test getValue( (-ηcompl) ) == getValue(-ηcompl)
  # Test Cauchy-Schwarz-Inequation
  @test abs(dot(N,xcompl,ωcompl,ηcompl)^2) <= abs(norm(N,xcompl,ωcompl)^2*norm(N,xcompl,ηcompl)^2)
  @test dot(N,xcompl,ωcompl,ηcompl) <= norm(N,xcompl,ωcompl)*norm(N,xcompl,ηcompl)
  # Test dot
  @test dot(N,xcompl,ωcompl,ηcompl) ≈ dot(N,xcompl,ηcompl,ωcompl) atol = 10.0^(-16)
  @test dot(N,xcompl,ωcompl,ηcompl+ξcompl) ≈ dot(N,xcompl,ωcompl,ηcompl) + dot(N,xcompl,ωcompl,ξcompl) atol = 10.0^(-15)
  @test dot(N,xcompl,ωcompl,s*ηcompl) ≈ s*dot(N,xcompl,ωcompl,ηcompl) atol = 10.0^(-14)
  # Test norm
  @test norm(N,xcompl,ωcompl)^2 ≈ dot(N,xcompl,ωcompl,ωcompl) atol = 10.0^(-14)
  @test norm(N,xcompl,ωcompl) ≈ sqrt(dot(N,xcompl,ωcompl,ωcompl)) atol = 10.0^(-14)
  # Test randomMPoint and randomTVector
  @test norm( getValue(wcompl)'*getValue(wcompl) - one(getValue(wcompl)'*getValue(wcompl)) ) ≈ 0 atol = 10.0^(-14)
  @test norm( getValue(xcompl)'*getValue(νcompl) + (getValue(xcompl)'*getValue(νcompl))' ) ≈ 0 atol = 10.0^(-5)
  # Test addNoise
  @test norm( getValue(x2compl)'*getValue(x2compl) - one(getValue(x2compl)'*getValue(x2compl)) ) ≈ 0 atol=10.0^(-5)
  @test norm( getValue(x3compl)'*getValue(x3compl) - one(getValue(x3compl)'*getValue(x3compl)) ) ≈ 0 atol=10.0^(-6)

  #Test manifoldDimension
  @test manifoldDimension(N) == manifoldDimension(wcompl)
  @test manifoldDimension(N) == 16
  @test manifoldDimension(xcompl) == 16
end
