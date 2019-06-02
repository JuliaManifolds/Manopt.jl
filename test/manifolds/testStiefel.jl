@testset "The Stiefel-Manifold V(3,4)" begin
  import Random: seed!
  seed!(42);
  M = Stiefel(3,4)
  x = StPoint([1. 0. 0.; 0. 1. 0.; 0. 0. 1.; 0. 0. 0.])
  y = StPoint([0. 0. 0.; 1. 0. 0.; 0. 1. 0.; 0. 0. 1.])
  z = StPoint([-0.851712 0.0231522 0.502726; 0.509415 -0.0752401 0.741671; -0.0955954 -0.918939 -0.212126; -0.0771071 0.386462 -0.39012])
  η = StTVector([0.0 0.359296 -0.387003; -0.359296 0.0 0.0453968; 0.387003 -0.0453968 0.0; 0.0037522 -0.176039 0.638079])
  ω = StTVector([0.0 -0.162363 -0.639113; 0.162363 0.0 0.153497; 0.639113 -0.153497 0.0; 0.133987 0.155025 -0.203072])
  ξ = StTVector([0.0 -0.306164 0.31943; 0.306164 0.0 -0.467336; -0.31943 0.467336 0.0; -0.0815496 0.193606 -0.357094])
  ν = randomTVector(M,x)
  x2 = addNoise(M,x,0.5)
  x3 = addNoise(M,x,500.0)
  w = randomMPoint(M)
  s = pi

  # Test Stiefel
  @test_throws ErrorException Stiefel(4,3)
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
  # Test randomMPoint and randomTVector
  @test norm(transpose(getValue(w))*getValue(w) - one(transpose(getValue(w))*getValue(w))) ≈ 0 atol = 10.0^(-14)
  @test norm(transpose(getValue(x))*getValue(ν) + transpose(transpose(getValue(x))*getValue(ν))) ≈ 0 atol = 10.0^(-15)
  # Test addNoise
  @test norm(transpose(getValue(x2))*getValue(x2) - one(transpose(getValue(x2))*getValue(x2))) ≈ 0 atol=10.0^(-15)
  @test norm(transpose(getValue(x3))*getValue(x3) - one(transpose(getValue(x3))*getValue(x3))) ≈ 0 atol=10.0^(-12)
  #Test retraction
  @test norm(getValue(inverseRetractionQR(M,x,retractionQR(M,x,ω))) - getValue(ω)) ≈ 0 atol = 10.0^(-14)
  @test norm(getValue(inverseRetraction(M,x,retraction(M,x,ω))) - getValue(ω)) ≈ 0 atol = 10.0^(-14)
  @test norm(getValue(inverseRetractionPolar(M,x,retractionPolar(M,x,ω))) - getValue(ω)) ≈ 0 atol = 10.0^(-14)
  @test norm(transpose(getValue(retractionQR(M,x,ω))) * getValue(retractionQR(M,x,ω)) - one(transpose(getValue(x))*getValue(x))) ≈ 0 atol = 10.0^(-15)
  @test norm(transpose(getValue(retraction(M,x,ω))) * getValue(retraction(M,x,ω)) - one(transpose(getValue(x))*getValue(x))) ≈ 0 atol = 10.0^(-15)
  @test norm(transpose(getValue(retractionPolar(M,x,ω))) * getValue(retractionPolar(M,x,ω)) - one(transpose(getValue(x))*getValue(x))) ≈ 0 atol = 10.0^(-15)
  #Test manifoldDimension
  @test manifoldDimension(M) == manifoldDimension(x)
  @test manifoldDimension(M) == 6
  @test manifoldDimension(x) == 6
  # Test distance – not yet implemented
  @test_throws DomainError distance(M,x,y)
  # Test parallelTransport
  @test norm(getValue(parallelTransport(M,x,z,η)) - getValue(projection(M,z,getValue(η)))) ≈ 0 atol = 10.0^(-16)
  # Test zeroTVector
  @test norm(M,x,zeroTVector(M,x)) ≈ 0 atol = 10.0^(-16)
  # Test validateMPoint and validateTVector
  ynot1 = StPoint([1. 0. 0.; 0. 1. 0.; 0. 0. 1.; 0. 0. 0.; 0. 0. 0.])
  ynot2 = StPoint([1. 0. 0. 0.; 0. 1. 0. 0.; 0. 0. 1. 0.; 0. 0. 0. 1.])
  ynot3 = StPoint([1. 0. 0.; 0. 1. 0.; 0. 0. 0; 0. 0. 0.])
  ynot4 = StPoint([1. 2. 0.; 0. 1. 0.; 0. 4. 0; 5. 0. 6.])
  ξnot1 = StTVector([1. 0. 0.; 0. 1. 0.; 0. 0. 1.; 0. 0. 0.; 0. 0. 0.])
  ξnot2 = StTVector([1. 0. 0. 0.; 0. 1. 0. 0.; 0. 0. 1. 0.; 0. 0. 0. 1.])
  ξnot3 = StTVector([ 1. 2. 3.; 4. 5. 6.; 7. 8. 9.; 0. 0. 0.])
  @test_throws ErrorException validateMPoint(M,ynot1)
  @test_throws ErrorException validateMPoint(M,ynot2)
  @test_throws ErrorException validateMPoint(M,ynot3)
  @test_throws ErrorException validateMPoint(M,ynot4)
  @test_throws ErrorException validateTVector(M,x,ξnot1)
  @test_throws ErrorException validateTVector(M,x,ξnot2)
  @test_throws ErrorException validateTVector(M,x,ξnot3)


  N = Stiefel{Complex{Float64}}(3,4)
  xcompl = StPoint{Complex{Float64}}([-0.761286+0.0462087im -0.18395+0.0566532im 0.525627+0.265715im; -0.104276+0.262847im -0.55139+0.157097im -0.582766+0.202596im; -0.324463+0.199309im -0.041674-0.427848im -0.202927+0.0456294im; -0.327554-0.29335im -0.078241+0.66583im -0.232484-0.418496im])
  ycompl = StPoint{Complex{Float64}}(Complex{Float64}[-0.503249+0.0451954im -0.106808-0.271355im -0.422033-0.674388im; 0.263371-0.361562im -0.212757-0.435859im -0.208488+0.233779im; 0.598422-0.153485im 0.289058+0.159355im -0.254219-0.321635im; -0.240499-0.32419im 0.252482+0.712056im -0.314129+0.0470639im])
  zcompl = StPoint{Complex{Float64}}(Complex{Float64}[-0.385214-0.00723968im -0.0221375-0.419541im -0.518744-0.430094im; -0.286931+0.256567im -0.174173+0.01588im 0.00148874-0.527176im; 0.516436+0.382863im 0.592404-0.321075im -0.062389-0.136172im; 0.0362203-0.537401im 0.322842+0.484405im -0.49128-0.0649267im])
  ηcompl = StTVector{Complex{Float64}}(Complex{Float64}[-0.144443+0.050361im 0.0883068-0.262202im -0.0739903+0.145955im; -0.117226-0.00340324im 0.0298396+0.182582im -0.0763624-0.314719im; 0.423676-0.0952191im 0.0747197-0.168339im -0.448123+0.0629099im; -0.227158+0.141742im 0.437945-0.0236934im 0.0141424+0.170016im])
  ωcompl = StTVector{Complex{Float64}}(Complex{Float64}[0.0340014+0.025003im -0.175668-0.373111im 0.0973745-0.27843im; 0.0580946-0.252544im -0.13527+0.203748im -0.121749+0.270144im; 0.0677755-0.266317im -0.198544+0.306502im 0.0923344+0.0506648im; -0.151337-0.418159im 0.208301+0.0321241im 0.243117+0.0715284im])
  ξcompl = StTVector{Complex{Float64}}(Complex{Float64}[0.0416013-0.190526im 0.114072-0.0248113im 0.0703455+0.367732im; 0.124414+0.38413im -0.225464-0.120304im 0.39404-0.00423936im; 0.0805494-0.322981im 0.2325+0.234871im -0.151441-0.20267im; -0.0148454-0.129968im -0.314709+0.00379301im -0.0887356-0.128297im])
  νcompl = randomTVector(N,xcompl)
  wcompl = randomMPoint(N)
  x2compl = addNoise(N,xcompl,0.5)
  x3compl = addNoise(N,xcompl,500.0)

  # Test Stiefel
  @test_throws ErrorException Stiefel{Complex{Float64}}(4,3)
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
  @test norm(getValue(wcompl)'*getValue(wcompl) - one(getValue(wcompl)'*getValue(wcompl))) ≈ 0 atol = 10.0^(-14)
  @test norm(getValue(xcompl)'*getValue(νcompl) + (getValue(xcompl)'*getValue(νcompl))') ≈ 0 atol = 10.0^(-5)
  # Test addNoise
  @test norm(getValue(x2compl)'*getValue(x2compl) - one(getValue(x2compl)'*getValue(x2compl))) ≈ 0 atol=10.0^(-5)
  @test norm(getValue(x3compl)'*getValue(x3compl) - one(getValue(x3compl)'*getValue(x3compl))) ≈ 0 atol=10.0^(-5)
  # Test parallelTransport
  @test norm(getValue(parallelTransport(N,xcompl,zcompl,ηcompl)) - getValue(projection(N,zcompl,getValue(ηcompl)))) ≈ 0 atol = 10.0^(-16)

  #Test manifoldDimension
  @test manifoldDimension(N) == manifoldDimension(wcompl)
  @test manifoldDimension(N) == 15
  @test manifoldDimension(xcompl) == 15
end
