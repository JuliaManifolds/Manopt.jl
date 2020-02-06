@testset "Data" begin

@test artificialInSARImage(2) == 2*π*ones(2,2)

@test artificialS1SlopeSignal(20,0.) == repeat(-π/2,20)

@test ismissing( artificialS1Signal(-1.) )
@test ismissing( artificialS1Signal(2.) )
@test artificialS1Signal(2) == [-3*π/4,-3*π/4]

# for the remainder check data types only
@test length( artificialS1Signal(20) ) == 20

@test size( artificialS2WhirlImage(64) ) == (64,64)
@test length(artificialS2WhirlImage(64)[1,1]) == 3

@test size( artificialS2RotationsImage(64) ) == (64,64)
@test length(artificialS2RotationsImage(64)[1,1]) == 3

@test size( artificialS2WhirlPatch(8) ) == (8,8)
@test length(artificialS2WhirlPatch(8)[1,1]) == 3

@test size( artificialSPDImage(8) ) == (8,8)
@test size(artificialSPDImage(8)[1,1]) == (3,3)

@test size( artificialSPDImage2(8) ) == (8,8)
@test size(artificialSPDImage2(8)[1,1]) == (3,3)
@test eltype( artificialSPDImage2(8))  == SPDPoint{Float64}

@test length( artificialS2Lemniscate([0.,0.,1.]),20)  == 20
@test length( artificialS2Lemniscate([0.,0.,1.]),20)[1] == 3

end