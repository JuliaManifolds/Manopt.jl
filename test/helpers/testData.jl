@testset "Data" begin

@test artificialInSARImage(2) == S1Point.(zeros(2,2))
@test artificialInSARImage(2,RnPoint) == RnPoint.(2*π*ones(2,2))
@test artificialInSARImage(2,SnPoint) == repeat([SnPoint([1.,0.])],2,2)

@test artificialS1SlopeSignal(20,0.) == repeat([S1Point(-π/2)],20)

@test ismissing( artificialS1Signal(-1.) )
@test ismissing( artificialS1Signal(2.) )
@test artificialS1Signal(2) == S1Point.([-3*π/4,-3*π/4])

# for the remainder check data types only
@test length( artificialS1Signal(20) ) == 20

@test size( artificialS2WhirlImage(64) ) == (64,64)
@test length(getValue(artificialS2WhirlImage(64)[1,1])) == 3
@test eltype( getValue(artificialS2WhirlImage(64)) ) == SnPoint{Float64}

@test size( artificialS2RotationsImage(64) ) == (64,64)
@test length(getValue(artificialS2RotationsImage(64)[1,1])) == 3
@test eltype( getValue(artificialS2RotationsImage(64)) ) == SnPoint{Float64}

@test size( artificialS2WhirlPatch(8) ) == (8,8)
@test length(getValue(artificialS2WhirlPatch(8)[1,1])) == 3
@test eltype( getValue(artificialS2WhirlPatch(8)) ) == SnPoint{Float64}

@test size( artificialSPDImage(8) ) == (8,8)
@test size(getValue(artificialSPDImage(8)[1,1])) == (3,3)
@test eltype( getValue(artificialSPDImage(8)) ) == SPDPoint{Float64}

@test size( artificialSPDImage2(8) ) == (8,8)
@test size(getValue(artificialSPDImage2(8)[1,1])) == (3,3)
@test eltype( getValue(artificialSPDImage2(8)) ) == SPDPoint{Float64}

@test length( artificialS2Lemniscate(SnPoint([0.,0.,1.]),20) ) == 20
@test length( getValue( artificialS2Lemniscate(SnPoint([0.,0.,1.]),20)[1])) == 3
@test eltype( artificialS2Lemniscate(SnPoint([0.,0.,1.]),20)) == SnPoint{Float64}

end