@testset "Data" begin
    @test artificialIn_SAR_image(2) == 2 * π * ones(2, 2)

    @test artificial_S1_slope_signal(20, 0.0) == repeat([-π / 2], 20)

    @test ismissing(artificial_S1_signal(-1.0))
    @test ismissing(artificial_S1_signal(2.0))
    @test artificial_S1_signal(2) == [-3 * π / 4, -3 * π / 4]

    # for the remainder check data types only
    @test length(artificial_S1_signal(20)) == 20

    @test size(artificial_S2_whirl_image(64)) == (64, 64)
    @test length(artificial_S2_whirl_image(64)[1, 1]) == 3

    @test size(artificial_S2_rotation_image(64)) == (64, 64)
    @test length(artificial_S2_rotation_image(64)[1, 1]) == 3

    @test size(artificial_S2_whirl_patch(8)) == (8, 8)
    @test length(artificial_S2_whirl_patch(8)[1, 1]) == 3

    @test size(artificial_SPD_image(8)) == (8, 8)
    @test size(artificial_SPD_image(8)[1, 1]) == (3, 3)

    @test size(artificial_SPD_image2(8)) == (8, 8)
    @test size(artificial_SPD_image2(8)[1, 1]) == (3, 3)
    @test eltype(artificial_SPD_image2(8)) == Array{Float64,2}

    @test length(artificial_S2_lemniscate([0.0, 0.0, 1.0], 20)) == 20
    @test length(artificial_S2_lemniscate([0.0, 0.0, 1.0], 20)[1]) == 3
end
