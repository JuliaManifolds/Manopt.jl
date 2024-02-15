using Manifolds, Manopt, Test, ManifoldsBase

@testset "Generic Parameters" begin
    # test one internal fallback
    Manopt.get_manopt_parameter(:None, Val(:default)) === nothing
    @test_logs (:info, "Setting the `Manopt.jl` parameter :TestValue to Å.") Manopt.set_manopt_parameter!(
        :TestValue, "Å"
    )
    @test Manopt.get_manopt_parameter(:TestValue) == "Å"
    @test_logs (:info, "Resetting the `Manopt.jl` parameter :TestValue to default.") Manopt.set_manopt_parameter!(
        :TestValue, ""
    ) # reset
    @test Manopt.get_manopt_parameter(:TestValue) === nothing
end
