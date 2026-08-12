using Manifolds, Manopt, Test, ManifoldsBase

@testset "Generic Parameters" begin
    # test one internal fallback
    Manopt.get_parameter(:None, Val(:default)) === nothing
    @test_logs (:info, "Setting the `Manopt.jl` parameter :TestValue to Å.") Manopt.set_parameter!(
        :TestValue, "Å"
    )
    @test Manopt.get_parameter(:TestValue) == "Å"
    @test Manopt.get_parameter(:TestValue, :Dummy) == "Å" # Dispatch ignores second symbol
    @test_logs (:info, "Resetting the `Manopt.jl` parameter :TestValue to default.") Manopt.set_parameter!(
        :TestValue, ""
    ) # reset
    @test Manopt.get_parameter(:TestValue) === nothing
end
