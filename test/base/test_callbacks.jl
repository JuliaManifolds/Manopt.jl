using Manifolds, Manopt, Test

@testset "Test callback utilities" begin
    # (a) providing a symbol and an array of symbol keys, expands the array
    cb = [:A => sin, [:B, :C] => cos]
    cb2 = Manopt.process_callbacks_arg(cb)
    @test cb2 == Dict(:A => sin, :B => cos, :C => cos)
    # Providing something else than these two yields an error
    @test_throws ArgumentError Manopt.process_callbacks_arg(["A" => sin])
    # test warning for unkown callbacks – GD does not have a :A callback
    @test_logs (:warn,) Manopt.process_callbacks_arg([:A => sin], GradientDescentState)
end
