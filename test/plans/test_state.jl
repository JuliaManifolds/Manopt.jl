using Manifolds, Manopt, Test, ManifoldsBase

using Dates

struct TestProblem <: AbstractManoptProblem end
struct TestState <: AbstractManoptSolverState end

@testset "generic State tests" begin
    p = TestProblem()
    s = TestState()
    a = ArmijoLinesearch(Euclidean(3); initial_stepsize=1.0)
    @test get_last_stepsize(p, s, a) == 1.0
    @test get_initial_stepsize(a) == 1.0
end
@testset "Decresaing Stepsize" begin
    dec_step = DecreasingStepsize(; length=10.0, factor=1.0, subtrahend=0.0, exponent=1.0)
    @test get_initial_stepsize(dec_step) == 10.0
    @test dec_step(TestProblem(), TestState(), 1) == 10.0
    @test dec_step(TestProblem(), TestState(), 2) == 5.0
end

@testset "Decorator State test" begin
    s = TestState()
    r = RecordSolverState(s, RecordIteration())
    d = DebugSolverState(s, DebugIteration())
    dr = DebugSolverState(r, DebugIteration())

    @test has_record(s) == has_record(d)
    @test !has_record(s)
    @test has_record(r)
    @test has_record(r) == has_record(dr)

    @test is_state_decorator(r)
    @test is_state_decorator(dr)
    @test is_state_decorator(d)
    @test !is_state_decorator(s)

    @test dispatch_state_decorator(r) === Val(true)
    @test dispatch_state_decorator(dr) === Val(true)
    @test dispatch_state_decorator(d) === Val(true)
    @test dispatch_state_decorator(s) === Val(false)

    @test get_state(r) == s
    @test get_state(dr) == s
    @test get_state(d) == s
    @test get_state(s) == s
    @test Manopt._get_state(s, Val(false)) == s

    @test Manopt._extract_val(Val(true))
    @test !Manopt._extract_val(Val(false))

    @test_throws ErrorException get_gradient(s)
    @test_throws ErrorException get_gradient(r)
    @test_throws ErrorException get_iterate(s)
    @test_throws ErrorException get_iterate(r)
    @test_throws ErrorException set_iterate!(s, 0)
    @test_throws ErrorException set_iterate!(r, 0)
end
