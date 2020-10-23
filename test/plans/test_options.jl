using Manifolds, Manopt, Test, ManifoldsBase

using Dates

struct TestProblem <: Problem end
struct TestOptions <: Options end

@testset "generic Options test" begin
    p = TestProblem()
    o = TestOptions()
    a = ArmijoLinesearch(1.0, ExponentialRetraction(), 0.99, 0.1)
    @test get_last_stepsize(p, o, a) == 1.0
    @test get_initial_stepsize(a) == 1.0
end

@testset "Decorator Options test" begin
    o = TestOptions()
    r = RecordOptions(o, RecordIteration())
    d = DebugOptions(o, DebugIteration())
    dr = DebugOptions(r, DebugIteration())

    @test has_record(o) == has_record(d)
    @test !has_record(o)
    @test has_record(r)
    @test has_record(r) == has_record(dr)

    @test is_options_decorator(r)
    @test is_options_decorator(dr)
    @test is_options_decorator(d)
    @test !is_options_decorator(o)

    @test dispatch_options_decorator(r) === Val(true)
    @test dispatch_options_decorator(dr) === Val(true)
    @test dispatch_options_decorator(d) === Val(true)
    @test dispatch_options_decorator(o) === Val(false)

    @test get_options(r) == o
    @test get_options(dr) == o
    @test get_options(d) == o
    @test get_options(o) == o
    @test get_options(o, Val(false)) == o

    @test Manopt._extract_val(Val(true))
    @test !Manopt._extract_val(Val(false))
end
