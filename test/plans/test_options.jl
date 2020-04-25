using Manifolds, Manopt, Test, ManifoldsBase

struct TestProblem <: Problem end
struct TestOptions <: Options end

@testset "generic Options test" begin
using Manifolds, Manopt, Test, ManifoldsBase

struct TestProblem <: Problem end
struct TestOptions <: Options end

p = TestProblem()
o = TestOptions()
a = ArmijoLinesearch(1.,ExponentialRetraction(),0.99,0.1)
@test get_last_stepsize(p,o,a) == 1.0
@test get_initial_stepsize(a) == 1.0
end