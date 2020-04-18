@testset "Manopt Cyclic Proximal Point" begin
using Dates
n = 100
N = PowerManifold(Circle(),n)
f = artificial_S1_signal(n)
F = x -> costL2TV(N,f,0.5,x)
proxes = [ (λ,x) -> proxDistance(N,λ,f,x), (λ,x) -> proxTV(N,0.5*λ,x) ]
o = cyclic_proximal_point(
    N,
    F,
    proxes,
    f;
    λ = i -> π/(2*i),
    stoppingCriterion = StopAfterIteration(100),
    debug = [DebugIterate()," ",DebugCost()," ",DebugProximalParameter(),"\n",10000],
    record = [RecordProximalParameter(), RecordIterate(f), RecordCost()],
    returnOptions=true
    )
fR = get_solver_result(o)
fR2 = cyclic_proximal_point(
    N,
    F,
    proxes,
    f;
    λ = i -> π/(2*i),
    stoppingCriterion = StopAfterIteration(100),
)
@test fR == fR2
rec = get_record(o)
@test F(f) > F(fR)
#
o = CyclicProximalPointOptions(f, StopAfterIteration(1), i -> π/(2*i))
p = ProximalProblem(N,F,proxes,[1,2])
@test_throws ErrorException getProximalMap(p,1.,f,3)
end