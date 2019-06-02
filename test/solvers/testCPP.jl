@testset "Manopt Cyclic Proximal Point" begin
using Dates
n = 100
N = Power(Circle(),(n,))
f = PowPoint(artificialS1Signal(n))
F = x -> costL2TV(N,f,0.5,x)
proxes = [ (λ,x) -> proxDistance(N,λ,f,x), (λ,x) -> proxTV(N,0.5*λ,x) ]
fR,rec = cyclicProximalPoint(N,F,proxes, f;
    λ = i -> π/(2*i),
    stoppingCriterion = stopWhenAll( stopAfter(Second(10)), stopAfterIteration(5000) ),
    debug = [DebugIterate()," ",DebugCost()," ",DebugProximalParameter(),"\n",10000],
    record = [RecordProximalParameter(), RecordIterate(f), RecordCost()]
    )
@test F(f) > F(fR)
#
o = CyclicProximalPointOptions(f, stopAfterIteration(1), i -> π/(2*i))
p = ProximalProblem(N,F,proxes,[1,2])
@test_throws ErrorException getProximalMap(p,1.,f,3)
end