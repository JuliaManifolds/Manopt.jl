#
# Check that deprecated but not yet removed constructors still work (without erroring)
#
using Manopt, ManifoldsBase, Test

@testset "Deprecated Constructors" begin
    M = ManifoldsBase.DefaultManifold(2)
    p = zeros(2)
    X = zeros(2)
    s = StopAfterIteration(10)
    ChambollePockOptions(p, p, p, X)
    CyclicProximalPointOptions(p, s)
    DouglasRachfordOptions(p)
    StochasticGradientDescentOptions(p, X, StochasticGradient(X))
    SubGradientMethodOptions(M, p, s, ConstantStepsize(1.0))
    d = QuasiNewtonMatrixDirectionUpdate(M, SR1(), DefaultBasis(), zeros(2, 2))
    QuasiNewtonOptions(p, X, d, s, ConstantStepsize(1.0))
    TruncatedConjugateGradientOptions(HessianProblem(M, x -> x, x-> x, x-> x, :s),p,X,1.0,true)
end
