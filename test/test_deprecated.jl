#
# Check that deprecated but not yet removed constructors still work (without erroring)
#
using Manopt, ManifoldsBase, Test

@testset "Deprecated Constructors" begin
    M = ManifoldsBase.DefaultManifold(2)
    p = zeros(2)
    X = zeros(2)
    s = StopAfterIteration(10)
    ChambollePockState(p, p, p, X)
    CyclicProximalPointState(p, s)
    DouglasRachfordState(p)
    StochasticGradientDescentState(p, X, StochasticGradient(X))
    SubGradientMethodState(M, p, s, ConstantStepsize(1.0))
    d = QuasiNewtonMatrixDirectionUpdate(M, SR1(), DefaultBasis(), zeros(2, 2))
    QuasiNewtonState(p, X, d, s, ConstantStepsize(1.0))
    TruncatedConjugateGradientState(
        HessianProblem(M, x -> x, x -> x, x -> x, :s), p, X, 1.0, true
    )
end
