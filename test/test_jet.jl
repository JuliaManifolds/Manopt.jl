using JET, Manifolds, ManifoldsBase, Manopt, Test

@testset "JET.jl" begin
    # only on released Julia versions, JET follows the compiler closely
    if isempty(VERSION.prerelease)
        JET.test_package(Manopt; target_modules = (Manopt, ManifoldsBase, Manifolds))
    end
end
