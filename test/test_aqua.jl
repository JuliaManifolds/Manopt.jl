using Aqua, Manopt, Test

@testset "Aqua.jl" begin
    Aqua.test_all(
        Manopt;
        ambiguities = (broken = false,),
        # We define reflect! 8from LInearAlgebra) on an AbstractManifold (from ManifoldsBase).
        # This formally could be considered piracy; in the long run one could move this to ManifoldsBase.
        piracies = (treat_as_own = [reflect, reflect!],)
    )
end
