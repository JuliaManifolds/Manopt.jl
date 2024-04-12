using Aqua

@testset "Aqua.jl" begin
    Aqua.test_all(
        Manopt;
        #ambiguities=(exclude=[SomePackage.some_function], broken=true),
        #stale_deps=(ignore=[:SomePackage],),
        #deps_compat=(ignore=[:SomeOtherPackage],),
        #piracies=false,
    )
end