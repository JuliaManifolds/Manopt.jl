using Aqua, Manopt

@testset "Aqua.jl" begin
    Aqua.test_all(
        Manopt;
        ambiguities=(exclude=[#For now exclude some high-level functions, since in their
            # different call schemes some ambiguities appear
            Manopt.truncated_conjugate_gradient_descent,
        ],
        broken=true),
        #stale_deps=(ignore=[:SomePackage]
        #deps_compat=(ignore=[:SomeOtherPackage],),
        #piracies=false,
    )
end