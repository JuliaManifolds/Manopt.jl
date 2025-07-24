using ManifoldsBase, Manifolds, Manopt, Test, Random

@testset "Test Mesh Adaptive Plan" begin
    M = ManifoldsBase.DefaultManifold(3)
    f(M, p) = norm(p)
    mesh_size = 1.0
    cmp = DefaultManoptProblem(M, ManifoldCostObjective(f))
    @testset "Test Poll Accessors" begin
        ltap = LowerTriangularAdaptivePoll(M, zeros(3))
        # On init there was not yet a check
        @test !Manopt.is_successful(ltap)
        @test Manopt.get_descent_direction(ltap) == ltap.X
        @test Manopt.get_candidate(ltap) == ltap.candidate
        p2 = [2.0, 0.0, 0.0]
        @test Manopt.update_basepoint!(M, ltap, p2) === ltap
        @test Manopt.get_basepoint(ltap) == p2
        @test startswith(repr(ltap), "LowerTriangularAdaptivePoll\n")
        # test call
        Random.seed!(42)
        ltap(cmp, mesh_size)
        # check that this was successful
        @test Manopt.is_successful(ltap)
        # test call2 scale down!
        Random.seed!(42)
        ltap(cmp, mesh_size; max_stepsize = 1.0)
        # check that this was successful as well
        @test Manopt.is_successful(ltap)
        #... and short enough
        @test norm(M, p2, Manopt.get_descent_direction(ltap)) <= 1.0
    end

    @testset "Test Search Accessors" begin
        p = ones(3)
        dmads = DefaultMeshAdaptiveDirectSearch(M, p)
        @test !Manopt.is_successful(dmads)
        @test Manopt.get_candidate(dmads) == dmads.p
        @test startswith(repr(dmads), "DefaultMeshAdaptiveDirectSearch\n")
        X = -ones(3)
        # This step would bring us to zero, but we only allow a max step 1.0
        dmads(cmp, 1.0, p, X; max_stepsize = 1.0)
        # and that should still improve
        @test Manopt.is_successful(dmads)
    end

    @testset "State Accessors" begin
        p = ones(3)
        mads = MeshAdaptiveDirectSearchState(M, p)
        @test startswith(
            repr(mads), "# Solver state for `Manopt.jl`s mesh adaptive direct search\n"
        )
        @test get_iterate(mads) == p
        @test get_solver_result(mads) == p
    end

    @testset "Stopping Criteria" begin
        sps = StopWhenPollSizeLess(1.0)
        @test get_reason(sps) === ""
        @test startswith(repr(sps), "StopWhenPollSizeLess(1.0)")
    end
end
