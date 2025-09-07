using Manopt, ManifoldsBase, Manifolds, Test
using Random
@testset "Particle Swarm" begin
    # Test the particle swarm algorithm
    A = [1.0 3.0 4.0; 3.0 -2.0 -6.0; 4.0 -6.0 5.0]
    @testset "Euclidean Particle Swarm" begin
        M = Euclidean(3)
        f(::Euclidean, p) = (p' * A * p) / (p' * p)
        p1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        p2 = copy.(Ref(M), p1)

        Random.seed!(35)
        o = particle_swarm(M, f, p1; return_state = true)
        @test startswith(
            repr(o), "# Solver state for `Manopt.jl`s Particle Swarm Optimization Algorithm"
        )
        g = get_solver_result(o)

        initF = min(f.(Ref(M), p1)...)
        Random.seed!(35)
        g2 = particle_swarm(M, f, p2)
        @test f(M, g2) < initF
        @test isapprox(M, g, g2)
        p3 = copy.(Ref(M), p1)
        g3 = particle_swarm!(M, f, p3)
        @test f(M, g3) < initF
        g4 = particle_swarm(M, f)
        @test f(M, g4) < initF

        # the cost of g and the p[i]'s are not greater after one step
        j = argmin([f(M, y) for y in p1])
        g0 = deepcopy(p1[j])
        @test f(M, g) <= f(M, g0) # global did not get worse
        for (p, q) in zip(o.positional_best, p1)
            @test f(M, p) <= f(M, q) # non-increased
            # the cost of g is not greater than the cost of any p[i]
            @test f(M, g) <= f(M, p)
        end
    end
    @testset "Spherical Particle Swarm" begin
        Random.seed!(42)
        M = Sphere(2)
        f(::Sphere, p) = transpose(p) * A * p
        p_start = [rand(M) for i in 1:3]
        X_start = [rand(M; vector_at = y) for y in p_start]
        p = DefaultManoptProblem(M, ManifoldCostObjective(f))
        o = ParticleSwarmState(M, zero.(p_start), X_start)
        # test `set_iterate``
        Manopt.set_parameter!(o, :Population, p_start)
        @test sum(norm.(Manopt.get_parameter(o, :Population) .- p_start)) == 0
        initialize_solver!(p, o)
        step_solver!(p, o, 1)
        for (p, v) in zip(o.swarm, o.velocity)
            # verify that the new particle locations are on the manifold
            @test is_point(M, p, true)
            # verify that the new velocities are tangent vectors of the original particle locations
            @test is_vector(M, p, v, true; atol = 2.0e-15)
        end
        set_iterate!(o, p_start[1])
        @test get_iterate(o) == p_start[1]
    end
    @testset "Spherical Particle Swarm" begin
        Random.seed!(42)
        M = Circle()
        f(::Circle, p) = p * 2 * p
        swarm = [-π / 4, 0, π / 4]
        s = particle_swarm(M, f, swarm)
        @test s ≈ 0.0
    end
    @testset "Specific Stopping criteria" begin
        sc = StopWhenSwarmVelocityLess(1.0)
        @test startswith(repr(sc), "StopWhenSwarmVelocityLess")
        @test get_reason(sc) == ""
        # Trigger manually
        sc.at_iteration = 2
        sc.velocity_norms = [0.001, 0.001]
        @test length(get_reason(sc)) > 0
    end
end
