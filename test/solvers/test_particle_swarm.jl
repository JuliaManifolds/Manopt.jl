using Manopt, ManifoldsBase, Manifolds, Test
using Random
@testset "Particle Swarm" begin
    # Test the particle swarm algorithm
    A = [1.0 3.0 4.0; 3.0 -2.0 -6.0; 4.0 -6.0 5.0]
    @testset "Eucliedean Particle Swarm" begin
        M = Euclidean(3)
        F(::Euclidean, x) = (x' * A * x) / (x' * x)
        x_start = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        x_start2 = deepcopy(x_start)

        Random.seed!(35)
        o = particle_swarm(M, F; x0=x_start, return_options=true)
        g = get_solver_result(o)

        Random.seed!(35)
        g2 = particle_swarm(M, F; x0=x_start2, return_options=false)
        @test isequal(g, g2)

        # the cost of g and the p[i]'s are not greater after one step
        j = argmin([F(M, y) for y in x_start])
        g0 = deepcopy(x_start[j])
        @test F(M, g) <= F(M, g0)
        for i in 1:3
            @test F(M, o.p[i]) <= F(M, x_start[i])
            # the cost of g is not greater than the cost of any p[i]
            @test F(M, g) <= F(M, o.p[i])
        end
    end
    @testset "Spherical Particle Swarm" begin
        Random.seed!(42)
        M = Sphere(2)
        F(::Sphere, x) = transpose(x) * A * x
        x_start = [random_point(M) for i in 1:3]
        v_start = [random_tangent(M, y) for y in x_start]
        p = CostProblem(M, F)
        o = ParticleSwarmOptions(x_start, v_start)
        initialize_solver!(p, o)
        step_solver!(p, o, 1)
        for i in 1:3
            # check that the new particle locations are on the manifold
            @test is_point(M, o.x[i], true)
            # check that the new velocities are tangent vectors of the original particle locations
            @test is_vector(M, o.x[i], o.velocity[i], true; atol=2e-15)
        end
    end
end
