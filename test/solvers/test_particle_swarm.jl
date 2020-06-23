@testset "Manopt Particle Swarm" begin
    # Test the particle swarm algorithm
    M = Euclidean(3,3)
    F(x) = sum(x);
    x0 = [1 2 3; 4 5 6; 7 8 9];
    o = particle_swarm(M,F;
        x0,
        record = [:Iteration, :Cost, 1],
        return_options = true
    )
    g = get_solver_result(o)
    rec = get_record(o)
    # the cost of g and the p[i]'s are not greater after one step 
    @test F(g) <= F(x0)
    @test F(p[1]) <= F(x0)
    # the cost of g is not greater than the cost of x0
end