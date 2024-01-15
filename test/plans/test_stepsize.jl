using Manopt, Manifolds, Test

@testset "Stepsize" begin
    @test Manopt.get_message(ConstantStepsize(1.0)) == ""
    s = ArmijoLinesearch()
    @test startswith(repr(s), "ArmijoLinesearch() with keyword parameters\n")
    s_stat = Manopt.status_summary(s)
    @test startswith(s_stat, "ArmijoLinesearch() with keyword parameters\n")
    @test endswith(s_stat, "of 1.0")
    @test Manopt.get_message(s) == ""

    s2 = NonmonotoneLinesearch()
    @test startswith(repr(s2), "NonmonotoneLinesearch() with keyword arguments\n")
    @test Manopt.get_message(s2) == ""

    s2b = NonmonotoneLinesearch(Euclidean(2)) # with manifold -> faster storage
    @test startswith(repr(s2b), "NonmonotoneLinesearch() with keyword arguments\n")

    s3 = WolfePowellBinaryLinesearch()
    @test Manopt.get_message(s3) == ""
    @test startswith(repr(s3), "WolfePowellBinaryLinesearch(DefaultManifold(), ")
    # no stepsize yet so repr and summary are the same
    @test repr(s3) == Manopt.status_summary(s3)
    s4 = WolfePowellLinesearch()
    @test startswith(repr(s4), "WolfePowellLinesearch(DefaultManifold(), ")
    # no stepsize yet so repr and summary are the same
    @test repr(s4) == Manopt.status_summary(s4)
    @test Manopt.get_message(s4) == ""
    @testset "Linesearch safeguards" begin
        M = Euclidean(2)
        f(M, p) = sum(p .^ 2)
        grad_f(M, p) = sum(2 .* p)
        p = [2.0, 2.0]
        s1 = Manopt.linesearch_backtrack(
            M, f, p, grad_f(M, p), 1.0, 1.0, 0.99; stop_decreasing_at_step=10
        )
        @test startswith(s1[2], "Max decrease")
        s2 = Manopt.linesearch_backtrack(
            M,
            f,
            p,
            grad_f(M, p),
            1.0,
            1.0,
            0.5,
            grad_f(M, p);
            retraction_method=ExponentialRetraction(),
        )
        @test startswith(s2[2], "The search direction")
        s3 = Manopt.linesearch_backtrack(
            M, f, p, grad_f(M, p), 1.0, 1.0, 0.5; stop_when_stepsize_less=0.75
        )
        @test startswith(s3[2], "Min step size (0.75)")
        # cheating for increase
        s4 = Manopt.linesearch_backtrack(
            M, f, p, grad_f(M, p), 1e-12, 0, 0.5; stop_when_stepsize_exceeds=0.1
        )
        @test startswith(s4[2], "Max step size (0.1)")
        s5 = Manopt.linesearch_backtrack(
            M, f, p, grad_f(M, p), 1e-12, 0, 0.5; stop_increasing_at_step=1
        )
        @test startswith(s5[2], "Max increase steps (1)")
    end
    @testset "Adaptive WN Gradient" begin
        # Build a dummy f, grad_f
        f(M, p) = 0
        grad_f(M, p) = [0.0, 0.75, 0.0] # We only stay at north pole
        M = Sphere(2)
        p = [1.0, 0.0, 0.0]
        mgo = ManifoldGradientObjective(f, grad_f)
        mp = DefaultManoptProblem(M, mgo)
        s = AdaptiveWNGradient(; gradient_reduction=0.5, count_threshold=2)
        gds = GradientDescentState(M, p)
        @test get_initial_stepsize(s) == 1.0
        @test get_last_stepsize(s) == 1.0
        @test s(mp, gds, 0) == 1.0
        @test s(mp, gds, 1) == 0.64 # running into the last case
        @test s.count == 0 # unchanged
        @test s.weight == 0.75 # unchanged
        # tweak bound
        s.gradient_reduction = 10.0
        @test s(mp, gds, 2) ≈ 0.5201560468140443 # running into the last case
        @test s.count == 1 # We ran into case 2
        @test s(mp, gds, 3) ≈ 3.1209362808842656
        @test s.count == 0 # was reset
        @test s.weight == 0.75 #also reset to orig
        @test startswith(repr(s), "AdaptiveWNGradient(;\n  ")
    end
    @testset "Absolute stepsizes" begin
        # Build a dummy f, grad_f
        f(M, p) = 0
        grad_f(M, p) = [0.0, 0.75, 0.0] # We only stay at north pole
        M = Sphere(2)
        p = [1.0, 0.0, 0.0]
        mgo = ManifoldGradientObjective(f, grad_f)
        mp = DefaultManoptProblem(M, mgo)
        gds = GradientDescentState(M, p)
        abs_dec_step = DecreasingStepsize(;
            length=10.0, factor=1.0, subtrahend=0.0, exponent=1.0, type=:absolute
        )
        solve!(mp, gds)
        @test abs_dec_step(mp, gds, 1) ==
            10.0 / norm(get_manifold(mp), get_iterate(gds), get_gradient(gds))
        abs_const_step = ConstantStepsize(1.0, :absolute)
        @test abs_const_step(mp, gds, 1) ==
            1.0 / norm(get_manifold(mp), get_iterate(gds), get_gradient(gds))
    end

    @testset "Hager-Zhang" begin
        M = Euclidean(4)

        function f(::AbstractManifold, p)
            result = 0.0
            for i in 1:2:length(p)
                result += (1.0 - p[i])^2 + 100.0 * (p[i + 1] - p[i]^2)^2
            end
            return result
        end

        function grad_f(::AbstractManifold, storage, p)
            for i in 1:2:length(p)
                storage[i] = -2.0 * (1.0 - p[i]) - 400.0 * (p[i + 1] - p[i]^2) * p[i]
                storage[i + 1] = 200.0 * (p[i + 1] - p[i]^2)
            end
            return storage
        end

        p = [1.0, 0.0, 0.0, 0.0]
        mgo = ManifoldGradientObjective(f, grad_f; evaluation=InplaceEvaluation())
        mp = DefaultManoptProblem(M, mgo)
        gds = GradientDescentState(M, p)
        initialize_solver!(mp, gds)
        s = HagerZhangLinesearch(M)
        @test s(mp, gds, 1) ≈ 0.0007566791242981903
        gds.p = [1.0, 1.0, 1.0, 1.0]
        grad_f(M, gds.X, gds.p)
        @test_throws ErrorException s(mp, gds, 1)
        gds.p = 1e10 .* one.(gds.p)
        grad_f(M, gds.X, gds.p)
        @test_throws ErrorException s(mp, gds, 1)

        function test_ϕdϕ_1(α)
            return (Inf, Inf)
        end
        @test_throws ErrorException s(test_ϕdϕ_1, 1.0, 1.0, 1.0)
        function test_ϕdϕ_2(α)
            if α < 1e-80
                return (1.0, 1.0)
            else
                return (Inf, Inf)
            end
        end
        @test s(test_ϕdϕ_2, 1.0, 2.0, 0.0)[1] == 0.0
        # the test below tests failure to get to a finite value
        @test s(test_ϕdϕ_2, 1.0, 2.0, -1.0)[1] == 0.0

        seq_A = Dict(
            5.0 => (2.204235476457027, 0.3939091703282501),
            1.0003004969933067 => (0.11139471010319403, 0.46673400040724694),
            1.000000000000001 => (-0.7609266443457385, -0.23870772915953856),
            1.0000000003108975 => (0.8710828953352873, 0.1894839768058072),
            1.0000000000000073 => (1.45925511098195, -1.287709475949499),
            1.3451609118081171 => (-0.8706167493146322, 2.214001105111046),
            1.0057036996096567 => (1.1272458746417433, 0.668899021557659),
            1.0000000216932503 => (-0.06369954282901213, -0.30871724503177106),
            1.0000000006217953 => (0.07061459767374156, -0.2922164989842865),
            1.0 => (-1.4369441931092128, -0.03720055660982347),
            1.0000000000000013 => (-0.15390779207560273, -0.6599826177249115),
            1.0000000108466252 => (-0.14188658979183247, 0.6117274750177726),
            1.0000000000510205 => (0.6070941124405822, 0.39989230406034443),
            1.0000000000000018 => (0.8038319181589618, -0.5949965880344299),
            1.0000000867730008 => (-0.42550361109987567, -0.7365359017567995),
            1.0000000000000009 => (-2.254336280197326, -0.6675630365783874),
            1.0000000000000036 => (0.13396618577410138, -1.1637857690940683),
            1.0000110913761073 => (0.5722836955879423, 1.3569659714442532),
            1.0000000000043423 => (0.6203171094108477, 0.708795491068729),
            1.0000221827522144 => (0.104097146457823, -0.003530491853577571),
            1.0000000000002165 => (-0.12166874725079804, 0.03417553984851831),
            1.0000002959512773 => (0.05307818374176109, 0.08967706826239678),
            1.0000000433865004 => (-0.34693123718302615, -0.034497535027746384),
        )
        function cf(dct)
            return function cf_inner(α)
                min_dist = Inf
                min_val = (Inf, Inf)
                for k in keys(dct)
                    cdist = abs(k - α)
                    if cdist < min_dist
                        min_dist = cdist
                        min_val = dct[k]
                    end
                end
                return min_val
            end
        end
        test_ϕdϕ_3 = cf(seq_A)
        @test s(test_ϕdϕ_3, 1.0, test_ϕdϕ_3(1.0)...)[1] ≈ 1.0

        # for i in 1:1000
        #     dct = Dict{Float64,Tuple{Float64,Float64}}()
        #     function test_ϕdϕ_k(α)
        #         if !haskey(dct, α)
        #             dct[α] = randn(), randn()
        #         end
        #         return dct[α]
        #     end
        #     try
        #         s(test_ϕdϕ_k, 1.0, test_ϕdϕ_k(1.0)...)
        #     catch e
        #         if e isa ArgumentError
        #    #         println(repr(dct))
        #         end
        #     end
        # end
    end
end
