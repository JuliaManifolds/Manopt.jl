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
        seq_B = Dict(
            5.0 => (0.7259371655135298, 0.5804126806934048),
            1.129299254815666 => (1.7632419047721304, -0.04530353238725029),
            1.0646706831228128 => (-0.42413904952039644, -0.7598437792436201),
            1.258598509631332 => (-0.43667276626913726, -3.556509344168271),
            1.064670686593514 => (0.07801945614171368, -0.12652927852636134),
            1.064675463910083 => (-0.04815144793833015, 0.882008799030555),
            1.064670682936227 => (-1.4022247707937494, -0.9035239398446079),
            1.0646706847124994 => (1.3148878886672555, -0.17147825361755226),
            1.064670682936274 => (1.124514547669323, -0.045574248397638704),
            1.0646802449886816 => (0.9327475882039934, -0.6416289738649009),
            1.0666391509528683 => (1.873741067254798, 0.418368260059298),
            1.0653234009646897 => (2.065812800425832, -0.6583882142875422),
            1.064671212663789 => (0.41783962348724046, -0.7442701917689782),
            1.064670682938773 => (0.3081852925266191, 0.9471947802083749),
            1.0646706829771486 => (-0.35131565649169016, 0.26654876726615107),
            1.0646706915375426 => (0.5647880469593292, 0.8966130917044652),
            1.0646707002436004 => (-0.11412766713297222, -1.086182478851999),
            1.0646706829362502 => (0.41036466515913483, -0.2529898418938524),
            1.0646707490605227 => (0.15574768494518973, 0.5325270673036965),
            1.064670682936321 => (0.9345671555845769, -0.34869290030954153),
            1.0808120342597913 => (0.03528438615093296, 0.6559750275701111),
            1.0646917382551364 => (-0.44205971050029336, -0.25437968186735505),
            1.0649865141862613 => (-0.31195833325348143, -0.2447534715301276),
            1.0646706828314847 => (-0.5947891217400582, -0.6822542688626344),
            1.0646706829362504 => (-1.4573412164739803, -0.10685094711354381),
            1.0646706829424855 => (0.4858193317211344, 1.3171765796022632),
            1.0646727681017634 => (-0.36745228920254097, -0.5170469901689546),
            1.0646706829374701 => (0.5428500852286007, -1.8598172014371408),
            1.064671725466624 => (-0.7118777288961157, 0.6603257684352277),
            2.7680736978691876 => (-0.02017406964780661, 0.473012051999378),
            1.0646812105433106 => (-1.4705845640546895, 0.06889175254033084),
            1.8715038687625762 => (0.4108868205446942, -0.026573532675563238),
            1.0646706829678263 => (0.483990510608711, -0.4558177314933213),
            1.0646706829368484 => (1.0035643636442504, 0.20120458863752352),
            1.0646707200275154 => (1.6916140632051198, 0.775187837778444),
            1.0646706829362511 => (-0.5642023512502382, 1.1908504344400999),
            1.06473384910244 => (1.4300685952602727, -1.2747729854796468),
            1.0646706829364811 => (-1.6109494731326195, 1.5331937796797184),
            1.064670682948744 => (0.7463359987495404, 0.6130866531012353),
            1.0686286744979039 => (0.6201202913761912, -0.8835561492051031),
            1.0646708152895608 => (1.300485319765946, -0.05268221879538461),
            1.064649627407833 => (-1.227462883664307, -0.21423938680503848),
            1.0646709477476368 => (0.018132507931798745, -0.0487965583135743),
            1.0 => (-0.5592789953853469, -0.4597877598664593),
            1.4357519343812881 => (-1.4500743518637385, 0.31497852185836706),
            1.0969744411117495 => (-0.16934606908330757, -1.0176139919931706),
            1.0646706829520265 => (0.4913142262946372, 0.23696120871852777),
            1.064670683771992 => (-0.6153616814649749, 1.5202965131493111),
            1.0646706829367352 => (0.6874129669551705, -1.101673254494),
            1.0648180707970472 => (1.3591702026734238, -0.9611037897457907),
            1.0646706829362622 => (0.4376121894785113, 1.2016522790471396),
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
        test_ϕdϕ_4 = cf(seq_B)
        @test s(test_ϕdϕ_4, 1.0, test_ϕdϕ_4(1.0)...)[1] ≈ 1.0646706829362507

        # for i in 1:10000
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
        #             println(repr(dct))
        #         end
        #     end
        # end
    end
end
