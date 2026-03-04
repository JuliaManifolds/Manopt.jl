using Manopt, Test

@testset "Robustifiers" begin
    @testset "Arctan robustifier" begin
        x = 1.0
        a = atan(x); b = 1 / (1 + x^2); c = -2 * x / (1 + x^2)^2
        @test Manopt.get_robustifier_values(ArctanRobustifier(), x) == (a, b, c)
    end
    @testset "Cauchy robustifier" begin
        x = 1.0
        a = log(x + 1); b = 1 / (1 + x); c = -1 / (1 + x)^2
        @test Manopt.get_robustifier_values(CauchyRobustifier(), x) == (a, b, c)
    end
    @testset "Componentwise robustifier" begin
        r = IdentityRobustifier()
        cr = ComponentwiseRobustifierFunction(r)
        @test ComponentwiseRobustifierFunction(cr).robustifier === cr.robustifier
        x = [1.0, 2.0]
        @test Manopt.get_robustifier_values(cr, x) == [Tuple(x), (1.0, 1.0), (0.0, 0.0)]
        @test Manopt.get_robustifier_values(cr, x[1]) == Manopt.get_robustifier_values(r, x[1])
    end
    @testset "Composed robustifier" begin
        r1 = ArctanRobustifier()
        r2 = CauchyRobustifier()
        r12 = r1 ∘ r2
        @test r12 == ComposedRobustifierFunction(r1, r2)
        x = 1.0
        (a2, b2, c2) = Manopt.get_robustifier_values(r2, x)
        (a1, b1, c1) = Manopt.get_robustifier_values(r1, a2)
        @test Manopt.get_robustifier_values(r12, x) == (a1, b1 * b2, c1 * b2^2 + b1 * c2)
    end
    @testset "Huber robustifier" begin
        x = 1.5
        a = 2 * sqrt(x) - 1; b = 1 / sqrt(x); c = 1 / (2 * x * sqrt(x))
        @test Manopt.get_robustifier_values(HuberRobustifier(), x) == (a, b, c)
        @test Manopt.get_robustifier_values(HuberRobustifier(), 0.5) == (0.5, 1.0, 0.0)
    end
    @testset "Identity robustifier" begin
        @test Manopt.get_robustifier_values(IdentityRobustifier(), 0.5) == (0.5, 1.0, 0.0)
    end
    @testset "Scaled robustifier" begin
        s = 1.5
        r = CauchyRobustifier()
        sr = ScaledRobustifierFunction(r, s)
        @test sr.scale == (s ∘ r).scale
        @test (s ∘ sr).scale == s^2
        x = 0.5
        (a, b, c) = Manopt.get_robustifier_values(r, x / s^2)
        @test Manopt.get_robustifier_values(sr, x) == (a * s^2, b, c / s^2)
    end
    @testset "Soft L1 robustifier" begin
        x = 1.5
        a = 2 * (sqrt(1 + x) - 1); b = 1 / sqrt(1 + x); c = -1 / (2 * sqrt(1 + x) * (1 + x))
        @test Manopt.get_robustifier_values(SoftL1Robustifier(), x) == (a, b, c)
        @test Manopt.get_robustifier_values(SoftL1Robustifier(), 0.0) == (0.0, 1.0, -0.5)
    end
    @testset "Tolerant robustifier" begin
        @test_throws ArgumentError TolerantRobustifier(1.0, 0.0)
        a = 1.0
        b = 0.5
        x = 1.5
        e1 = exp((x - a) / b)
        e2 = exp(-a / b)
        s1 = log(1 + e1)
        s2 = log(1 + e2)
        a1 = b * (s1 - s2)
        b1 = 1 / (1 + exp((a - x) / b))
        c1 = 1 / (4 * b * cosh((a - x) / (2b))^2)
        @test Manopt.get_robustifier_values(TolerantRobustifier(a, b), x) == (a1, b1, c1)
    end
    @testset "Tukey robustifier" begin
        @test Manopt.get_robustifier_values(TukeyRobustifier(), 0.0) == (0.0, 1.0, -2.0)
        x = 0.5
        a = (1 - (1 - x)^3) / 3
        b = (1 - x)^2
        c = 2 * (x - 1)
        @test Manopt.get_robustifier_values(TukeyRobustifier(), x) == (a, b, c)
        @test Manopt.get_robustifier_values(TukeyRobustifier(), 1.5) == (1 / 3, 0.0, 0.0)
    end
    @testset "RobustifierFunction" begin
        # Manual functions
        f(x) = 1
        g(x) = 2
        h(x) = 3
        rf = RobustifierFunction(f, g, h)
        @test Manopt.get_robustifier_values(rf, 1.0) == (1, 2, 3)
    end
end
