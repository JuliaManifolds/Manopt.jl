using Manifolds, Manopt, Test

@testset "Test Manifold Functions" begin
    @testset "Test automatic wrappers." begin
        # Test unwrapping of variables
        p = 1.0
        q = fill(p)
        @test Manopt.maybe_unwrap_variable(p, q) == p
        @test Manopt.maybe_unwrap_variable(p, q) isa Float64
        # Test unwrapping of variables
        p = 1.0
        q = [p]
        @test Manopt.maybe_unwrap_variable(p, q) == p
        @test Manopt.maybe_unwrap_variable(p, q) isa Float64
    end
    @testset "Test mutable function wrapper" begin
        M = Euclidean()
        p = 1.0 #Nonmutable
        f(M, p) = p^2
        mf = Manopt.MutableManifoldFunction(f, p)
        @test repr(mf) == "MutableManifoldFunction(f, Number)"
        @test Manopt.MutableManifoldFunction(mf, p) === mf #we never double wrap
        @test mf(M, fill(p)) == f(M, p)

        df = Manopt.Test.DummyDecoratedFunction(f, 1)
        mdf = Manopt.MutableManifoldFunction(df, p)
        # Test that parameter pass through works
        Manopt.set_parameter!(mdf, :Field, 2)
        @test Manopt.get_parameter(mdf, :Field) == 2
        # Test that status summary passthrough works
        @test Manopt.status_summary(mdf) == Manopt.status_summary(df)
    end
    @testset "Test in-place function wrapper" begin
        M = Sphere(1)
        p = [1.0, 0.0]
        grad_f(M, p) = [-p[2], p[1]]
        imf = Manopt.InplaceManifoldFunction(grad_f, :TangentVector)
        @test Manopt.InplaceManifoldFunction(imf, :TangentVector) === imf #we do not double wrap
        X = zero_vector(M, p)
        @test imf(M, X, p) == grad_f(M, p)
        @test X == grad_f(M, p)
        df = Manopt.Test.DummyDecoratedFunction(grad_f, 1)
        idf = Manopt.InplaceManifoldFunction(df, :TangentVector)
        # Test that parameter pass through works
        Manopt.set_parameter!(idf, :Field, 2)
        @test Manopt.get_parameter(idf, :Field) == 2
        # Test that status summary passthrough works
        @test Manopt.status_summary(idf) == Manopt.status_summary(df)
    end
end
