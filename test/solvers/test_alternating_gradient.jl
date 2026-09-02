using Manopt, Manifolds, ManifoldsBase, Test, RecursiveArrayTools

# a retraction that errors as soon as it is used, to prove the keyword is honoured
struct TripRetraction <: AbstractRetractionMethod end
ManifoldsBase._retract!(::AbstractManifold, q, p, X, ::TripRetraction) = error("TripRetraction was used")
function ManifoldsBase._retract_fused!(::AbstractManifold, q, p, X, t::Number, ::TripRetraction)
    return error("TripRetraction was used")
end

@testset "Alternating Gradient Descent" begin
    # Note that this is merely an alternating gradient descent toy example
    M = Sphere(2)
    N = M × M
    data = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    f(N, p) = 1 / 2 * (distance(N[1], p[N, Val(1)], data[1])^2 + distance(N[2], p[N, Val(2)], data[2])^2)
    grad_f1(N, p) = -log(N[1], p[N, 1], data[1])
    grad_f1!(N, X, p) = (X .= -log(N[1], p[N, 1], data[1]))
    grad_f2(N, p) = -log(N[2], p[N, 2], data[2])
    grad_f2!(N, X, p) = (X .= -log(N[2], p[N, 2], data[2]))
    grad_f(N, p) = ArrayPartition([-log(N[i], p[N, i], data[i]) for i in [1, 2]]...)
    function grad_f!(N, X, p)
        log!(N[1], X[N, 1], p[N, 1], data[1])
        log!(N[2], X[N, 2], p[N, 2], data[2])
        return X .*= -1
    end
    p = ArrayPartition([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    objf = ManifoldAlternatingGradientObjective(f, grad_f)

    @testset "Test gradient access" begin
        Pf = DefaultManoptProblem(N, objf)
        objv = ManifoldAlternatingGradientObjective(f, [grad_f1, grad_f2])
        Pv = DefaultManoptProblem(N, objv)
        objf! = ManifoldAlternatingGradientObjective(
            f, grad_f!; evaluation = InplaceEvaluation()
        )
        Pf! = DefaultManoptProblem(N, objf!)
        objv! = ManifoldAlternatingGradientObjective(
            f, [grad_f1!, grad_f2!]; evaluation = InplaceEvaluation()
        )
        Pv! = DefaultManoptProblem(N, objv!)
        X = zero_vector(N, p)
        @test repr(Manopt.AlternatingGradientRule(X)) == "AlternatingGradientRule($X)"

        for P in [Pf, Pv, Pf!, Pv!]
            @test get_gradient(P, p)[N, 1] == grad_f(N, p)[N, 1]
            @test get_gradient(P, p)[N, 2] == grad_f(N, p)[N, 2]
            get_gradient!(P, X, p)
            @test X[N, 1] == grad_f(N, p)[N, 1]
            @test X[N, 2] == grad_f(N, p)[N, 2]
            @test get_gradient(P, p, 1) == grad_f(N, p)[N, 1]
            @test get_gradient(P, p, 2) == grad_f(N, p)[N, 2]
            X = zero_vector(N, p)
            get_gradient!(P, X[N, 1], p, 1)
            @test X[N, 1] == grad_f(N, p)[N, 1]
            get_gradient!(P, X[N, 2], p, 2)
            @test X[N, 2] == grad_f(N, p)[N, 2]
        end
    end
    @testset "Test show/repr" begin
        s1 = repr(objf)
        @test startswith(s1, "ManifoldAlternatingGradientObjective(")
        @test Manopt.status_summary(objf; context = :short) == s1
        @test startswith(Manopt.status_summary(objf; context = :inline), "An alternating gradient objective")
        s2 = Manopt.status_summary(objf)
        @test startswith(s2, "An alternating gradient objective")
        @test contains(s2, "## Functions")
    end
    @testset "Test high level interface" begin
        q = allocate(p)
        copyto!(N, q, p)
        q2 = allocate(p)
        copyto!(N, q2, p)
        q3 = alternating_gradient_descent(
            N, f, [grad_f1!, grad_f2!], p; order_type = :Linear, evaluation = InplaceEvaluation(),
        )
        r = alternating_gradient_descent!(
            N, f, [grad_f1!, grad_f2!], q;
            order_type = :Linear, evaluation = InplaceEvaluation(), return_state = true,
        )
        @test startswith(
            Manopt.status_summary(r; context = :default),
            "# Solver state for `Manopt.jl`s Alternating Gradient Descent Solver"
        )
        @test startswith(repr(r), "AlternatingGradientDescentState(; ")
        # the summary header shows the total iteration count, not the inner counter
        rc = alternating_gradient_descent!(
            N, f, [grad_f1!, grad_f2!], copy(N, q);
            order_type = :Linear, evaluation = InplaceEvaluation(), return_state = true,
            stopping_criterion = StopAfterIteration(7),
        )
        @test contains(Manopt.status_summary(rc; context = :default), "After 7 iterations")
        @test_throws DomainError AlternatingGradientDescentState(N; order_type = :WrongSymbol)
        # r has the same message as the internal stepsize
        @test Manopt.get_message(r) == Manopt.get_message(r.stepsize)
        @test isapprox(N, q3, q)
    end
    @testset "Callbacks" begin
        sk_record = Tuple{Symbol, Int}[]
        cb(symbol, problem, state, k) = append!(sk_record, [(symbol, k)])
        alternating_gradient_descent(
            N, f, [grad_f1!, grad_f2!], p;
            order_type = :Linear,
            evaluation = InplaceEvaluation(),
            stopping_criterion = StopAfterIteration(1),
            callbacks = cb,
        )
        @test sk_record == [
            (:BeforeInit, 0), (:Init, 0), (:BeforeStop, 0),
            (:BeforeStep, 1), (:Stepsize, 1), (:Step, 1), (:BeforeStop, 1), (:Stop, 1),
        ]
    end
    @testset "retraction_method is used" begin
        # the keyword was stored and shown but never used, neither in the update nor the line search
        s = alternating_gradient_descent(
            N, f, grad_f, copy(p); retraction_method = ProductRetraction(ProjectionRetraction(), ProjectionRetraction()),
            stopping_criterion = StopAfterIteration(1), return_state = true,
        )
        @test s.retraction_method == ProductRetraction(ProjectionRetraction(), ProjectionRetraction())
        # the line search has to agree with the update, else the accepted step violates its own Armijo condition
        @test s.stepsize.retraction_method == s.retraction_method
        # a retraction that is never allowed to be called proves the update really uses it
        @test_throws ErrorException alternating_gradient_descent(
            N, f, grad_f, copy(p); retraction_method = ProductRetraction(TripRetraction(), TripRetraction()),
            stopping_criterion = StopAfterIteration(2),
        )
        # the component of a ProductRetraction is selected for the per-component update
        @test Manopt._component_retraction(ProductRetraction(ProjectionRetraction(), ExponentialRetraction()), 2) ==
            ExponentialRetraction()
        @test Manopt._component_retraction(ProjectionRetraction(), 1) == ProjectionRetraction()
    end
end
