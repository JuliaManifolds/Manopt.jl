using Manifolds, Manopt, Test, LRUCache

@testset "Conjugate Residual" begin
    M = ℝ^2
    p = [1.0, 1.0]
    TpM = TangentSpace(M, p)

    Am = [2.0 1.0; 1.0 4.0]
    bv = [1.0, 2.0]
    ps = Am \ (-bv)
    X0 = [3.0, 4.0]
    A(M, X, V) = Am * V
    b(M, p) = bv

    slso = SymmetricLinearSystemObjective(A, b)
    pT = conjugate_residual(TpM, slso, X0)
    pT2 = conjugate_residual(TpM, A, b, X0)
    pT3 = conjugate_residual(TpM, A, b, X0; stopping_criterion = StopAfterIteration(3))
    pT4 = conjugate_residual(TpM, A, b, X0; warm_start = false)
    @test norm(ps - pT) < 3.0e-15
    @test norm(ps - pT4) < 3.0e-15
    Y0 = copy(X0)
    rT = conjugate_residual!(TpM, slso, Y0)
    @test rT === Y0            # in-place: result lands in the passed vector
    @test norm(ps - Y0) < 3.0e-15
    Y1 = copy(X0)
    r5 = conjugate_residual!(TpM, A, b, Y1; stopping_criterion = StopAfterIteration(3))
    @test r5 === Y1 # the function-based in-place entry exists and works in place
    @test norm(pT2 - pT) < 3.0e-15
    @test get_cost(TpM, slso, pT) < 5.0e-15
    s = repr(slso)
    @test startswith(s, "SymmetricLinearSystemObjective")
    s2 = Manopt.status_summary(slso)
    @test startswith(s2, "An objective modelling a symmetric linear system")
    cgrs = conjugate_residual(TpM, slso, X0; return_state = true)
    @test startswith(Manopt.status_summary(cgrs), "# Solver state for `Manopt.jl`s Conjugate Residual Method")
    @test startswith(repr(cgrs), "ConjugateResidualState(; ")
    # Start without warmstart – though for this setting we get a NaN
    X1 = conjugate_residual(TpM, slso, pT; warm_start = false)

    @testset "Callbacks" begin
        cr_record = Tuple{Symbol, Int}[]
        cb(symbol, problem, state, k) = append!(cr_record, [(symbol, k)])
        conjugate_residual(TpM, A, b, X0; callbacks = cb)
        @test cr_record[1:6] == [(:BeforeInit, 0), (:Init, 0), (:BeforeStop, 0), (:BeforeStep, 1), (:Stepsize, 1), (:Step, 1)]
    end

    @testset "Decorated objectives" begin
        # `count=`/`cache=` are accepted keywords, so the accessors must pass through the
        # decorators rather than the solver unwrapping them (which would bypass a cache)
        for kwargs in (
                (; count = [:Gradient]),
                (; count = [:Gradient], warm_start = false),
                (; cache = (:LRU, [:Cost, :Gradient], 10)),
            )
            @test norm(ps - conjugate_residual(TpM, A, b, X0; kwargs...)) < 3.0e-15
        end
        # the decorator has to survive into the solver, otherwise a cache is never used
        o_c, _ = conjugate_residual(
            TpM, A, b, X0; count = [:Gradient], return_state = true, return_objective = true
        )
        @test o_c isa Manopt.ManifoldCountObjective
    end

    scs = StopWhenRelativeResidualLess(1.0, 0.1)
    @test repr(scs) == "StopWhenRelativeResidualLess(1.0, 0.1)"
    @test startswith(Manopt.status_summary(scs), "A stopping criterion to stop when the relative residual is less")
end
