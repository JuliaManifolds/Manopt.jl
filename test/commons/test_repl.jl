using Manopt, Test

@testset "Test REPL printing" begin
    M = Euclidean(1)
    f(M, p) = sum(p .^ 2)
    o = ManifoldCostObjective(f)
    s = Manopt.Test.DummyState()
    io = IOBuffer()
    show(io, MIME"text/plain"(), (o, s))
    # objetived are ignored in general here
    r = String(take!(io))
    @test r == Manopt.status_summary(s)
    c = ManifoldCountObjective(M, o, Dict{Symbol, Int}(:Count => 0))
    show(io, MIME"text/plain"(), (c, s))
    # Statistics
    r2 = String(take!(io))
    @test  contains(r2, ":Count")
    @test  contains(r2, r)
end
