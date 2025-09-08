using Manopt, Test

@testset "Keywords and their errors" begin
    @testset "Constructor" begin
        #With deprecated
        kwd = Manopt.Keywords(Set{Symbol}(), Set([:a, :b]))
        @test contains(repr(kwd), "deprecated")
        @test contains(repr(kwd), "accepted: none")
        @test contains(repr(kwd), "A set of") #default for from=nothing
        kwds = Manopt.Keywords(Set{Symbol}(), Set([:a, :b]); from = show)
        @test contains(repr(kwds), "show")
        @test contains(repr(kwd), "deprecated")
        kwds2 = copy(kwds)
        @test repr(kwds) == repr(kwds2)
        @test repr(Manopt.Keywords()) == "Keywords()"
        # Test repr for one without deprecatedkwd
        @test contains(repr(kwd), "accepted: none")
        kwa = Manopt.Keywords(Set([:a, :b]))
        @test contains(repr(kwa), "accepted")
        @test contains(repr(kwa), "* a")
        @test contains(repr(kwa), "* b")
    end
    @testset "check errors" begin
        @test Manopt.keywords_accepted(show, :error, Manopt.Keywords(Set([:a])))
        @test Manopt.keywords_accepted(show, :error, Manopt.Keywords(Set([:a])); a = 1)
        # Always warn on deprecations
        @test_logs (:warn,) Manopt.keywords_accepted(
            show, :none,
            Manopt.Keywords(Set{Symbol}(), Set([:a])); a = 1
        )
        @test_throws Manopt.ManoptKeywordError Manopt.keywords_accepted(
            show, :error,
            Manopt.Keywords(Set([:a])); b = 1
        )
    end
end
