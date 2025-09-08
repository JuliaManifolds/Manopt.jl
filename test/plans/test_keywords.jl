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
    @testset "Error printing" begin
        kwd = Manopt.Keywords(Set([:a, :b]))
        err = Manopt.ManoptKeywordError(show, kwd)
        io = IOBuffer()
        showerror(io, err)
        str = String(take!(io))
        @test contains(str, "show does not accept the keyword(s)")
        @test contains(str, "* a")
        @test contains(str, "* b")
        @test contains(str, "does accept the following") # From Hint
        # With accepted kws
        kwd2 = Manopt.Keywords(Set([:a]), Set([:b]))
        err2 = Manopt.ManoptKeywordError(show, kwd2)
        io2 = IOBuffer()
        showerror(io2, err2)
        str2 = String(take!(io2))
        @test contains(str2, "show does not accept the keyword(s)")
        @test contains(str2, "* a")
        @test contains(str2, "show accepts, but deprecates the keyword(s):")
        @test contains(str2, "b")
    end
end
