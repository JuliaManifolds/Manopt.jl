@testset "StoppingCriteria" begin
    struct myStoppingCriteriaSet <: StoppingCriterionSet end
    @test_throws ErrorException getStoppingCriteriaArray(myStoppingCriteriaSet())
end