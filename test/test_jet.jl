using JET, Manopt, Test
# load every package Manopt has an extension for, so that JET analyses the extensions as well
using LRUCache, LineSearches, Manifolds, QuadraticModels, RecursiveArrayTools, RipQP

@testset "JET.jl" begin
    # only on released Julia versions, JET follows the compiler closely
    if isempty(VERSION.prerelease)
        JET.test_package(Manopt; target_modules = (Manopt,))
    end
end
