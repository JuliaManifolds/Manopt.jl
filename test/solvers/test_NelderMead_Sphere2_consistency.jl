# test NelderMead consistency
# see lead-up discussions at Manopt #65

using Test
using Manopt, Manifolds
using Random


##

@testset "check NelderMead consistency on Sphere(2)" begin

## build problem (from Manopt docs)
n = 100
σ = π / 8
M = Sphere(2)
dim = manifold_dimension(M)
x = 1 / sqrt(2) * [1.0, 0.0, 1.0]
Random.seed!(42)
data = [exp(M, x, random_tangent(M, x, Val(:Gaussian), σ)) for i in 1:n]

F(M, y) = sum(1 / (2 * n) * distance.(Ref(M), Ref(y), data) .^ 2)


## solve with defined initialization
xMean1 = NelderMead(M, F, data[1:(dim+1)])
xMean2 = NelderMead(M, F, data[(dim+2):(2*dim+1)])

@test isapprox(xMean1, xMean2, atol = 1e-2)

## solve with random initialization
xMean3 = NelderMead(M, F, data[1:(dim+1)])
xMean4 = NelderMead(M, F, data[1:(dim+1)])
xMean5 = NelderMead(M, F, data[1:(dim+1)])

@test isapprox(xMean1, xMean3, atol = 1e-2)
@test isapprox(xMean1, xMean4, atol = 1e-2)
@test isapprox(xMean1, xMean5, atol = 1e-2)

@test isapprox(xMean2, xMean3, atol = 1e-2)
@test isapprox(xMean2, xMean4, atol = 1e-2)
@test isapprox(xMean2, xMean5, atol = 1e-2)

@test isapprox(xMean3, xMean4, atol = 1e-2)
@test isapprox(xMean3, xMean5, atol = 1e-2)

@test isapprox(xMean4, xMean5, atol = 1e-2)

##

end


#
