
using Manifolds, Manopt, BenchmarkTools, Test

include("../test/solvers/trust_region_model.jl")

n = size(A, 1)
p = 2
N = Grassmann(n, p)
M = PowerManifold(N, ArrayPowerRepresentation(), 2)
x = random_point(M)

x_opt = trust_regions(M, cost, rgrad, rhess, x; max_trust_region_radius=8.0)
@btime x_opt = trust_regions($M, $cost, $rgrad, $rhess, $x; max_trust_region_radius=8.0)

h = RHess(M, A, p)
g = RGrad(M, A)
x_opt2 = trust_regions(
    M, cost, g, h, x; max_trust_region_radius=8.0, evaluation=MutatingEvaluation()
)

@btime trust_regions(
    $M, $cost, $g, $h, x2; max_trust_region_radius=8.0, evaluation=$(MutatingEvaluation())
) setup = (x2 = deepcopy($x))

x3 = deepcopy(x)
trust_regions!(
    M, cost, g, h, x3; max_trust_region_radius=8.0, evaluation=MutatingEvaluation()
)

@btime trust_regions!(
    $M, $cost, $g, $h, x3; max_trust_region_radius=8.0, evaluation=$(MutatingEvaluation())
) setup = (x3 = deepcopy($x))

@test distance(M, x_opt, x_opt2) ≈ 0
@test distance(M, x_opt, x3) ≈ 0
