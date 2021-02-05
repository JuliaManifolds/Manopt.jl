
using Manifolds, Manopt, BenchmarkTools

include("../test/solvers/trust_region_model.jl")

n=size(A,1)
p=2
N = Grassmann(n,p)
M = PowerManifold(N, NestedPowerRepresentation(),2)
x = random_point(M)

@btime x_opt = trust_regions($M, $cost, $rgrad, $rhess, $x; max_trust_region_radius=8.0)

h = RHess(A,p)
g = RGrad(A,p)
@btime trust_regions($M, $cost, $g, $h, x2; max_trust_region_radius=8.0, evaluation=$(MutatingEvaluation()))  setup = (x2 = deepcopy($x))
