using Pkg;
#cd(@__DIR__)
#Pkg.activate("."); # for reproducibility use the local tutorial environment.
#Pkg.develop(PackageSpec(; path=(@__DIR__) * "/../"))

using Manopt, Manifolds, Random, LinearAlgebra, LRUCache
Random.seed!(23)

n=512
p=12
A_init=randn(n,n)
A=(A_init+A_init')/2
M=Grassmann(n,p)

function f(M,p)
	return -0.5*tr(p'*A*p)
end
function grad_f(M,p)
	return -A*p+p*(p'*A*p)
end
function Hess_f(M,p,X)
	return -A*X +p*p'*A*X+X*p'*A*p
end

p0 = rand(M)
opt3=adaptive_regularization_with_cubics!(
    M,f, grad_f, Hess_f, p0;
    σ=28.86751,
    retraction_method=PolarRetraction(),
    debug=[:Iteration,:Cost,:ρ, " ", DebugGradientNorm(),"\n", :Stop],
    count=[:Cost,:Gradient, :Hessian],
    cache=(:LRU, [:Cost, :Gradient, :Hessian], 25),
    return_objective=true,
    #return_state=true
)