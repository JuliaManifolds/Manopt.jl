#
# Minimizing the Raleigh Quotient
#
# This example illustrates the gradient descent and its capabilities
# minimizin the raleigh Quotient on the sphere to compute an eigenvector 
# corresponding to the smallest eigen value
#
# R. Bergmann, 2019-01-16
using Manopt, LinearAlgebra

A = [8. 3.; 4. 10.]
n = size(A,1)-1
M = Sphere(n)
# Raleigh simplified to unit vectorts 
Rayleigh(x::SnPoint) = transpose(getValue(x))*A*getValue(x)
∇Rayleigh(x::SnPoint) = SnTVector(
    2/dot(getValue(x),getValue(x)) * (A*getValue(x) - Rayleigh(x)*getValue(x))
)

x0 = SnPoint( 1/sqrt(2.)*[ -1. , 1. ] );

eV = steepestDescent(M, Rayleigh, ∇Rayleigh, x0;
    stepsize = ConstantStepsize(π/64),
    stoppingCriterion = stopWhenAny( stopAfterIteration(50), stopWhenGradientNormLess(10.0^-8) ),
    debug= [:Iteration, :Divider, :Iterate, :Divider, :GradientNorm, :Divider, :Stepsize, :Divider, :Cost, :Newline, :stoppingCriterion],
);

eV2 = steepestDescent(M, Rayleigh, ∇Rayleigh, x0;
    stepsize = Armijo(π/64,exp,0.968,1.), # use Armijo lineSearch
    debug=[:Iteration, :Divider, :Iterate, :Divider, :GradientNorm, :Divider, :Stepsize, :Divider, :Cost, :Newline, :stoppingCriterion]
);
v,E = eigen(A)
print(eV, "  (", norm(getValue(eV)-E[:,1]),")   ", eV2, "  (", norm(getValue(eV2)-E[:,1]),") vs. the original ", E[:,1] )