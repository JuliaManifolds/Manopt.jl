#
#
#
@testset "Test Nelder-Mead" begin
M = Euclidean(6)
# From Wikipedia https://en.wikipedia.org/wiki/Rosenbrock_function
Rosenbrock(x::RnPoint) = Rosenbrock(getValue(x))
Rosenbrock(x) = sum( [ 100*( x[2*i-1]^2-x[2*i] )^2 + (x[2*i-1]-1)^2 for i=1:div(length(x),2) ] )

x,rec = NelderMead(M,Rosenbrock, record=[RecordCost()])

nonincreasing = [ rec[i] >= rec[i+1] for i=1:length(rec)-1 ]

@test any(map(!,nonincreasing)) == false
end