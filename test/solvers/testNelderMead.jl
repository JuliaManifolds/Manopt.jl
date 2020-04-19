#
#
#
@testset "Test Nelder-Mead" begin
M = Euclidean(6)
# From Wikipedia https://en.wikipedia.org/wiki/Rosenbrock_function
function Rosenbrock(x)
    sum( [ 100*( x[2*i-1]^2-x[2*i] )^2 + (x[2*i-1]-1)^2 for i=1:div(length(x),2) ] )
end
x0 = [ randn(6) for i=1:7 ]
o = NelderMead(M, Rosenbrock, x0; record=[RecordCost()], returnOptions=true)

x = get_solver_result(o)
rec = get_record(o)

x2 = o = NelderMead(M, Rosenbrock, x0)

@test x==x2

nonincreasing = [ rec[i] >= rec[i+1] for i=1:length(rec)-1 ]

@test any(map(!,nonincreasing)) == false
end