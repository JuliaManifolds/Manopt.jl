@testset "Test the 2-by-2 symmetric positive definite matrices" begin
  # explicitly compute an easy exp
	M = Circle(2)
	r = [-π/2,π/4,0,π/4];
  	p = S1Point.(r);
	F(p) = 1/2*sum(distance.(M,pts,p).^2);
	gradF(p) = sum(-log.(M,p,pts));
	lP = LineSearchProblem(M,F);
	stoppingCrit(i,ξ,x,xnew) = (i>0); #one iteration
	dP = DescentProblem(M,F,gradF,stoppingCrit,pts[1],ArmijoLineSearch,lP);
	x = steepestDescent(dP)
	# after one step for local enough data -> equal to real valued data
	@test abs(x-sum(r)/length(r) ≈ 0 atol=10.0^(-16)
end
