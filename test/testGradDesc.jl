@testset "gradient descent..." begin
  # explicitly compute an easy exp
	M = Circle()
	r = [-π/2,π/4,0.0,π/4];
  	f = S1Point.(r);
	F(x) = 1/2*sum(distance.(M,f,x).^2);
	gradF(x) = sum(-log.(M,x,f));
	lO = ArmijoLineSearchOptions(f[1]);
	stoppingCrit(i,ξ,x,xnew) = (i>0), (i<1)?"":"Stopped after $(i) iterations"; #one iteration
	dP = GradientProblem(M,F,gradF);
	dO = GradientDescentOptions(f[1],stoppingCrit,ArmijoLineSearch,lO);
	x, = steepestDescent(dP,dO)
	# after one step for local enough data -> equal to real valued data
	@test abs(getValue(x)-sum(r)/length(r)) ≈ 0 atol=10.0^(-16)
end
