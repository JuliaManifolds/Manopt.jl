@testset "Manopt Gradient Descent" begin
  # Test the gradient descent with
  # the distance function squared
  # on S1, such that we can easily also verify exp and log
  M = Circle()
  r = [-π/2,π/4,0.0,π/4];
  f = S1Point.(r);
  F(x) = 1/10*sum(distance.(Ref(M),f,Ref(x)).^2)
  ∇F(x) = 1/5*sum(-log.(Ref(M),Ref(x),f))
  x,rec = steepestDescent(M,F,∇F,f[1];
    stoppingCriterion = stopWhenAny(stopAfterIteration(200), stopWhenChangeLess(10^-16)),
    stepsize = ArmijoLinesearch(1.,exp,0.99,0.1),
    debug = [:Iteration," ",:Cost, :Stop, 100,"\n"],
    record = [:Iteration, :Cost, 1]
  )
  # after one step for local enough data -> equal to real valued data
  @test abs(getValue(x)-sum(r)/length(r)) ≈ 0 atol=5*10.0^(-14)
  # Test Fallbacks -> we can't do steps with the wrong combination
  p = SubGradientProblem(M,F,∇F)
  o = GradientDescentOptions(f[1],stopAfterIteration(20),ConstantStepsize(1.))
  @test_throws ErrorException initializeSolver!(p,o)
  @test_throws ErrorException doSolverStep!(p,o,1)
  @test_throws ErrorException getSolverResult(p,o)
end
