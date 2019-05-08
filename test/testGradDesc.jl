@testset "gradient descent..." begin
  # Test the gradient descent with
  # the distance function squared
  # on S1, such that we can easily also verify exp and log
  M = Circle()
  r = [-π/2,π/4,0.0,π/4];
  f = S1Point.(r);
  F(x) = 1/10*sum(distance.(Ref(M),f,Ref(x)).^2)
  ∇F(x) = 1/5*sum(-log.(Ref(M),Ref(x),f))
  x = steepestDescent(M,F,∇F,f[1];
    stoppingCriterion = stopWhenAny(stopAtIteration(200), stopChangeLess(10^-16)),
    stepsize = Armijo(1.,exp,0.99,0.1),
  )
  # after one step for local enough data -> equal to real valued data
  @test abs(getValue(x)-sum(r)/length(r)) ≈ 0 atol=10.0^(-15)
end
