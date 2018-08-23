#
# Manopt.jl – line Search
#
# methods related to line search
#
# ---
# Manopt.jl – Ronny Bergmann – 2018-06-25
export ArmijoLineSearch
"""
  ArmijoLineSearch(GradientProblem, ArmijoLineSearchOptions)
    compute the step size with respect to Armijo's rule for a `GradientProblem`

    INPUT
      problem - a GradientProblem (with Manifold, costFunction and a gradient)
      x - current point on M
      ∇Fx - the gradient of the costFunction
      descentDirection - (optional) a descentDirection, set to -∇Fx if not
        explicitly specified
      retraction – (optional) a retraction on M. Set to exp if not given

    OUTPUT
      s - the resulting stepsize
"""
function ArmijoLineSearch(problem::GradientProblem{Mc},
    options::ArmijoLineSearchOptions)::Float64 where {Mc<:Manifold}
  # for local shortness
  F = problem.costFunction
  M = problem.M
  x = options.x
  ν = getGradient(problem,x)
  s = options.initialStepsize
  ρ = options.ρ
  c = options.c
  retr = options.retraction
  if ismissing(options.direction)
    ξ = -ν
  else
    ξ = options.direction
  end
  e = F(x)
  eNew = e-1

  while e > eNew
    xNew = retr(M,x,s*ξ)
    eNew = F(xNew) - c*s*dot(M,x,ξ,ν)
    if e >= eNew
      s = s/ρ
    end
  end
  while (e < eNew) && (s > typemin(Float64))
    s = s*ρ
    xNew = retr(M,x,s*ξ)
    eNew = F(xNew) - c*s*dot(M,x,ξ,ν)
  end
  return s
end
