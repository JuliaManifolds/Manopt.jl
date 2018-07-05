#
# Manopt.jl – line Search
#
# methods related to line search
#
# ---
# Manopt.jl – Ronny Bergmann – 2018-06-25
export ArmijoLineSearch
"""
  ArmijoLineSearch(LineSearchProblem)
    compute the step size with respect to Armijo Rule for a LineSearchProblem.

    INPUT
      problem - a LineSearchProblem (with Manifold, costFunction, initialStepSize, c and rho)
      x - current point on M
      gradFx - the gradient of the costFunction
      descentDirection - (optional) a descentDirection, set to -gradFx if not
        explicitly specified
      retraction – (optional) a retraction on M. Set to exp if not given

    OUTPUT
      s - the resulting stepsize
"""
function ArmijoLineSearch{Mc<:Manifold}(problem::GradientProblem{Mc},
    options::DescentLineSearchOptions)::Float64
  # for local shortness
  F = problem.costFunction
  M = problem.M
  x = options.x
  ν = getGradient(problem,x)
  s = options.initialStepsize
  ρ = options.rho
  c = options.c
  retr = options.retraction
  ξ = options.descentDirection
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
ArmijoLineSearch{Mc<:Manifold}(problem::GradientProblem{Mc},
    options::GradientLineSearchOptions)::Float64 = ArmijoLineSearch(problem,
    DescentLineSearchOptions(options,-getGradient(problem, options.x)) )
