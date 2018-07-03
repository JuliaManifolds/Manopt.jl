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
function ArmijoLineSearch{Mc<:Manifold, MP <: MPoint, MT <: MTVector}(problem::LineSearchProblem{Mc},
    x::MP,gradFx::MT,descentDirection::MT, retraction::Function=exp)::Float64
  e = problem.costFunction(x)
  eNew = e-1
  # for local shortness
  F = problem.costFunction
  s = problem.initialStepsize
  M = problem.manifold
  ξ = descentDirection
  ν = gradFx
  ρ = problem.rho
  c = problem.c
  while e > eNew
    xNew = exp(M,x,s*ξ)
    eNew = F(xNew) - c*s*dot(M,x,ξ,ν)
    if e >= eNew
      s = s/ρ
    end
  end
  while (e < eNew) && (s > typemin(Float64))
    s = s*ρ
    xNew = exp(M,x,s*ξ)
    eNew = F(xNew) - c*s*dot(M,x,ξ,ν)
  end
  return s
end
ArmijoLineSearch{Mc<:Manifold, MP <: MPoint, MT <: MTVector}(problem::LineSearchProblem{Mc},x::MP, gradFx::MT, retraction::Function=exp) = ArmijoLineSearch(problem, x, gradFx, -gradFx,retraction)
