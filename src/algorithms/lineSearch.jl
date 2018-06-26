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

compute the step size with respect to Armijo Rule for a LineSearchProblem, i.e.
for a manifold M (.Manifold), a cost function (.costFunction), point x on M,
a gradient (.Gradient) at x and
optionally a descent direction (.DescentDirection=-.Gradient) and a decreaseFactor
(.Rho) as well as a scalar c for the gain (.c) on front of the inner product.
"""
function ArmijoLineSearch{Mc<:Manifold, MP <: MPoint, MT <: MTVector}(problem::LineSearchProblem{Mc},
    x::MP,gradFx::MT,descentDirection::MT)::Float64
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
ArmijoLineSearch{MP <: MPoint, MT <: MTVector}(problem::LineSearchProblem,
                x::MP, gradFx::MT ) = ArmijoLineSearch(problem, x, gradFx, -gradFx)
