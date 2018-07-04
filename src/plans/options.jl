#
# While a problem consists of all things one has to know about the Optimization
# problem itself (independent of the solver), the options collect parameters,
# that steer the solver (indipendent of the problem at hand)
#
export Options, LineSearchOptions, GradientDescentOptions, DebugOptions
export getGradient, getStepSize, evaluateStoppingCriterion, getVerbosity

abstract type Options end

type LineSearchOptions <: Options
    initialStepsize::Float64
    rho::Float64
    c::Float64
end

type GradientDescentOptions <: Options
    initX::T where T <: MPoint
    stoppingCriterion::Function
    retraction::Function
    lineSearch::Function
    lineSearchProblem::LineSearchProblem
    verbosity::Int
end

# Todo getter/setter should pass down to the inner
type DebugDecoOptions{O<: Options} <: Options
    options::O
    debugFunction::Function
    debugSettings::Dict{String,<:Any}
end
getOptions{O <: Options}(o::O) = O;
getOptions{O <: DebugDecoOptions}(o::O) = O.options;

"""
    evaluateStoppingCriterion(problem,iter,x1,x2)

evaluates the stoppinc criterion of problem with respect to the current
iteration and two (successive) values of the algorithm
"""
function evaluateStoppingCriterion{P<:Problem, MP <: MPoint,I<:Integer}(p::P,
                          iter::I,x1::MP,x2::MP)
  p.stoppingCriterion(iter,ξ,x1,x2)
end
"""
    evaluateStoppingCriterion(problem,iter,x1,x2)

evaluates the stopping criterion of problem with respect to the current
iteration iter, the descent direction ξ, and two (successive) iterates x1, x2
of the algorithm.
"""
function evaluateStoppingCriterion{O<:GradientDescentOptions, MP <: MPoint, MT <: TVector, I<:Integer}(o::O,
                          iter::I,ξ::MT, x1::MP, x2::MP)
  o.stoppingCriterion(iter,ξ,x1,x2)
end

"""
    getVerbosity(problem)

returns the verbosity of the problem.
"""
function getVerbosity{O<:Options}(o::O)
  getOptions(o).verbosity
end
