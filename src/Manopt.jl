"""
`Manopt.jl` – Optimization on Manifolds in Julia.
"""
module Manopt
  using SimpleTraits
  using Markdown
  using ManifoldsBase


  # ...corresponding plans consisting of problems and options
  include("plans/plan.jl")
  # Functions
  include("functions/adjointDifferentials.jl")
  include("functions/costFunctions.jl")
  include("functions/differentials.jl")
  include("functions/gradients.jl")
  include("functions/jacobiFields.jl")
  include("functions/proximalMaps.jl")
  # ...solvers (1) general framework
  include("solvers/solver.jl")
  # ...solvers (2) specific solvers
  include("solvers/cyclicProximalPoint.jl")
  include("solvers/DouglasRachford.jl")
  include("solvers/NelderMead.jl")
  include("solvers/steepestDescent.jl")
  include("solvers/truncatedConjugateGradient.jl")
  include("solvers/trustRegions.jl")
  include("solvers/subGradientMethod.jl")
  # extended metasolvers
  include("solvers/debugSolver.jl")
  include("solvers/recordSolver.jl")
  # algorithms
  include("algorithms/basicAlgorithms.jl")
  # Plots
  include("plots/SpherePlots.jl")
  # helpers
  include("helpers/errorMeasures.jl")
  # Exports
  include("helpers/exports/Asymptote.jl")
  # data
  include("data/artificialDataFunctions.jl")
end
