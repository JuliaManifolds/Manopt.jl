#
# Plans included
#
# Fallbacks / General
export getVerbosity, getOptions, setDebugFunction, setDebugOptions
include("problem.jl")
include("options.jl")
include("gradientPlan.jl")
include("proximalPlan.jl")
include("subGradientPlan.jl")

include("debugOptions.jl")
