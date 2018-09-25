export evaluateStoppingCriterion, getTrustRadius, updateTrustRadius, getVerbosity
export DebugOptions

#
#
# Debug Decorator
#
#
mutable struct DebugOptions <: Options
    options::O where {O<: Options}
    debugFunction::Function
    debugOptions::Dict{String,<:Any}
    verbosity::Int
end
evaluateStoppingCriterion(o::DebugOptions,vars...) = evaluateStoppingCriterion(o.options,vars...)
getOptions(o::O) where {O <: Options} = o; # fallback and end
getOptions(o::O) where {O <: DebugOptions} = getOptions(o.options); #unpeel recursively

function setDebugFunction!(o::O,f::Function) where {O<:Options}
    if getOptions(o) != o #decorator
        setDebugFunction!(o.options,f)
    end
end
function setDebugFunction!(o::O,f::Function) where {O<:DebugOptions}
    o.debugFunction = f;
end
function getDebugFunction(o::O) where {O<:Options}
    if getOptions(o) != o #We have a decorator
        return getDebugFunction(o.options,f)
    end
end
function getDebugFunction(o::O,f::Function) where {O<:DebugOptions}
    return o.debugFunction;
end
function setDebugOptions!(o::O,dO::Dict{String,<:Any}) where {O<:Options}
    if getOptions(o) != o #decorator
        setDebugOptions(o.options,dO)
    end
end
function setDebugOptions!(o::O,dO::Dict{String,<:Any}) where {O<:DebugOptions}
    o.debugOptions = dO;
end
function getDebugOptions(o::O) where {O<:Options}
    if getOptions(o) != o #decorator
        return getDebugOptions(o.options)
    end
end
function getDebugOptions(o::O,dO::Dict{String,<:Any}) where {O<:DebugOptions}
    return o.debugOptions;
end
function optionsHasDebug(o::O) where {O<:Options}
    if getOptions(o) == o
        return false;
    else
        return optionsHaveDebug(o.options)
    end
end
optionsHasDebug(o::O) where {O<:DebugOptions} = true

"""
    getVerbosity(Options)

returns the verbosity of the options, if any decorator provides such, otherwise 0
    if more than one decorator has a verbosity, the maximum is returned
"""
function getVerbosity(o::O) where {O<:Options}
  if getOptions(o) == o # no Decorator
      return 0
  end
  # else look into encapsulated
  return getVerbosity(getOptions(o))
end
# List here any decorator that has verbosity
function getVerbosity(o::O) where {O<:DebugOptions}
  if o.options == getOptions(o.options) # we do not have any further inner decos
      return o.verbosity;
  end
  # else maximum of all decorators
    return max(o.verbosity,getVerbosity(o.options));
end
