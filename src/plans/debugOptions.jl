export evaluateStoppingCriterion,getVerbosity,updateDebugValues,Debug
export setDebugFunction,getDebugFunction,setDebugOptions,getDebugOptions
export DebugOptions

#
#
# Debug Decorator
#
#
@doc doc"""
    DebugOptions <: Options

The debug options append to any options a debug functionality, i.e. they act as
a decorator pattern. The Debug keeps track of a dictionary of values and only
these are kept up to date during e.g. the iterations. The original options can
still be accessed using the `getOptions` function.

The amount of output the `debugFunction` provides can be determined by the
`verbosity`, which should follow the following rough categories, where the
higher level always includes all levels below in output

* 1 - starts and results (low)
* 2 - not yet used
* 3 - End criteria of algorithms etc.
* 4 - Time measurements
* 5 - Iteration interims values

# Fields
* `options` – the options that are extended by debug information
* `debugFunction` – a function called to produce debug output, e.g. on REPL and
  gets a dictionary of values (see `DebugValues` as input)
* `debugValues` – a dictionary of values to store, where the strings to store
  values with is determined by the algorithm itself
* `verbosity` – A verbosity level
"""
mutable struct DebugOptions <: Options
    options::O where {O<: Options}
    debugFunction::Function
    debugValues::Dict{String,<:Any}
    verbosity::Int
end
evaluateStoppingCriterion(o::DebugOptions,vars...) = evaluateStoppingCriterion(o.options,vars...)
getOptions(o::O) where {O <: Options} = o; # fallback and end
getOptions(o::O) where {O <: DebugOptions} = getOptions(o.options); #unpeel recursively

"""
    Debug(o)

perform debug output for options `o`.
"""
function Debug(o::O) where {O<:DebugOptions}
    o.debugFunction(o.debugValues)
end
@doc doc"""
    setDebugFunction!(o,f)

set the debug function within the options `o` to `f`.
"""
function setDebugFunction!(o::O,f::Function) where {O<:Options}
    if getOptions(o) != o #decorator
        setDebugFunction!(o.options,f)
    end
end
function setDebugFunction!(o::O,f::Function) where {O<:DebugOptions}
    o.debugFunction = f;
end
@doc doc"""
    getDebugFunction(o)

get the debug function within the options `o`.
"""
function getDebugFunction(o::O) where {O<:Options}
    if getOptions(o) != o #We have a decorator
        return getDebugFunction(o.options,f)
    end
end
function getDebugFunction(o::O,f::Function) where {O<:DebugOptions}
    return o.debugFunction;
end
@doc doc"""
    setDebugValues(o,v)

set the dictionary of debug values to `v`, i.e. especially also update the
currently present keys within the debug values.
"""
function setDebugOptions!(o::O,v::Dict{String,<:Any}) where {O<:Options}
    if getOptions(o) != o #decorator
        setDebugOptions(o.options,v)
    end
end
function setDebugOptions!(o::O,v::Dict{String,<:Any}) where {O<:DebugOptions}
    o.debugValues = v;
end
@doc doc"""
    setDebugValues(o,v)

set the dictionary of debug values to `v`, i.e. especially also update the
currently present keys within the debug values.
"""
function getDebugOptions(o::O) where {O<:Options}
    if getOptions(o) != o #decorator
        return getDebugOptions(o.options)
    end
end
function getDebugOptions(o::O,dO::Dict{String,<:Any}) where {O<:DebugOptions}
    return o.debugValues;
end
@doc doc"""
    optionsHasDebug(o)

returns true if the give options `o` are decorated with debug options.
"""
function optionsHasDebug(o::O) where {O<:Options}
    if getOptions(o) == o
        return false;
    else
        return optionsHaveDebug(o.options)
    end
end
optionsHasDebug(o::O) where {O<:DebugOptions} = true

"""
    updateDebugValues!(o,v)
update all values in the debug options `o` for which `v` has a new value.
"""
function updateDebugValues!(o::O,v::Dict{String,<:Any}) where {O<:DebugOptions}
    for k in keys(v)
        if haskey(o.debugValues,k) # key presend -> update
            o.debugValues[k] = v[k]
        end
    end
end
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
