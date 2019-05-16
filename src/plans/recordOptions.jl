export RecordOptions
export record!, record, getRecord, getLastRecord, recordType, hasRecord
#
#
# record Options Decorator
#
#
@doc doc"""
    RecordOptions <: Options

The record options append to any options a record functionality, i.e. they act
as a decorator pattern. The record, similar to [`DebugOptions`](@ref), keeps
track of a dictionary of values and only these are recorded during iterations.
The original options can still be accessed using the [`getOptions`](@ref) function.

The amount of data the `recordOptions` store can be determined by specifying
fields that should be recorded, adressed by symbols within `recordKeys`. for
any such symbol two functions have to be implemented, here illustrated for `:Iterate`
- [`record`](@ref)`(p::P,o::O,::Val{:Iterate},iter) where {P <: `[`Problem`](@ref)`, O <: `[`Options`](@ref)`} = `[`getOptions`](@ref)(o).x`
- [`recordType`](@ref)`(o::O, ::Val{:Iterate}) where {P <: `[`Problem`](@ref)`, O <: `[`Options`](@ref)} = typeof(`[`getOptions`](@ref)`(o).x)

the first one is called every iteration and returns the value to be recorded,
the second one provides the `DataType` of the recorded value in order to initialize
the array of records. You can hence record anything you store between iterations
(i.e. in the `Options` of an algorithm.)

# Fields
* `options` – the options that are extended by debug information
* `recordKeys` - a tuple of `Symbols` which values to store.
* `recordedValues` – an array of `Tuple` where the entries of that tuple are the `typeof`s,
belonging to the `recordedKeys` (provided by [`recordType`](@ref))

# Constructor
`recordOptions(o,r)`, where `o` are the options to be decorated and `r` is the
`NTuple` of `Symbols` what to record.
"""
mutable struct RecordOptions{T,N} <: Options where {T,N}
  options::O where {O<: Options}
  recordKeys::NTuple{N,Symbol}
  recordedValues::Array{T,1}
  RecordOptions{T,N}(
    o::Op,
    r::NTuple{N,Symbol},
  ) where {Op <: Options,N,T} = 
    new(o,r,Array{T,1}(undef,0))
end
RecordOptions(o::Op, r::NTuple{N,Symbol}) where {Op <: Options,N} = 
  RecordOptions{Tuple{[ recordType(getOptions(o),Val(v)) for v in r]...},N}(o,r)

@traitimpl IsOptionsDecorator{RecordOptions}

"""
    record!(p,o,iter)

perform one record for the `iter`th iteration of the solver for
[`Problem`](@ref)` p` and [`RecordOptions`](@ref)` o`, where the latter contains
the current state after that iteration internally.
"""
function record!(p::P, oR::RecordOptions, iter::Int) where {P <: Problem}
  values = Tuple(
    [record(p,getOptions(oR.options),Val(recordKey),iter)
      for recordKey in oR.recordKeys]
  )
  push!(oR.recordedValues,values)
end
doc"""
    getLastRecord(o,key)

returns the last recorded data item with key `key` from the records array of the
[`RecordOptions`](@ref)` o`, i.e. the final result with all its (recorded)
metadata.
"""
function getLastRecord(o::RecordOptions, key::Symbol)
  if key in o.recordKeys
    pos = findall(x->x==key, o.recordKeys)[1] # find frst position
    return o.recordedValues[end][pos]
  else
    throw(
      ErrorException("The key \"$(key)\" is not among the recorded keys of these Record Options.")
    )
  end
end
@traitfn getLastRecord(o::O,k) where {O <: Options; IsOptionsDecorator{O}} = getLastRecord(o.options,k)
@traitfn getLastRecord(o::O,k) where {O <: Options; !IsOptionsDecorator{O}} = throw( ErrorException(" None of the decorators is a options Recorder"))

function getRecord(o::RecordOptions, key::Symbol,iter)
  if iter > length(o.recordedValues)
    throw(
      ErrorException("No iterate $(iter) existst in therse records.")
    )
  end
  if key in o.recordKeys
     pos = findall(x->x==key, o.recordKeys)[1] # find frst position
    return o.recordedValues[iter][pos]
  else
    throw(
      ErrorException("The key \"$(key)\" is not among the recorded keys of these Record Options.")
    )
  end
end
@traitfn getRecord(o::O,k,iter) where {O <: Options; IsOptionsDecorator{O}} = getRecord(o.options,k,iter)
@traitfn getRecord(o::O,k,iter) where {O <: Options; !IsOptionsDecorator{O}} = throw( ErrorException(" None of the decorators is a options Recorder"))

doc"""
    getRecord(o,key)
get the record of the value recorded during the iterations under the key `key`
in the [`RecordOptions`](@ref)` o`.
"""
function getRecord(o::RecordOptions, key::Symbol)
  if key in o.recordKeys
    pos = findall(x->x==key, o.recordKeys)[1] # find frst position
    return [ x[pos] for x in o.recordedValues ]
  else
    throw(
      ErrorException("The key \"$(key)\" is not among the recorded keys of these Record Options.")
    )
  end
end
@traitfn getRecord(o::O, key::Symbol) where {O <: Options; IsOptionsDecorator{O}} = getRecord(o.options, key::Symbol)
@traitfn getRecord(o::O, key::Symbol) where {O <: Options; !IsOptionsDecorator{O}} = throw(ErrorException(
    "No record decorator found within the decorators of $(typeof(o))."
))

"""
    getRecord(o)

Return the array of tuples with the recorded values within the [`Options`]` o`,
if one of its decorators is a [`RecordOptions`](@ref).
"""
getRecord(o::RecordOptions) = o.recordedValues
# decorated ones: recursive search
@traitfn getRecord(o::O) where {O <: Options; IsOptionsDecorator{O}} = getRecord(o.options)
@traitfn getRecord(o::O) where {O <: Options; !IsOptionsDecorator{O}} = throw(ErrorException(
    "No record decorator found within the decorators of $(typeof(o))."
))
"""
    hasRecord(o)

check whether the [`Options`](@ref)` o` are decorated with
[`RecordOptions`](@ref)
"""
hasRecord(o::RecordOptions) = true
@traitfn hasRecord(o::O) where {O <: Options; IsOptionsDecorator{O}} = hasRecord(o.options)
@traitfn hasRecord(o::O) where {O <: Options; !IsOptionsDecorator{O}} = false


#
# provide a few simple defaults (as long as problem and options keep to use
# general naming scheme, i.e. this does not hold for primalDual)
record(p::P,o::O,::Val{:Iteration},iter::Int) where {P <: Problem, O <: Options} = iter
recordType(o::O, ::Val{:Iteration}) where {P <: Problem, O <: Options} = Int

record(p::P,o::O,::Val{:Iterate},iter::Int) where {P <: Problem, O <: Options} = o.x
recordType(o::O, ::Val{:Iterate}) where {P <: Problem, O <: Options} = typeof(o.x)

record(p::P,o::O,::Val{:Change},iter::Int) where {P <: Problem, O <: Options} = distance(p.M,o.x,o.xOld)
recordType(o::O, ::Val{:Change}) where {P <: Problem, O <: Options} = Float64

record(p::P, o::O,::Val{:Cost}, iter::Int) where {P <: Problem, O <: Options} = getCost(p,o.x)
recordType(o::O, ::Val{:Cost}) where {O <: Options} = Float64

#and a fallback tha also explains the function with a documentation
@doc doc"""
    record(p,o,s,iter)
record one data item during the [`doSolverStep!`](@ref)`r` of the [`Problem`](@ref)` p`
with current state stored in [`Options`](@ref)` o` after iteration `iter`.
The `Symbol s` determines which datum to pass to the [`record!`](@ref) function.

It for a `Symbol s` there is no `record` specified, this function will issue
a corresponding warning.

See also [`recordType`](@ref)
"""
record(p::P,o::O,s::Val{e},iter) where {P <: Problem, O <: Options, e} =
  @warn(string("No record function for Symbol $(s) within $(typeof(p)) and $(typeof(o)) provided yet."))

@doc doc"""
  recordType(o,s)
provides the `DataType` a [`record`](@ref) for these [`Options`](@ref)` o` and
`Symbol s` returns in order to properly create the `Tuple` of values recorded
each iteration as well as the `Array` they are stored in.

See also [`record`](@ref)
"""
recordType(o::O,s::Val{e}) where {O <: Options, e} =
  @warn(string("No recordType provided for Symbol $(s) within $(typeof(o))."))