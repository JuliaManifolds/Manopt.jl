import Base: stdout
export DebugOptions, getOptions, debug
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
still be accessed using the [`getOptions`](@ref) function.

The amount of output the `debugFunction` provides can be determined by the
`verbosity`, which should follow the following rough categories, where the
higher level always includes all levels below in output

* 1 - starts and results (low)
* 2 - not yet used
* 3 - End criteria of algorithms etc.
* 4 - Time measurements
* 5 - Iteration interims values

# Fields (defaults in brackets)
* `options` – the options that are extended by debug information
* `debugActivated` – a set of Symbols whose debug is activated
* `debugEvery` (`1`) - reduce debug output to be only performed every `k`th iteration
* `verbosity` - (`3`) a verbosity, see above.
* `debugOut` – (`Base.stdout`) where to produce debug to, e.g. a file.
"""
mutable struct DebugOptions{O<:Options} <: Options
    options::O
    debugActivated::Array{Symbol,1}
    debugEvery::Int
    verbosity::Int
    debugOut::IO
    DebugOptions{O}(o::O, dA::Array{Symbol,1}, dbgE::Int=1,dbgV::Int=5,dOut::IO=Base.stdout) where {O <: Options} = new(o,dA,dbgE,dbgV,dOut)
end
DebugOptions(o::O, dA::Array{Symbol,1},e=1,v=3, dOut::IO=Base.stdout) where {O <: Options} = DebugOptions{O}(o,dA,e,v,dOut)

@traitimpl IsOptionsDecorator{DebugOptions}

@doc doc"""
    debug(p,o,s[, status,iter=0])

perform debug for `Symbol s` during run of the solver with respect to
[`Problem`](@ref)` p` and [`DebugOptions`](@ref)` o`, that decorate the
original options. While some symbols might require an additional status String,
depending on the specific `Symbol`, during the iterations most should just rely
on values stored within the [`Options`](@ref)` o`.
The optional `status` String might be used to pass down a String directly.
The iteration may be used to
compute certain values, but is mainly used to reduce debug to `debugEvery`th iteration.
"""
function debug(p::P,dO::DebugOptions{O},debugFor::Symbol,iter::Int=0) where {P <: Problem, O <: Options}
    if (rem(iter,dO.debugEvery)==0) && ( debugFor ∈ dO.debugActivated )
        debug(p,dO.options,Val(debugFor),iter,dO.debugOut)
    end
end
function debug(p::P,dO::DebugOptions{O},debugFor::Symbol,status::String,iter::Int=0) where {P <: Problem, O <: Options}
    if (rem(iter,dO.debugEvery)==0) && ( debugFor ∈ dO.debugActivated )
        debug(p,dO.options,Val(debugFor),status,iter,dO.debugOut)
    end
end#
# general debugs - defaults
#
# overwrite if you do something different and do not store an iterate in x.
@doc doc"""
    debug(p,o,v,iter,[out=Base.stdout])

implement the specific output a certain symbol generated, based on a
[`Problem`](@ref)` p`, [`Options`](@ref)` o` and certain `Symbol`ic value `v`, as
well as the current `iter`ation. The optional parameter `out` classically prints
to the default out, but can also be used to print the log to a file, when set to
somethins different in the external interface, the [`DebugOptions`](@ref).

In order to generate your own debug output, you have to Choose a `:Symbol`,
that for a certain [`Problem`](@ref)` p`, [`Options`](@ref)` o` and implement
this function. Then add this symbol to the [`DebugOptions`](@ref) initialization.
"""
debug(p::P, o::O, ::Val{:Change}, iter::Int, out::IO=Base.stdout) where {P <: Problem, O <: Options} = print(out,"Change: ",distance(p.M, o.xOld, o.x) )
debug(p::P, o::O, ::Val{:Cost}, iter::Int, out::IO=Base.stdout)  where {P <: Problem, O <: Options} = print(out,"Cost Function: ", getCost(p,o.x))
debug(p::P, o::O, ::Val{:InitialCost}, iter::Int, out::IO=Base.stdout)  where {P <: Problem, O <: Options} = println(out,"Initial Cost Function: ", getCost(p,o.x))
debug(p::P, o::O, ::Val{:FinalCost}, iter::Int, out::IO=Base.stdout)  where {P <: Problem, O <: Options} = println(out,"Final Cost Function after $(iter) iterations: ", getCost(p,o.x))
debug(p::P, o::O, ::Val{:Divider}, iter::Int, out::IO=Base.stdout) where {P <: Problem, O <: Options} = print(out," | ")
debug(p::P, o::O, ::Val{:Iteration},iter::Int,out::IO=Base.stdout) where {P <: Problem, O <: Options} = print(out," #$(iter)")
debug(p::P, o::O, ::Val{:Iterate},iter::Int,out::IO=Base.stdout) where {P <: Problem, O <: Options} = print(out," Iterate: $(o.x)")
debug(p::P, o::O, ::Val{:Newline}, iter::Int, out::IO=Base.stdout) where {P <: Problem, O <: Options} = print(out,"\n")
debug(p::P, o::O, ::Val{:Solver}, status::String, iter::Int, out::IO=Base.stdout) where {P <: Problem, O <: Options} = print(out,status)
debug(p::P, o::O, ::Val{:stoppingCriterion}, status::String, iter::Int, out::IO=Base.stdout) where {P <: Problem, O <: Options} = print(out,status)
# Fallback
debug(p::P,o::O,s::Val{e},r...) where {e, P <: Problem, O <: Options} =
  @warn(string("No debug Symbol $(s) within $(typeof(p)) and $(typeof(o)) provided yet."))
