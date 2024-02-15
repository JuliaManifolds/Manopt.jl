# How to print debug output
Ronny Bergmann

This tutorial aims to illustrate how to perform debug output. For that we consider an
example that includes a subsolver, to also consider their debug capabilities.

The problem itself is hence not the main focus.

We consider a nonnegative PCA which we can write as a constraint problem on the Sphere

Let’s first load the necessary packages.

``` julia
using Manopt, Manifolds, Random, LinearAlgebra
Random.seed!(42);
```

``` julia
d = 4
M = Sphere(d - 1)
v0 = project(M, [ones(2)..., zeros(d - 2)...])
Z = v0 * v0'
#Cost and gradient
f(M, p) = -tr(transpose(p) * Z * p) / 2
grad_f(M, p) = project(M, p, -transpose.(Z) * p / 2 - Z * p / 2)
# Constraints
g(M, p) = -p # now p ≥ 0
mI = -Matrix{Float64}(I, d, d)
# Vector of gradients of the constraint components
grad_g(M, p) = [project(M, p, mI[:, i]) for i in 1:d]
```

Then we can take a starting point

``` julia
p0 = project(M, [ones(2)..., zeros(d - 3)..., 0.1])
```

## Simple debug output

Any solver accepts the keyword `debug=`, which in the simplest case can be set to an array of strings, symbols and a number.

- Strings are printed in every iteration as is (cf. [`DebugDivider`](@ref)) and should be used to finish the array with a line break.
- the last number in the array is used with [`DebugEvery`](@ref) to print the debug only every $i$th iteration.
- Any Symbol is converted into certain debug prints

Certain symbols starting with a capital letter are mapped to certain prints, for example `:Cost` is mapped to [`DebugCost`](@ref)`()` to print the current cost function value. A full list is provided in the [`DebugActionFactory`](@ref).
A special keyword is `:Stop`, which is only added to the final debug hook to print the stopping criterion.

Any symbol with a small letter is mapped to fields of the [`AbstractManoptSolverState`](@ref) which is used. This way you can easily print internal data, if you know their names.

Let’s look at an example first: if we want to print the current iteration number, the current cost function value as well as the value `ϵ` from the [`ExactPenaltyMethodState`](@ref). To keep the amount of print at a reasonable level, we want to only print the debug every twentyfifth iteration.

Then we can write

``` julia
p1 = exact_penalty_method(
    M, f, grad_f, p0; g=g, grad_g=grad_g,
    debug = [:Iteration, :Cost, " | ", :ϵ, 25, "\n", :Stop]
);
```

    Initial f(x): -0.497512 | ϵ: 0.001


    # 25    

    f(x): -0.499449 | ϵ: 0.0001778279410038921
    # 50    f(x): -0.499995 | ϵ: 3.1622776601683734e-5
    # 75    f(x): -0.500000 | ϵ: 5.623413251903474e-6
    # 100   f(x): -0.500000 | ϵ: 1.0e-6
    The value of the variable (ϵ) is smaller than or equal to its threshold (1.0e-6).
    The algorithm performed a step with a change (6.5347623783315016e-9) less than 1.0e-6.

## Advanced debug output

There is two more advanced variants that can be used. The first is a tuple of a symbol and a string, where the string is used as the format print, that most [`DebugAction`](@ref)s have. The second is, to directly provide a `DebugAction`.

We can for example change the way the `:ϵ` is printed by adding a format string
and use [`DebugCost`](@ref)`()` which is equivalent to using `:Cost`.
Especially with the format change, the lines are more consistent in length.

``` julia
p2 = exact_penalty_method(
    M, f, grad_f, p0; g=g, grad_g=grad_g,
    debug = [:Iteration, DebugCost(), (:ϵ," | ϵ: %.8f"), 25, "\n", :Stop]
);
```

    Initial f(x): -0.497512 | ϵ: 0.00100000
    # 25    f(x): -0.499449 | ϵ: 0.00017783
    # 50    f(x): -0.499995 | ϵ: 0.00003162
    # 75    f(x): -0.500000 | ϵ: 0.00000562
    # 100   f(x): -0.500000 | ϵ: 0.00000100
    The value of the variable (ϵ) is smaller than or equal to its threshold (1.0e-6).
    The algorithm performed a step with a change (6.5347623783315016e-9) less than 1.0e-6.

You can also write your own [`DebugAction`](@ref) functor, where the function to implement has the same signature as the `step` function, that is an [`AbstractManoptProblem`](@ref), an [`AbstractManoptSolverState`](@ref), as well as the current iterate. For example the already mentioned[`DebugDivider`](@ref)`(s)` is given as

``` julia
mutable struct DebugDivider{TIO<:IO} <: DebugAction
    io::TIO
    divider::String
    DebugDivider(divider=" | "; io::IO=stdout) = new{typeof(io)}(io, divider)
end
function (d::DebugDivider)(::AbstractManoptProblem, ::AbstractManoptSolverState, i::Int)
    (i >= 0) && (!isempty(d.divider)) && (print(d.io, d.divider))
    return nothing
end
```

or you could implement that of course just for your specific problem or state.

## Subsolver debug

most subsolvers have a `sub_kwargs` keyword, such that you can pass keywords to the sub solver as well. This works well if you do not plan to change the subsolver. If you do you can wrap your own `solver_state=` argument in a [`decorate_state!`](@ref) and pass a `debug=` password to this function call.
Keywords in a keyword have to be passed as pairs (`:debug => [...]`).

A main problem now is, that this debug is issued every sub solver call or initialisation, as the following print of just a `.` per sub solver test/call illustrates

``` julia
p3 = exact_penalty_method(
    M, f, grad_f, p0; g=g, grad_g=grad_g,
    debug = ["\n",:Iteration, DebugCost(), (:ϵ," | ϵ: %.8f"), 25, "\n", :Stop],
    sub_kwargs = [:debug => ["."]]
);
```


    Initial f(x): -0.497512 | ϵ: 0.00100000
    ........................................................
    # 25    f(x): -0.499449 | ϵ: 0.00017783
    ..................................................
    # 50    f(x): -0.499995 | ϵ: 0.00003162
    ..................................................
    # 75    f(x): -0.500000 | ϵ: 0.00000562
    ..................................................
    # 100   f(x): -0.500000 | ϵ: 0.00000100
    ....The value of the variable (ϵ) is smaller than or equal to its threshold (1.0e-6).
    The algorithm performed a step with a change (6.5347623783315016e-9) less than 1.0e-6.

The different lengths of the dotted lines come from the fact that —at least in the beginning— the subsolver performs a few steps.

For this issue, there is the next symbol (similar to the `:Stop`) to indicate that a debug set is a subsolver set `:Subsolver`, which introduces a [`DebugWhenActive`](@ref) that is only activated when the outer debug is actually active, or inother words [`DebugEvery`](@ref) is active itself.
Let’s

``` julia
p4 = exact_penalty_method(
  M, f, grad_f, p0; g=g, grad_g=grad_g,
  debug = [:Iteration, DebugCost(), (:ϵ," | ϵ: %.8f"), 25, "\n", :Stop],
  sub_kwargs = [
    :debug => [" | ", :Iteration, :Cost, "\n",:Stop, :Subsolver]
  ]
);
```

    Initial f(x): -0.497512 | ϵ: 0.00100000
     | Initial f(x): -0.499127
     | # 1     f(x): -0.499147
    The algorithm reached approximately critical point after 1 iterations; the gradient norm (0.00021218898527217448) is less than 0.001.
    # 25    f(x): -0.499449 | ϵ: 0.00017783
     | Initial f(x): -0.499993
     | # 1     f(x): -0.499994
    The algorithm reached approximately critical point after 1 iterations; the gradient norm (1.6025009584517956e-5) is less than 0.001.
    # 50    f(x): -0.499995 | ϵ: 0.00003162
     | Initial f(x): -0.500000
     | # 1     f(x): -0.500000
    The algorithm reached approximately critical point after 1 iterations; the gradient norm (9.966301158134047e-7) is less than 0.001.
    # 75    f(x): -0.500000 | ϵ: 0.00000562
     | Initial f(x): -0.500000
     | # 1     f(x): -0.500000
    The algorithm reached approximately critical point after 1 iterations; the gradient norm (5.4875346930698466e-8) is less than 0.001.
    # 100   f(x): -0.500000 | ϵ: 0.00000100
    The value of the variable (ϵ) is smaller than or equal to its threshold (1.0e-6).
    The algorithm performed a step with a change (6.5347623783315016e-9) less than 1.0e-6.

where we now see that the subsolver always only requires one step. Note that since debug of an iteration is happening *after* a step, we see the sub solver run *before* the debug for an iteration number.
