#
# Common files for bunlde-based solvers
#

function bundle_method_subsolver end
@doc raw"""
    bundle_method_subsolver(M, bms<:Union{ConvexBundleMethodState, ProximalBundleMethodState})

solver for the subproblem of both the convex and proximal bundle methods.

The subproblem for the convex bundle method is
```math
\begin{align*}
    \operatorname*{arg\,min}_{λ ∈ ℝ^{\lvert J_k\rvert}}&
    \frac{1}{2} \Bigl\lVert \sum_{j ∈ J_k} λ_j \mathrm{P}_{p_k←q_j} X_{q_j} \Bigr\rVert^2
    + \sum_{j ∈ J_k} λ_j \, c_j^k
    \\
    \text{s. t.}\quad &
    \sum_{j ∈ J_k} λ_j = 1,
    \quad λ_j ≥ 0
    \quad \text{for all }
    j ∈ J_k,
\end{align*}
```
where ``J_k = \{j ∈ J_{k-1} \ | \ λ_j > 0\} \cup \{k\}``.

The subproblem for the proximal bundle method is

```math
\begin{align*}
    \operatorname*{arg\,min}_{λ ∈ ℝ^{\lvert L_l\rvert}} &
    \frac{1}{2 \mu_l} \Bigl\lVert \sum_{j ∈ L_l} λ_j \mathrm{P}_{p_k←q_j} X_{q_j} \Bigr\rVert^2
    + \sum_{j ∈ L_l} λ_j \, c_j^k
    \\
    \text{s. t.} \quad &
    \sum_{j ∈ L_l} λ_j = 1,
    \quad λ_j ≥ 0
    \quad \text{for all } j ∈ L_l,
\end{align*}
```
where ``L_l = \{k\}`` if ``q_k`` is a serious iterate, and ``L_l = L_{l-1} \cup \{k\}`` otherwise.
"""
bundle_method_subsolver(M, s) #change to problem state?

@doc raw"""
    StopWhenLagrangeMultiplierLess <: StoppingCriterion

Stopping Criteria for Lagrange multipliers.

Currenlty these are meant for the [`convex_bundle_method`](@ref) and [`proximal_bundle_method`](@ref),
where based on the Lagrange multipliers an approximate (sub)gradient ``g`` and an error estimate ``ε``
is computed.

In `mode=:both` we require that both
``ε`` and ``\lvert g \rvert`` are smaller than their `tolerance`s

In the `mode=:estimate` we require that ``-ξ = \lvert g \rvert^2 + ε``
is less than a given `tolerance`.

# Constructors

    StopWhenLagrangeMultiplierLess(tolerance=1e-6; mode::Symbol=:estimate)

Create the stopping criterion for one of the `mode`s mentioned.
Note that tolerance can be a single number for the `:estimate` case,
but a vector of two values is required for the `:both` mode.
Here the first entry specifies the tolerance for ``ε``,
the second the tolerance for ``\lvert g \rvert``, respectively.
"""
mutable struct StopWhenLagrangeMultiplierLess{T<:Real,A<:AbstractVector{<:T}} <:
               StoppingCriterion
    tolerance::A
    mode::Symbol
    reason::String
    at_iteration::Int
    function StopWhenLagrangeMultiplierLess(tol::T; mode::Symbol=:estimate) where {T<:Real}
        return new{T,Vector{T}}([tol], mode, "", 0)
    end
    function StopWhenLagrangeMultiplierLess(
        tols::A; mode::Symbol=:estimate
    ) where {T<:Real,A<:AbstractVector{<:T}}
        return new{T,A}(tols, mode, "", 0)
    end
end
function status_summary(sc::StopWhenLagrangeMultiplierLess)
    s = length(sc.reason) > 0 ? "reached" : "not reached"
    msg = ""
    (sc.mode === :both) && (msg = " ε ≤ $(sc.tolerance[1]) and |g| ≤ $(sc.tolerance[2])")
    (sc.mode === :estimate) && (msg = "  -ξ ≤ $(sc.tolerance[1])")
    return "Stopping parameter: $(msg) :\t$(s)"
end
function show(io::IO, sc::StopWhenLagrangeMultiplierLess)
    return print(
        io,
        "StopWhenLagrangeMultiplierLess($(sc.tolerance); mode=:$(sc.mode))\n    $(status_summary(sc))",
    )
end

@doc raw"""
    DebugWarnIfLagrangeMultiplierIncreases <: DebugAction

print a warning if the Lagrange parameter based value ``-ξ`` of the bundle method increases.

# Constructor

    DebugWarnIfLagrangeMultiplierIncreases(warn=:Once; tol=1e2)

Initialize the warning to warning level (`:Once`) and introduce a tolerance for the test of `1e2`.

The `warn` level can be set to `:Once` to only warn the first time the cost increases,
to `:Always` to report an increase every time it happens, and it can be set to `:No`
to deactivate the warning, then this [`DebugAction`](@ref) is inactive.
All other symbols are handled as if they were `:Always:`
"""
mutable struct DebugWarnIfLagrangeMultiplierIncreases <: DebugAction
    status::Symbol
    old_value::Float64
    tol::Float64
    function DebugWarnIfLagrangeMultiplierIncreases(warn::Symbol=:Once; tol=1e2)
        return new(warn, Float64(Inf), tol)
    end
end
function show(io::IO, di::DebugWarnIfLagrangeMultiplierIncreases)
    return print(io, "DebugWarnIfLagrangeMultiplierIncreases(; tol=\"$(di.tol)\")")
end
