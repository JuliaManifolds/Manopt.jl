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

Two stopping criteria for [`convex_bundle_method`](@ref) and [`proximal_bundle_method`](@ref) to indicate to stop when either

* the parameters ε and ``\lvert g \rvert``

are less than given tolerances `tol_errors` and `tol_search_dir` respectively, or

* the parameter ``-ξ = \lvert g \rvert^2 + ε``

is less than a given tolerance `tol_lag_mult`.

# Constructors

    StopWhenLagrangeMultiplierLess(tol_errors=1e-6, tol_search_dir=1e-3)

    StopWhenLagrangeMultiplierLess(tol_lag_mult=1e-6)

"""
mutable struct StopWhenLagrangeMultiplierLess{T,R} <: StoppingCriterion
    tol_errors::T
    tol_search_dir::T
    tol_lag_mult::R
    reason::String
    at_iteration::Int
    function StopWhenLagrangeMultiplierLess(tol_errors::T, tol_search_dir::T) where {T}
        return new{T,Nothing}(tol_errors, tol_search_dir, nothing, "", 0)
    end
    function StopWhenLagrangeMultiplierLess(tol_lag_mult::R=1e-6) where {R}
        return new{Nothing,R}(nothing, nothing, tol_lag_mult, "", 0)
    end
end

function status_summary(b::StopWhenLagrangeMultiplierLess{T,Nothing}) where {T}
    s = length(b.reason) > 0 ? "reached" : "not reached"
    return "Stopping parameter: ε ≤ $(b.tol_errors), |g| ≤ $(b.tol_search_dir):\t$s"
end
function status_summary(b::StopWhenLagrangeMultiplierLess{Nothing,R}) where {R}
    s = length(b.reason) > 0 ? "reached" : "not reached"
    return "Stopping parameter: -ξ ≤ $(b.tol_lag_mult):\t$s"
end
function show(io::IO, b::StopWhenLagrangeMultiplierLess{T,Nothing}) where {T}
    return print(
        io,
        "StopWhenLagrangeMultiplierLess($(b.tol_errors), $(b.tol_search_dir))\n    $(status_summary(b))",
    )
end
function show(io::IO, b::StopWhenLagrangeMultiplierLess{Nothing,R}) where {R}
    return print(
        io, "StopWhenLagrangeMultiplierLess($(b.tol_lag_mult))\n    $(status_summary(b))"
    )
end

@doc raw"""
    DebugWarnIfLagrangeMultiplierIncreases <: DebugAction

print a warning if the stopping parameter of the bundle method increases.

# Constructor
    DebugWarnIfLagrangeMultiplierIncreases(warn=:Once; tol=1e2)

Initialize the warning to warning level (`:Once`) and introduce a tolerance for the test of `1e2`.

The `warn` level can be set to `:Once` to only warn the first time the cost increases,
to `:Always` to report an increase every time it happens, and it can be set to `:No`
to deactivate the warning, then this [`DebugAction`](@ref) is inactive.
All other symbols are handled as if they were `:Always:`
"""
mutable struct DebugWarnIfLagrangeMultiplierIncreases <: DebugAction
    # store if we need to warn – :Once, :Always, :No, where all others are handled
    # the same as :Always
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
