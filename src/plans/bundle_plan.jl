#
# Common files for bundle-based solvers
#

function convex_bundle_method_subsolver end
function convex_bundle_method_subsolver! end
@doc """
    λ = convex_bundle_method_subsolver(M, p_last_serious, linearization_errors, transported_subgradients)
    convex_bundle_method_subsolver!(M, λ, p_last_serious, linearization_errors, transported_subgradients)

solver for the subproblem of the convex bundle method
at the last serious iterate ``p_k`` given the current linearization errors ``c_j^k``,
and transported subgradients ``$(_tex(:rm, "P"))_{p_k←q_j} X_{q_j}``.

The computation can also be done in-place of `λ`.

The subproblem for the convex bundle method is
```math
\\begin{align*}
    $(_tex(:argmin))_{λ ∈ ℝ^{$(_tex(:abs, "J_k"))}
    &
    $(_tex(:frac, "1", "2"))$(_tex(:Bigl))\\lVert $(_tex(:sum, "j ∈ J_k")) λ_j $(_tex(:rm, "P"))_{p_k←q_j} X_{q_j} $(_tex(:Bigl))\\rVert^2
    + $(_tex(:sum, "j ∈ J_k")), "λ_j \\, c_j^k"
    \\\\
    $(_tex(:text, "s. t."))$(_tex(:quad)) &
    $(_tex(:sum, "j ∈ J_k")) λ_j = 1,
    $(_tex(:quad)) λ_j ≥ 0
    $(_tex(:quad)) $(_tex(:text, "for all "))
    j ∈ J_k,
\end{align*}
```

where ``J_k = $(_tex(:set, "j ∈ J_{k-1} \\ | \\ λ_j > 0")) ∪ $(_tex(:set, "k"))``.
See [BergmannHerzogJasa:2024](@cite) for more details

!!! tip
    A default subsolver based on [`RipQP`.jl](https://github.com/JuliaSmoothOptimizers/RipQP.jl) and [`QuadraticModels`](https://github.com/JuliaSmoothOptimizers/QuadraticModels.jl)
    is available if these two packages are loaded.
"""
convex_bundle_method_subsolver(
    M, p_last_serious, linearization_errors, transported_subgradients
)

function proximal_bundle_method_subsolver end
function proximal_bundle_method_subsolver! end
@doc """
    λ = proximal_bundle_method_subsolver(M, p_last_serious, μ, approximation_errors, transported_subgradients)
    proximal_bundle_method_subsolver!(M, λ, p_last_serious, μ, approximation_errors, transported_subgradients)

solver for the subproblem of the proximal bundle method.

The subproblem for the proximal bundle method is
```math
\\begin{align*}
    $(_tex(:argmin))_{λ ∈ ℝ^{$(_tex(:abs, "L_l"))} &
    $(_tex(:frac, "1", "2 μ_l")) $(_tex(:Bigl)) \\lVert $(_tex(:sum, "j ∈ L_l")) λ_j $(_tex(:rm, "P"))_{p_k←q_j} X_{q_j}$(_tex(:Bigr))\rVert^2
    + $(_tex(:sum, "j ∈ L_l")) "λ_j \\, c_j^k
    \\\\
    $(_tex(:text, "s. t.")) $(_tex(:quad)) &
    $(_tex(:sum, "j ∈ L_l")) λ_j = 1,
    $(_tex(:quad)) λ_j ≥ 0
    $(_tex(:quad)) $(_tex(:text, "for all ")) j ∈ L_l,
\\end{align*}
```
where ``L_l = $(_tex(:set, "k"))`` if ``q_k`` is a serious iterate, and ``L_l = L_{l-1}  ∪ $(_tex(:set, "k"))`` otherwise.
See [HoseiniMonjeziNobakhtianPouryayevali:2021](@cite).

!!! tip
    A default subsolver based on [`RipQP`.jl](https://github.com/JuliaSmoothOptimizers/RipQP.jl) and [`QuadraticModels`](https://github.com/JuliaSmoothOptimizers/QuadraticModels.jl)
    is available if these two packages are loaded.
"""
proximal_bundle_method_subsolver(
    M, p_last_serious, μ, approximation_errors, transported_subgradients
)

@doc """
    StopWhenLagrangeMultiplierLess <: StoppingCriterion

Stopping Criteria for Lagrange multipliers.

Currently these are meant for the [`convex_bundle_method`](@ref) and [`proximal_bundle_method`](@ref),
where based on the Lagrange multipliers an approximate (sub)gradient ``g`` and an error estimate ``ε``
is computed.

The `mode=:both` requires that both
``ε`` and ``$(_tex(:abs, "g"))`` are smaller than their `tolerance`s for the [`convex_bundle_method`](@ref),
and that
``c`` and ``$(_tex(:abs, "d"))`` are smaller than their `tolerance`s for the [`proximal_bundle_method`](@ref).

The `mode=:estimate` requires that, for the [`convex_bundle_method`](@ref)
``-ξ = $(_tex(:abs, "g"))^2 + ε`` is less than a given `tolerance`.
For the [`proximal_bundle_method`](@ref), the equation reads ``-ν = μ $(_tex(:abs, "d"))^2 + c``.

# Constructors

    StopWhenLagrangeMultiplierLess(tolerance=1e-6; mode::Symbol=:estimate, names=nothing)

Create the stopping criterion for one of the `mode`s mentioned.
Note that tolerance can be a single number for the `:estimate` case,
but a vector of two values is required for the `:both` mode.
Here the first entry specifies the tolerance for ``ε`` (``c``),
the second the tolerance for ``$(_tex(:abs, "g"))`` (``$(_tex(:abs, "d"))``), respectively.
"""
mutable struct StopWhenLagrangeMultiplierLess{
        T <: Real, A <: AbstractVector{<:T}, B <: Union{Nothing, <:AbstractVector{<:String}},
    } <: StoppingCriterion
    tolerances::A
    values::A
    names::B
    mode::Symbol
    at_iteration::Int
    function StopWhenLagrangeMultiplierLess(
            tol::T; mode::Symbol = :estimate, names::B = nothing
        ) where {T <: Real, B <: Union{Nothing, <:AbstractVector{<:String}}}
        return new{T, Vector{T}, B}([tol], zero([tol]), names, mode, -1)
    end
    function StopWhenLagrangeMultiplierLess(
            tols::A; mode::Symbol = :estimate, names::B = nothing
        ) where {T <: Real, A <: AbstractVector{<:T}, B <: Union{Nothing, <:AbstractVector{<:String}}}
        return new{T, A, B}(tols, zero(tols), names, mode, -1)
    end
end
function get_reason(sc::StopWhenLagrangeMultiplierLess)
    if (sc.at_iteration >= 0)
        if isnothing(sc.names)
            tol_str = join(
                ["$ai < $bi" for (ai, bi) in zip(sc.values, sc.tolerances)], ", "
            )
        else
            tol_str = join(
                [
                    "$si = $ai < $bi" for
                        (si, ai, bi) in zip(sc.names, sc.values, sc.tolerances)
                ],
                ", ",
            )
        end
        return "After $(sc.at_iteration) iterations the algorithm reached an approximate critical point with tolerances $tol_str.\n"
    end
    return ""
end

function status_summary(sc::StopWhenLagrangeMultiplierLess; inline = false)
    s = (sc.at_iteration >= 0) ? "reached" : "not reached"
    msg = "Lagrange multipliers"
    isnothing(sc.names) && (msg *= " with tolerances $(sc.tolerances)")
    if !isnothing(sc.names)
        msg *= join(["$si < $bi" for (si, bi) in zip(sc.names, sc.tolerances)], ", ")
    end

    return (inline ? "" : "A stopping criterion to stop when the Lagrange multipliers are less than $(sc.tolerances).\n$(_MANOPT_INDENT)") * "$(msg):$(_MANOPT_INDENT)$(s)"
end
function show(io::IO, sc::StopWhenLagrangeMultiplierLess)
    n = isnothing(sc.names) ? "" : ", $(names)"
    return print(
        io,
        "StopWhenLagrangeMultiplierLess($(sc.tolerances); mode=:$(sc.mode)$n)",
    )
end

@doc """
    DebugWarnIfLagrangeMultiplierIncreases <: DebugAction

print a warning if the Lagrange parameter based value ``-ξ`` of the bundle method increases.

# Constructor

    DebugWarnIfLagrangeMultiplierIncreases(warn=:Once; tol=1e2)

Initialize the warning to warning level (`:Once`) and introduce a tolerance for the test of `1e2`.

The `warn` level can be set to `:Once` to only warn the first time the cost increases,
to `:Always` to report an increase every time it happens, and it can be set to `:No`
to deactivate the warning, then this [`DebugAction`](@ref) is inactive.
All other symbols are handled as if they were `:Always`.
"""
mutable struct DebugWarnIfLagrangeMultiplierIncreases <: DebugAction
    status::Symbol
    old_value::Float64
    tol::Float64
    function DebugWarnIfLagrangeMultiplierIncreases(warn::Symbol = :Once; tol = 1.0e2)
        return new(warn, Float64(Inf), tol)
    end
end
function show(io::IO, di::DebugWarnIfLagrangeMultiplierIncreases)
    return print(io, "DebugWarnIfLagrangeMultiplierIncreases(; tol=\"$(di.tol)\")")
end
