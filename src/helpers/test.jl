"""
    Manopt.Test

The module `Manopt.Test` provides dummy types and small test problems and examples
that can be used throughout testing.

Some of these are simplified variants from problems from `ManoptExamples.jl`,
that are added here to not introduce a circular dependency.

Some of the functionality is only populated when certain packages are loaded,
that is
* `Test.jl`
* `Manifolds.jl`
"""
module Test
using ..Manopt
using ..Manopt: AbstractManifoldObjective, AbstractManoptProblem, AbstractEvaluationType
using ..Manopt: AbstractManoptSolverState
using ..Manopt: StoppingCriterionSet, StoppingCriterion
using ManifoldsBase
using ManifoldDiff

#
#
# Dummy types
struct DummyManifold <: AbstractManifold{ManifoldsBase.ℝ} end

struct DummyDecoratedObjective{E, O <: AbstractManifoldObjective} <:
    Manopt.AbstractDecoratedManifoldObjective{E, O}
    objective::O
end
function DummyDecoratedObjective(
        o::O
    ) where {E <: AbstractEvaluationType, O <: AbstractManifoldObjective{E}}
    return DummyDecoratedObjective{E, O}(o)
end
function Manopt.status_summary(
        ddo::DummyDecoratedObjective; kwargs...
    )
    return "A dummy decorator for " * Manopt.status_summary(ddo.objective; kwargs...)
end
function Base.show(io::IO, ddo::DummyDecoratedObjective)
    print(io, "DummyDecoratedObjective(")
    print(io, ddo.objective)
    return print(io, ")")
end

struct DummyProblem{M <: AbstractManifold} <: AbstractManoptProblem{M} end
struct DummyStoppingCriteriaSet <: StoppingCriterionSet end
struct DummyStoppingCriterion <: StoppingCriterion end

mutable struct DummyState <: AbstractManoptSolverState
    storage::Vector{Float64}
end
DummyState() = DummyState([])
Manopt.status_summary(ds::DummyState; context = :Default) = "A Manopt Test state with storage $(ds.storage)"
Base.show(io::IO, ds::DummyState) = print(io, "Manopt.Test.DummyState($(ds.storage))")
Manopt.get_iterate(::DummyState) = NaN
Manopt.set_parameter!(s::DummyState, ::Val, v) = s
Manopt.set_parameter!(s::DummyState, ::Val{:StoppingCriterion}, v) = s
"""
    M, f, grad_f, p0, p_star = Circle_mean_task()

Create a small mean problem on the circle to test Number-based algorithms
Requires `Manifolds.jl` to be loaded, use [`Manopt.Test.mean_task`](@ref)`(M, data)`
for the general case
"""
function Circle_mean_task end

@doc raw"""
    f, grad_f = Manopt.Test.mean_task(M, data)

Returns cost and gradient for computing the mean of `data` ``d_i`` on manifold `M`

```math
\begin{align*}
f(p) = \frac{1}{2n} \sum_{i=1}^n d_M(p, d_i)^2
\operatorname{grad} f(p) = -\frac{1}{n} \sum_{i=1}^n \log_p(d_i)
\end{align*}
"""
function mean_task(M::AbstractManifold, data::AbstractVector)
    n = length(data)
    f(M, p) = 1 / (2n) * sum(distance.(Ref(M), Ref(p), data) .^ 2)
    grad_f(M, p) = -1 / n * sum(log.(Ref(M), Ref(p), data))
    return f, grad_f
end

#
#
# From ManoptExamples – to avoid a circular dependency
# Maybe the examples using these could also be simplified instead.
function adjoint_differential_forward_logs(
        M::PowerManifold{𝔽, TM, TSize, TPR}, p, X
    ) where {𝔽, TM, TSize, TPR}
    Y = zero_vector(M, p)
    return adjoint_differential_forward_logs!(M, Y, p, X)
end
function adjoint_differential_forward_logs!(
        M::PowerManifold{𝔽, TM, TSize, TPR}, Y, p, X
    ) where {𝔽, TM, TSize, TPR}
    power_size = power_dimensions(M)
    d = length(power_size)
    N = PowerManifold(M.manifold, TPR(), power_size..., d)
    R = CartesianIndices(Tuple(power_size))
    maxInd = last(R).I
    # since we add things in Y, make sure we start at zero.
    zero_vector!(M, Y, p)
    for i in R # iterate over all pixel
        for k in 1:d # for all direction combinations
            I = [i.I...] # array of index
            J = I .+ 1 .* (1:d .== k) #i + e_k is j
            if all(J .<= maxInd) # is this neighbor in range?
                j = CartesianIndex{d}(J...) # neighbour index as Cartesian Index
                Y[M, I...] =
                    Y[M, I...] + ManifoldDiff.adjoint_differential_log_basepoint(
                    M.manifold, p[M, I...], p[M, J...], X[N, I..., k]
                )
                Y[M, J...] =
                    Y[M, J...] + ManifoldDiff.adjoint_differential_log_argument(
                    M.manifold, p[M, J...], p[M, I...], X[N, I..., k]
                )
            end
        end # directions
    end # i in R
    return Y
end
function differential_forward_logs(M::PowerManifold, p, X)
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    maxInd = last(R).I
    d2 = (d > 1) ? ones(Int, d + 1) + (d - 1) * (1:(d + 1) .== d + 1) : 1
    if d > 1
        N = PowerManifold(M.manifold, NestedPowerRepresentation(), power_size..., d)
    else
        N = PowerManifold(M.manifold, NestedPowerRepresentation(), power_size...)
    end
    Y = zero_vector(N, repeat(p; inner = d2))
    return differential_forward_logs!(M, Y, p, X)
end
function differential_forward_logs!(M::PowerManifold, Y, p, X)
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    maxInd = last(R).I
    e_k_vals = [1 * (1:d .== k) for k in 1:d]
    if d > 1
        N = PowerManifold(M.manifold, NestedPowerRepresentation(), power_size..., d)
    else
        N = PowerManifold(M.manifold, NestedPowerRepresentation(), power_size...)
    end
    for i in R # iterate over all pixel
        for k in 1:d # for all direction combinations
            I = i.I # array of index
            J = I .+ e_k_vals[k] #i + e_k is j
            if all(J .<= maxInd)
                # this is neighbor in range,
                # collects two, namely in kth direction since xi appears as base and arg
                Y[N, I..., k] =
                    ManifoldDiff.differential_log_basepoint(
                    M.manifold, p[M, I...], p[M, J...], X[M, I...]
                ) .+ ManifoldDiff.differential_log_argument(
                    M.manifold, p[M, I...], p[M, J...], X[M, J...]
                )
            else
                Y[N, I..., k] = zero_vector(M.manifold, p[M, I...])
            end
        end # directions
    end # i in R
    return Y
end
function forward_logs(M::PowerManifold{𝔽, TM, TSize, TPR}, p) where {𝔽, TM, TSize, TPR}
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    sX = size(p)
    maxInd = last(R).I
    if d > 1
        d2 = fill(1, d + 1)
        d2[d + 1] = d
    else
        d2 = 1
    end
    sN = d > 1 ? [power_size..., d] : [power_size...]
    N = PowerManifold(M.manifold, TPR(), sN...)
    xT = repeat(p; inner = d2)
    X = zero_vector(N, xT)
    e_k_vals = [1 * (1:d .== k) for k in 1:d]
    for i in R # iterate over all pixel
        for k in 1:d # for all direction combinations
            I = i.I
            J = I .+ 1 .* e_k_vals[k] #i + e_k is j
            if all(J .<= maxInd) # is this neighbor in range?
                j = CartesianIndex{d}(J...) # neighbour index as Cartesian Index
                X[N, i.I..., k] = log(M.manifold, p[M, i.I...], p[M, j.I...])
            end
        end # directions
    end # i in R
    return X
end
function forward_logs!(M::PowerManifold{𝔽, TM, TSize, TPR}, X, p) where {𝔽, TM, TSize, TPR}
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    sX = size(p)
    maxInd = last(R).I
    if d > 1
        d2 = fill(1, d + 1)
        d2[d + 1] = d
    else
        d2 = 1
    end
    sN = d > 1 ? [power_size..., d] : [power_size...]
    N = PowerManifold(M.manifold, TPR(), sN...)
    e_k_vals = [1 * (1:d .== k) for k in 1:d]
    for i in R # iterate over all pixel
        for k in 1:d # for all direction combinations
            I = i.I
            J = I .+ 1 .* e_k_vals[k] #i + e_k is j
            if all(J .<= maxInd) # is this neighbor in range?
                j = CartesianIndex{d}(J...) # neighbour index as Cartesian Index
                X[N, i.I..., k] = log(M.manifold, p[M, i.I...], p[M, j.I...])
            else
                X[N, i.I..., k] = zero_vector(M.manifold, p[M, i.I...])
            end
        end # directions
    end # i in R
    return X
end
function L2_Total_Variation(M, p_data, α, p)
    return 1 / 2 * distance(M, p_data, p)^2 + α * Total_Variation(M, p)
end
function project_collaborative_TV(N::PowerManifold, λ, x, Ξ, p = 2.0, q = 1.0, α = 1.0)
    pdims = power_dimensions(N)
    if length(pdims) == 1
        d = 1
        s = 1
        iRep = (1,)
    else
        d = pdims[end]
        s = length(pdims) - 1
        if s != d
            throw(
                ErrorException(
                    "the last dimension ($(d)) has to be equal to the number of the previous ones ($(s)) but its not.",
                ),
            )
        end
        iRep = (Integer.(ones(d))..., d)
    end
    if q == 1 # Example 3 case 2
        if p == 1
            normΞ = norm.(Ref(N.manifold), x, Ξ)
            return max.(normΞ .- λ, 0.0) ./ ((normΞ .== 0) .+ normΞ) .* Ξ
        end
        if p == 2 # Example 3 case 3
            norms = sqrt.(sum(norm.(Ref(N.manifold), x, Ξ) .^ 2; dims = d + 1))
            if length(iRep) > 1
                norms = repeat(norms; inner = iRep)
            end
            # if the norm is zero add 1 to avoid division by zero, also then the
            # nominator is already (max(-λ,0) = 0) so it stays zero then
            return max.(norms .- λ, 0.0) ./ ((norms .== 0) .+ norms) .* Ξ
        end
        throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
    elseif q == Inf
        if p == 2
            norms = sqrt.(sum(norm.(Ref(N.manifold), x, Ξ) .^ 2; dims = d + 1))
            if length(iRep) > 1
                norms = repeat(norms; inner = iRep)
            end
        elseif p == 1
            norms = sum(norm.(Ref(N.manifold), x, Ξ); dims = d + 1)
            if length(iRep) > 1
                norms = repeat(norms; inner = iRep)
            end
        elseif p == Inf
            norms = norm.(Ref(N.manifold), x, Ξ)
        else
            throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
        end
        return (α .* Ξ) ./ max.(Ref(α), norms)
    end # end q
    return throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
end
function project_collaborative_TV(N::PowerManifold, λ, x, Ξ, p::Int, q::Float64 = 1.0, α = 1.0)
    return project_collaborative_TV(N, λ, x, Ξ, Float64(p), q, α)
end
function project_collaborative_TV(N::PowerManifold, λ, x, Ξ, p::Float64, q::Int, α = 1.0)
    return project_collaborative_TV(N, λ, x, Ξ, p, Float64(q), α)
end
function project_collaborative_TV(N::PowerManifold, λ, x, Ξ, p::Int, q::Int, α = 1.0)
    return project_collaborative_TV(N, λ, x, Ξ, Float64(p), Float64(q), α)
end
function project_collaborative_TV!(N::PowerManifold, Θ, λ, x, Ξ, p = 2.0, q = 1.0, α = 1.0)
    pdims = power_dimensions(N)
    if length(pdims) == 1
        d = 1
        s = 1
        iRep = (1,)
    else
        d = pdims[end]
        s = length(pdims) - 1
        if s != d
            throw(
                ErrorException(
                    "the last dimension ($d) has to be equal to the number of the previous ones ($s) but its not.",
                ),
            )
        end
        iRep = (Integer.(ones(d))..., d)
    end
    if q == 1 # Example 3 case 2
        if p == 1
            normΞ = norm.(Ref(N.manifold), x, Ξ)
            Θ .= max.(normΞ .- λ, 0.0) ./ ((normΞ .== 0) .+ normΞ) .* Ξ
            return Θ
        elseif p == 2 # Example 3 case 3
            norms = sqrt.(sum(norm.(Ref(N.manifold), x, Ξ) .^ 2; dims = d + 1))
            if length(iRep) > 1
                norms = repeat(norms; inner = iRep)
            end
            # if the norm is zero add 1 to avoid division by zero, also then the
            # nominator is already (max(-λ,0) = 0) so it stays zero then
            Θ .= max.(norms .- λ, 0.0) ./ ((norms .== 0) .+ norms) .* Ξ
            return Θ
        else
            throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
        end
    elseif q == Inf
        if p == 2
            norms = sqrt.(sum(norm.(Ref(N.manifold), x, Ξ) .^ 2; dims = d + 1))
            (length(iRep) > 1) && (norms = repeat(norms; inner = iRep))
        elseif p == 1
            norms = sum(norm.(Ref(N.manifold), x, Ξ); dims = d + 1)
            (length(iRep) > 1) && (norms = repeat(norms; inner = iRep))
        elseif p == Inf
            norms = norm.(Ref(N.manifold), x, Ξ)
        else
            throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
        end
        Θ .= (α .* Ξ) ./ max.(Ref(α), norms)
        return Θ
    end # end q
    return throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
end
function project_collaborative_TV!(
        N::PowerManifold, Θ, λ, x, Ξ, p::Int, q::Float64 = 1.0, α = 1.0
    )
    return project_collaborative_TV!(N, Θ, λ, x, Ξ, Float64(p), q, α)
end
function project_collaborative_TV!(N::PowerManifold, Θ, λ, x, Ξ, p::Float64, q::Int, α = 1.0)
    return project_collaborative_TV!(N, Θ, λ, x, Ξ, p, Float64(q), α)
end
function project_collaborative_TV!(N::PowerManifold, Θ, λ, x, Ξ, p::Int, q::Int, α = 1.0)
    return project_collaborative_TV!(N, Θ, λ, x, Ξ, Float64(p), Float64(q), α)
end
function prox_Total_Variation(
        M::AbstractManifold, λ::Number, x::Tuple{T, T}, p::Int = 1
    ) where {T}
    d = distance(M, x[1], x[2])
    if p == 1
        t = min(0.5, λ / d)
    elseif p == 2
        t = λ / (1 + 2 * λ)
    else
        throw(
            ErrorException(
                "Proximal Map of TV(M,x1,x2,p) not implemented for p=$(p) (requires p=1 or 2)",
            ),
        )
    end
    return (
        ManifoldsBase.exp_fused(M, x[1], log(M, x[1], x[2]), t),
        ManifoldsBase.exp_fused(M, x[2], log(M, x[2], x[1]), t),
    )
end
function prox_Total_Variation(
        M::PowerManifold, λ::Number, x::Tuple{T, T}, p::Int = 1
    ) where {T}
    d = distance(M, x[1], x[2])
    if p == 1
        t = min(0.5, λ / d)
    elseif p == 2
        t = λ / (1 + 2 * λ)
    else
        throw(
            ErrorException(
                "Proximal Map of TV(M,x1,x2,p) not implemented for p=$(p) (requires p=1 or 2)",
            ),
        )
    end
    return (
        ManifoldsBase.exp_fused(M, x[1], log(M, x[1], x[2]), t),
        ManifoldsBase.exp_fused(M, x[2], log(M, x[2], x[1]), t),
    )
end

function prox_Total_Variation!(
        M::AbstractManifold, y, λ::Number, x::Tuple{T, T}, p::Int = 1
    ) where {T}
    d = distance(M, x[1], x[2])
    if p == 1
        t = min(0.5, λ / d)
    elseif p == 2
        t = λ / (1 + 2 * λ)
    else
        throw(
            ErrorException(
                "Proximal Map of TV(M,x1,x2,p) not implemented for p=$(p) (requires p=1 or 2)",
            ),
        )
    end
    X1 = log(M, x[1], x[2])
    X2 = log(M, x[2], x[1])
    ManifoldsBase.exp_fused!(M, y[1], x[1], X1, t)
    ManifoldsBase.exp_fused!(M, y[2], x[2], X2, t)
    return y
end
function prox_Total_Variation!(
        M::PowerManifold, y, λ::Number, x::Tuple{T, T}, p::Int = 1
    ) where {T}
    d = distance(M, x[1], x[2])
    if p == 1
        t = min(0.5, λ / d)
    elseif p == 2
        t = λ / (1 + 2 * λ)
    else
        throw(
            ErrorException(
                "Proximal Map of TV(M,x1,x2,p) not implemented for p=$(p) (requires p=1 or 2)",
            ),
        )
    end
    X1 = log(M, x[1], x[2])
    X2 = log(M, x[2], x[1])
    ManifoldsBase.exp_fused!(M, y[1], x[1], X1, t)
    ManifoldsBase.exp_fused!(M, y[2], x[2], X2, t)
    return y
end
function prox_Total_Variation(M::PowerManifold, λ, x, p::Int = 1)
    y = deepcopy(x)
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    maxInd = last(R).I
    for k in 1:d # for all directions
        ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
        for l in 0:1
            for i in R # iterate over all pixel
                if (i[k] % 2) == l
                    J = i.I .+ ek.I #i + e_k is j
                    if all(J .<= maxInd) # is this neighbor in range?
                        j = CartesianIndex(J...) # neighbour index as Cartesian Index
                        (y[i], y[j]) = prox_Total_Variation(M.manifold, λ, (y[i], y[j]), p) # Compute TV on these
                    end
                end
            end # i in R
        end # even odd
    end # directions
    return y
end
function prox_Total_Variation!(M::PowerManifold, y, λ, x, p::Int = 1)
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    copyto!(M, y, x)
    maxInd = last(R).I
    for k in 1:d # for all directions
        ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
        for l in 0:1
            for i in R # iterate over all pixel
                if (i[k] % 2) == l # even/odd splitting
                    J = i.I .+ ek.I #i + e_k is j
                    if all(J .<= maxInd) # is this neighbor in range?
                        j = CartesianIndex(J...) # neighbour index as Cartesian Index
                        prox_Total_Variation!(M.manifold, [y[i], y[j]], λ, (y[i], y[j]), p) # Compute TV on these
                    end
                end
            end # i in R
        end # even odd
    end # directions
    return y
end
function Total_Variation(M::PowerManifold, x, p = 1, q = 1)
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    maxInd = last(R)
    cost = fill(0.0, Tuple(power_size))
    for k in 1:d # for all directions
        ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
        for i in R # iterate over all pixel
            j = i + ek # compute neighbor
            if all(map(<=, j.I, maxInd.I)) # is this neighbor in range?
                cost[i] += distance(M.manifold, x[M, Tuple(i)...], x[M, Tuple(j)...])^p
            end
        end
    end
    cost = (cost) .^ (1 / p)
    if q > 0
        return sum(cost .^ q)^(1 / q)
    else
        return cost
    end
end
#
#
# Further example functions - Chambolle-Pock
function differential_project_collaborative_TV(N::PowerManifold, p, ξ, η, p1 = 2.0, p2 = 1.0)
    ζ = zero_vector(N, p)
    return differential_project_collaborative_TV!(N, ζ, p, ξ, η, p1, p2)
end
function differential_project_collaborative_TV!(
        N::PowerManifold, ζ, p, ξ, η, p1 = 2.0, p2 = 1.0
    )
    ζ = zero_vector!(N, ζ, p)
    pdims = power_dimensions(N)
    if length(pdims) == 1
        d = 1
        s = 1
        R = CartesianIndices(Tuple(pdims))
    else
        d = pdims[end]
        s = length(pdims) - 1
        if s != d
            throw(
                ErrorException(
                    "the last dimension ($(d)) has to be equal to the number of the previous ones ($(s)) but its not.",
                ),
            )
        end
        R = CartesianIndices(Tuple(pdims[1:(end - 1)]))
    end

    # R = CartesianIndices(Tuple(power_size))
    maxInd = last(R).I
    e_k_vals = [1 * (1:d .== k) for k in 1:d]

    if p2 == Inf
        if p1 == Inf || d == 1
            norms = norm.(Ref(N.manifold), p, ξ)

            for i in R # iterate over all pixel
                for k in 1:d # for all direction combinations
                    I = i.I # array of index
                    J = I .+ e_k_vals[k] #`i + e_k` is `j`
                    if all(J .<= maxInd)
                        # this is neighbor in range,
                        ζ[N, I..., k] += if norms[I..., k] <= 1
                            η[N, I..., k]
                        else
                            1 / norms[I..., k] * (
                                η[N, I..., k] .-
                                    1 / norms[I..., k]^2 .* inner(
                                    N.manifold,
                                    p[N, I..., k],
                                    η[N, I..., k],
                                    ξ[N, I..., k],
                                ) .* ξ[N, I..., k]
                            )
                        end
                    else
                        ζ[N, I..., k] = zero_vector(N.manifold, p[N, I..., k])
                    end
                end # directions
            end # end iterate over all pixel in R
            return ζ
        elseif p1 == 2
            norms = norm.(Ref(N.manifold), p, ξ)
            norms_ = sqrt.(sum(norms .^ 2; dims = length(pdims)))

            for i in R # iterate over all pixel
                for k in 1:d # for all direction combinations
                    I = i.I # array of index
                    J = I .+ e_k_vals[k] # `i + e_k` is `j`
                    if all(J .<= maxInd)
                        # this is neighbor in range,
                        if norms_[I...] <= 1
                            ζ[N, I..., k] += η[N, I..., k]
                        else
                            for κ in 1:d
                                ζ[N, I..., κ] += if k != κ
                                    -1 / norms_[I...]^3 * inner(
                                        N.manifold,
                                        p[N, I..., k],
                                        η[N, I..., k],
                                        ξ[N, I..., k],
                                    ) .* ξ[N, I..., κ]
                                else
                                    1 / norms_[I...] * (
                                        η[N, I..., k] .-
                                            1 / norms_[I...]^2 .* inner(
                                            N.manifold,
                                            p[N, I..., k],
                                            η[N, I..., k],
                                            ξ[N, I..., k],
                                        ) .* ξ[N, I..., k]
                                    )
                                end
                            end
                        end
                    else
                        ζ[N, I..., k] = zero_vector(N.manifold, p[N, I..., k])
                    end
                end # directions
            end # end iterate over all pixel in R
            return ζ
        else
            throw(ErrorException("The case p=$p1, q=$p2 is not yet implemented"))
        end
    end # end q
    throw(ErrorException("The case p=$p1, q=$p2 is not yet implemented"))
end
# PDSSN
function differential_project_collaborative_TV(N::PowerManifold, λ, x, Ξ, Η, p, q, γ)
    Y = zero_vector(N, x)
    # print("Ξ = $(Ξ)")

    pdims = power_dimensions(N)
    if length(pdims) == 1
        d = 1
        s = 1
        R = CartesianIndices(Tuple(pdims))
    else
        d = pdims[end]
        s = length(pdims) - 1
        if s != d
            throw(
                ErrorException(
                    "the last dimension ($(d)) has to be equal to the number of the previous ones ($(s)) but its not.",
                ),
            )
        end
        R = CartesianIndices(Tuple(pdims[1:(end - 1)]))
    end

    # R = CartesianIndices(Tuple(power_size))
    maxInd = last(R).I
    e_k_vals = [1 * (1:d .== k) for k in 1:d]

    if q == Inf
        if p == Inf || d == 1
            norms = norm.(Ref(N.manifold), x, Ξ)

            for i in R # iterate over all pixel
                for k in 1:d # for all direction combinations
                    I = i.I # array of `index`
                    J = I .+ e_k_vals[k] # `i + e_k` is `j`
                    if all(J .<= maxInd)
                        # this is neighbor in range,
                        Y[N, I..., k] += if norms[I..., k] <= (1 + λ * γ)
                            Η[N, I..., k] ./ (1 + λ * γ)
                        else
                            1 / norms[I..., k] * (
                                Η[N, I..., k] .-
                                    1 / norms[I..., k]^2 .* inner(
                                    N.manifold,
                                    x[N, I..., k],
                                    Η[N, I..., k],
                                    Ξ[N, I..., k],
                                ) .* Ξ[N, I..., k]
                            )
                        end
                    else
                        Y[N, I..., k] = zero_vector(N.manifold, x[N, I..., k])
                    end
                end # directions
            end # `i` in R
            return Y
        elseif p == 2
            norms = norm.(Ref(N.manifold), x, Ξ)
            norms_ = sqrt.(sum(norms .^ 2; dims = length(pdims)))

            for i in R # iterate over all pixel
                for k in 1:d # for all direction combinations
                    I = i.I # array of `index`
                    J = I .+ e_k_vals[k] #`i + e_k` is `j`
                    if all(J .<= maxInd)
                        # this is neighbor in range,
                        if norms_[I...] <= (1 + λ * γ)
                            Y[N, I..., k] += Η[N, I..., k] ./ (1 + λ * γ)
                        else
                            for κ in 1:d
                                Y[N, I..., κ] += if k != κ
                                    -1 / norms_[I...]^3 * inner(
                                        N.manifold,
                                        x[N, I..., k],
                                        Η[N, I..., k],
                                        Ξ[N, I..., k],
                                    ) .* Ξ[N, I..., κ]
                                else
                                    1 / norms_[I...] * (
                                        Η[N, I..., k] .-
                                            1 / norms_[I...]^2 .* inner(
                                            N.manifold,
                                            x[N, I..., k],
                                            Η[N, I..., k],
                                            Ξ[N, I..., k],
                                        ) .* Ξ[N, I..., k]
                                    )
                                end
                            end
                        end
                    else
                        Y[N, I..., k] = zero_vector(N.manifold, x[N, I..., k])
                    end
                end # directions
            end # `ì` in R
            return Y
        else
            throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
        end
    end # end q
    throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
end
end
