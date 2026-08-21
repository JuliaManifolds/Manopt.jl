mutable struct LevenbergMarquardtBoxSubsolver{TSt <: AbstractManoptSolverState, F <: Real} <: AbstractManoptSolverState
    internal_state::TSt
    last_gcd_result::Symbol
    last_gcd_stepsize::F
end
function LevenbergMarquardtBoxSubsolver(::AbstractManifold, sub_state_::AbstractManoptSolverState, p)
    return LevenbergMarquardtBoxSubsolver{typeof(sub_state_), number_eltype(p)}(
        sub_state_, :not_searched, NaN,
    )
end
"""
    hessian_value(ha::LevenbergMarquardtBoxSubsolver, M, p, X::UnitVector, Y)

Evaluate the quadratic form associated with the stored Hessian approximation.
"""
function hessian_value(ha::LevenbergMarquardtBoxSubsolver, M::AbstractManifold, p, X::UnitVector, Y)
    return hessian_value(ha.internal_state, M, p, X, Y)
end

"""
    hessian_value_diag(ha::LevenbergMarquardtBoxSubsolver, M, p, X)

Evaluate the quadratic form associated with the stored Hessian approximation.
"""
function hessian_value_diag(ha::LevenbergMarquardtBoxSubsolver, M::AbstractManifold, p, X)
    return hessian_value_diag(ha.internal_state, M, p, X)
end

"""
    CoordinatesNormalSystemState <: AbstractManoptSolverState

A solver state indicating that the [`LevenbergMarquardtLinearSurrogateObjective`](@ref) is
solved using a linear system in coordinates of the tangent space at the current iterate.

# Fields

* `A`: an ``n×n`` matrix storing the normal-equations linear operator from [`get_linear_operator`](@ref) in coordinates, where `n` is the number of coordinates
* `b`: an ``n`` vector storing the right hand side of the normal equations in coordinates
* `basis::`[`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`): the basis the coordinates refer to
* `c`: an ``n`` vector storing the solution of the linear system in coordinates
* `linsolve!`: a function `(c, A, b) -> c` solving the linear system in-place of `c`

# Constructor
    CoordinatesNormalSystemState(
        M::AbstractManifold, p = rand(M);
        linsolve = default_lm_lin_solve!,
        basis = DefaultOrthonormalBasis(),
        A = nothing
    )

Construct the state, where not providing a memory for `A` uses the `eltype` of `p` to
determine the element type of the matrix to store. Note that the keyword is `linsolve`,
while the field it is stored in is called `linsolve!`.
"""
mutable struct CoordinatesNormalSystemState{F, TA <: AbstractMatrix, TB <: AbstractVector, TBA <: AbstractBasis} <: AbstractManoptSolverState
    A::TA
    b::TB
    basis::TBA
    c::TB
    linsolve!::F
end
function CoordinatesNormalSystemState(
        M::AbstractManifold, p = rand(M);
        linsolve::F = default_lm_lin_solve!, basis::B = DefaultOrthonormalBasis(), A = nothing
    ) where {F, B <: AbstractBasis}
    n = number_of_coordinates(M, basis)
    c = zeros(number_eltype(p), n)
    if isnothing(A)
        A = zeros(eltype(c), n, n)
    end
    b = zeros(eltype(c), n)
    return CoordinatesNormalSystemState{F, typeof(A), typeof(b), B}(A, b, basis, c, linsolve)
end

function get_solver_result(
        dmp::DefaultManoptProblem{<:TangentSpace, <:NormalEquationsObjective{<:AbstractLevenbergMarquardtLinearSurrogateObjective}},
        cnss::CoordinatesNormalSystemState
    )
    TpM = get_manifold(dmp)
    M = base_manifold(TpM)
    p = base_point(TpM)
    return get_vector(M, p, cnss.c, cnss.basis)
end
"""
    hessian_value(ha::CoordinatesNormalSystemState, M, p, X::UnitVector, Y)

Evaluate the quadratic form associated with the stored Hessian approximation.
Returns the scalar ``c_b^{$(_tex(:top))} B c`` where ``c_b`` are the coordinates of the
[`UnitVector`](@ref) `X` at `p` (assumed to correspond to the basis `ha.basis`),
``c`` are the coordinates of the tangent vector `Y` at `p` (in the basis `ha.basis`)
and ``B`` is `ha.A`.
"""
function hessian_value(ha::CoordinatesNormalSystemState, M::AbstractManifold, p, X::UnitVector, Y)
    b = to_coordinate_index(M, X, ha.basis)
    return dot(view(ha.A, b, :), get_coordinates(M, p, Y, ha.basis))
end

"""
    hessian_value_diag(ha::CoordinatesNormalSystemState, M::AbstractManifold, p, X::UnitVector)

Evaluate the quadratic form associated with the Hessian approximation of a [`CoordinatesNormalSystemState`](@ref).
Returns the scalar ``c^{$(_tex(:top))} A c`` where ``c`` are the coordinates of the
[`UnitVector`](@ref) `X` at `p` (in the basis `ha.basis`) and ``A`` is `ha.A`.
"""
function hessian_value_diag(ha::CoordinatesNormalSystemState, M::AbstractManifold, p, X::UnitVector)
    b = to_coordinate_index(M, X, ha.basis)
    return ha.A[b, b]
end
"""
    hessian_value_diag(ha::CoordinatesNormalSystemState, M::AbstractManifold, p, X)

Evaluate the quadratic form associated with the Hessian approximation of a [`CoordinatesNormalSystemState`](@ref).
Returns the scalar ``c^{$(_tex(:top))} A c`` where ``c`` are the coordinates of `X` at `p`
(in the basis `ha.basis`) and ``A`` is `ha.A`.
"""
function hessian_value_diag(ha::CoordinatesNormalSystemState, M::AbstractManifold, p, X)
    c = get_coordinates(M, p, X, ha.basis)
    return dot(c, ha.A, c)
end


# The objective here should be a LevenbergMarquardtLinearSurrogateObjective, but might be decorated as well, so for now lets not type it (yet?)
function solve!(dmp::DefaultManoptProblem{<:TangentSpace, <:NormalEquationsObjective}, cnss::CoordinatesNormalSystemState)
    # Update A and b
    TpM = get_manifold(dmp)
    M = base_manifold(TpM)
    p = base_point(TpM)
    neo = get_objective(dmp)
    get_linear_operator!(M, cnss.A, neo, p, cnss.basis)
    get_vector_field!(M, cnss.b, neo, p, cnss.basis)
    cnss.b .*= -1
    cnss.linsolve!(cnss.c, cnss.A, cnss.b)
    return cnss
end
