using Manifolds
A = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]

f(M::PowerManifold, p) = -0.5 * norm(transpose(p[M, 1]) * A * p[M, 2])^2

function egrad(M::PowerManifold, X::Array)
    U = X[M, 1]
    V = X[M, 2]
    AV = A * V
    AtU = transpose(A) * U
    AR = similar(X)
    AR[:, :, 1] .= -AV * (transpose(AV) * U)
    AR[:, :, 2] .= -AtU * (transpose(AtU) * V)
    return AR
end
struct EGrad{T,TM} <: Function
    M::TM
    A::Matrix{T}
end
function (e::EGrad)(Y::Array, X::Array)
    U = X[e.M, 1]
    V = X[e.M, 2]
    AV = A * V
    AtU = transpose(A) * U
    view(Y, :, :, 1) .= -AV * (transpose(AV) * U)
    view(Y, :, :, 2) .= -AtU * (transpose(AtU) * V)
    return Y
end

rgrad(M::PowerManifold, p) = project(M, p, egrad(M, p))
struct RGrad{T,TM} <: Function
    egrad::EGrad{T,TM}
end
function RGrad(M::PowerManifold, A::Matrix{T}) where {T}
    return RGrad{T,typeof(M)}(EGrad{T,typeof(M)}(M, A))
end
function (r::RGrad)(M::PowerManifold, X, p)
    return project!(M, X, p, r.egrad(X, p))
end

function e2rHess(M::Grassmann, p, X, e_grad, e_hess)
    return project(M, p, project(M, p, e_hess) - X * (p' * e_grad))
end
function e2rhess!(M::Grassmann, Y, p, X, e_grad, e_Hess)
    project!(M, Y, p, e_Hess)
    Y .-= X * (p' * e_grad)
    return project!(M, Y, p, Y)
end

function eHess(M::AbstractManifold, X::Array{Float64,3}, H::Array{Float64,3})
    U = X[M, 1]
    V = X[M, 2]
    Udot = H[M, 1]
    Vdot = H[M, 2]
    AV = A * V
    AtU = transpose(A) * U
    AVdot = A * Vdot
    AtUdot = transpose(A) * Udot
    R = similar(X)
    #! format: off
    view(R, :, :, 1) .= -(
            AVdot * transpose(AV) * U +
            AV * transpose(AVdot) * U +
            AV * transpose(AV) * Udot
        )
    view(R, :, :, 2) .= -(
            AtUdot * transpose(AtU) * V +
            AtU * transpose(AtUdot) * V +
            AtU * transpose(AtU) * Vdot
        )
    #! format: on
    return R
end
struct EHess{T,TM} <: Function
    M::TM
    A::Matrix{T}
end
function (e::EHess)(Y, X, H)
    U = X[e.M, 1]
    V = X[e.M, 2]
    Udot = H[e.M, 1]
    Vdot = H[e.M, 2]
    AV = e.A * V
    AtU = transpose(e.A) * U
    AVdot = e.A * Vdot
    AtUdot = transpose(e.A) * Udot
    #! format: off
    view(Y, :, :, 1) .= -AVdot * transpose(AV) * U - AV * transpose(AVdot) * U - AV * transpose(AV) * Udot
    view(Y, :, :, 2) .= AtUdot * transpose(AtU) * V + AtU * transpose(AtUdot) * V + AtU * transpose(AtU) * Vdot
    #! format: on
    return Y
end

function rhess(M::PowerManifold, p, X)
    eG = egrad(M, p)
    eH = eHess(M, p, X)
    Ha = similar(p)
    for i in 1:2
        e2rhess!(
            M.manifold,
            view(Ha, :, :, i),
            view(p, :, :, i),
            view(X, :, :, i),
            view(eG, :, :, i),
            view(eH, :, :, i),
        )
    end
    return Ha
end
struct RHess{T,TM} <: Function
    e_grad!::EGrad{T,TM}
    e_hess!::EHess{T,TM}
    G::Array{T,3}
    H::Array{T,3}
end
function RHess(M::AbstractManifold, A::Matrix{T}, p) where {T}
    return RHess{T,typeof(M)}(
        EGrad(M, A), EHess(M, A), zeros(T, size(A, 1), p, 2), zeros(T, size(A, 1), p, 2)
    )
end
function (r::RHess)(M::PowerManifold, Y, p, X)
    r.e_grad!(r.G, p)
    r.e_hess!(r.H, p, X)
    for i in 1:2
        e2rhess!(
            M.manifold,
            view(Y, :, :, i),
            view(p, :, :, i),
            view(X, :, :, i),
            view(r.G, :, :, i),
            view(r.H, :, :, i),
        )
    end
    return Y
end
