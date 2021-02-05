A = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]

cost(::PowerManifold, p) = cost(p)
cost(X::Array{Matrix{Float64},1}) = -0.5 * norm(transpose(X[1]) * A * X[2])^2

function egrad(X::Array{Matrix{Float64},1})
    U = X[1]
    V = X[2]
    AV = A * V
    AtU = transpose(A) * U
    return [-AV * (transpose(AV) * U), -AtU * (transpose(AtU) * V)]
end
struct EGrad{T}
    A::Matrix{T}
end
Egrad(A::Matrix{T}) where {T} = EGrad{T}(A)
function (e::EGrad{Float64})(Y::Array{Matrix{Float64},1}, X::Array{Matrix{Float64},1})
    Y[1] .= - e.A * X[2] * (transpose(e.A * X[2]) * X[1])
    Y[2] .= - transpose(e.A) * X[1] * (transpose(X[1]) * e.A * X[2])
    return Y
end

rgrad(M::PowerManifold, p) = project(M, p, egrad(p))
struct RGrad{T}
    egrad::EGrad{T}
end
RGrad(A::Matrix{T},p) where {T} = RGrad{T}(EGrad{T}(A))
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

function eHess(X::Array{Matrix{Float64},1}, H::Array{Matrix{Float64},1})
    U = X[1]
    V = X[2]
    Udot = H[1]
    Vdot = H[2]
    AV = A * V
    AtU = transpose(A) * U
    AVdot = A * Vdot
    AtUdot = transpose(A) * Udot
    return [
        -(
            AVdot * transpose(AV) * U +
            AV * transpose(AVdot) * U +
            AV * transpose(AV) * Udot
        ),
        -(
            AtUdot * transpose(AtU) * V +
            AtU * transpose(AtUdot) * V +
            AtU * transpose(AtU) * Vdot
        ),
    ]
end
struct EHess{T}
    A::Matrix{T}
end
EHess(A::Matrix{T}) where {T} = EHess{T}(A)
function (e::EHess)(Y,X,H)
    Y[1] .= -e.A*H[2]*transpose(e.A*X[2])*X[1] - e.A*X[2] * transpose(e.A*H[2])*X[1] - e.A*X[2]*transpose(e.A*X[2])*H[1]
    Y[2] .= transpose(e.A)*H[1]*transpose(X[1])*e.A*X[2] + transpose(e.A)*X[1] * transpose(H[1])*e.A*X[2] + transpose(e.A)*X[1]*transpose(X[1])*e.A*H[2]
    return Y
end

function rhess(M::PowerManifold, p, X)
    eG = egrad(p)
    eH = eHess(p, X)
    return e2rHess.(Ref(M.manifold), p, X, eG, eH)
end
struct RHess{T}
    e_grad!::EGrad{T}
    e_hess!::EHess{T}
    G::Array{Matrix{T},1}
    H::Array{Matrix{T},1}
end
function RHess(A::Matrix{T},p) where {T}
    return RHess{T}(
        EGrad(A),
        EHess(A),
        [zeros(T,size(A,1),p) for _ ∈ 1:2],
        [zeros(T,size(A,1),p) for _ ∈ 1:2],
    )
end
function (r::RHess)(M::PowerManifold, Y, p, X)
    r.e_grad!(r.G, p)
    r.e_hess!(r.H, p, X)
    e2rhess!.(Ref(M.manifold), Y, p, X, r.G, r.H)
    return Y
end