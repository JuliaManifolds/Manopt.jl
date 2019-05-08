
export approximateHessianForwardDifferences

doc"""
    approximateHessianForwardDifferences(p,x,ξ,[stepsize=2^(-14)],[retraction=exp])
Computes an approximate Hessian by finite difference of the gradient function.

A direction `ξ` in $ T_x\mathcal M$ is taken to get a second point $y = \operatorname{retr}_xs\xi\in M$
near `x` (step size $ s=\frac{\text{stepsize}}{\lVert\xi\rVert_x}$ away) to approximate the hessian by

$ \frac{1}{s}(P_{y\to x}\grad f(y) - \grad f(x)) $

# Input
- `p` - a [`GradientProblem`](@ref) providing the gradient ∇f and a [`Manifold`](@ref) `M`
- `x` - an [`MPoint`](@ref) on  `M`
- `ξ` - a direction [`TVector`](@ref) in $T_x\mathcal M$ of `x`.
- `stepsize` - ($2^(-14)$) a stepsize, i.e a length of $c=\frac{\text{stepsize}}{\lVert\xi\rVert_x}$ is used
- `retraction` - ([`exp`](@ref)) a [`retraction`](@ref)`(M,x,ξ,t)` to map the tangent back to =`M`.

# Output

"""
function approximateHessianForwardDifferences{GP <: GradientProblem, P <: MPoint, T <: TVector}(p::GradientProblem,x::P,ξ::T,stepsize::Float64=2^(-14), retraction::Function=exp)
    nξ = norm(M,x,ξ) # length of the direction
    if nξ < eps(Float64)
        return zeroTVector(p.M,x)
    end
    # have a real direction of size stepsize
    c = stepsize / nξ
    ∇fx = getGradient(p,x)
    y = retraction(p.M,x,ξ,c)
    ∇fy = getGradient(p,y)
    # transport ∇fy back to x and finite difference -> approximate Hessian
    return 1/c*( parallelTransport(M,y,x,∇fy) - ∇fx)
end
