#
# A conjugate Gradient algorithm implementation
#
export conjugateGradientDescent
export steepestCoefficient, HeestenesStiefelCoefficient, FletcherReevesCoefficient
export PolyakCoefficient, ConjugateDescentCoefficient, LiuStoreyCoefficient
export DaiYuanCoefficient, HagerZhangCoefficient
@doc doc"""
    conjugateGradientDescent(M, F, ∇F, x)
perform a conjugate gradient based descent $x_{k+1} = \exp_{x_k} s_k\delta_k$
whith different rules to compute the direction $\delta_k$ based on the last direction
$\delta_{k-1}$ and both gradients $\nabla f(x_k)$,$\nabla f(x_{k-1})$ are available.
Further, the step size $s_k$ may be refined by a line search.

# Input
* `M` : a manifold $\mathcal M$
* `F` : a cost function $F\colon\mathcal M\to\mathbb R$ to minimize
* `∇F`: the gradient $\nabla F\colon\mathcal M\to T\mathcal M$ of F
* `x` : an initial value $x\in\mathcal M$

# Optional
* `debug` : (off) a tuple `(f,p,v)` of a DebugFunction `f`
  that is called with its settings dictionary `p` and a verbosity `v`. Existing
  fields of `p` are updated during the iteration from (iter, x, xnew, stepSize).
* `directionUpdate` : ([`steepestCoefficient`](@ref), [`SimpleDirectionUpdateOptions()`](@ref))
  rule to update the descent direction δ based on `(M,x,δ,ξ,xnew,ξnew,[o])`
  a manifold, the (last) iterate, direction and gradient `x,δ,ξ` and current values
  `xnew,ξnew` as well as options `o::DirectionUpdateOptions`, empty by default).

  Available rules are: [`steepestCoefficient`](@ref),
  [`HeestenesStiefelCoefficient`](@ref), [`FletcherReevesCoefficient`](@ref),
  [`PolyakCoefficient`](@ref), [`ConjugateDescentCoefficient`](@ref),
  [`LiuStoreyCoefficient`](@ref), [`DaiYuanCoefficient`](@ref),
  [`HagerZhangCoefficient`](@ref).
* `lineSearch` : (`(p,lO) -> 1, lO::`[`LineSearchOptions`](@ref)`)`) A tuple `(lS,lO)`
  consisting of a line search function `lS` (called with two arguments, the
  problem `p` and the lineSearchOptions `lO`) with its LineSearchOptions `lO`.
  The default is a constant step size 1.
* `retraction` : (`exp`) a retraction(M,x,ξ) to use.
* `returnReason` : (`false`) whether or not to return the reason as second return
   value.
* `stoppingCriterion` : (`(i,ξ,x,xnew) -> ...`) a function indicating when to stop.
  Default is to stop if the norm of the gradient $\lVert \xi\rVert_x$ is less
  than $10^{-4}$ or the iterations `i` exceed 500.

# Output
* `xOpt` – the resulting (approximately critical) point of conjugateGradientDescent
* `reason` - if activated a String containing the stopping criterion stopping
  reason.
"""
function conjugateGradientDescent(M::mT,F::Function, ∇F::Function, x::P;
        directionUpdate::Tuple{Function,Options} =
            (steepestDirection, SimpleDirectionUpdateOptions()),
        lineSearch::Tuple{Function,Options}=
            ( (p::GradientProblem{mT}, o::LineSearchOptions) -> 1,
                SimpleLineSearchOptions(x) ),
        retraction::Function = exp,
        stoppingCriterion::Function = (i,ξ,x,xnew) -> (norm(M,x,ξ) < 10.0^-4 || i > 499, (i>499) ? "max Iter $(i) reached." : "critical point reached"),
        returnReason=false,
        kwargs... #especially may contain debug
    ) where {mT <: Manifold, P <: MPoint}
    # TODO Test Input
    p = GradientProblem(M,F,∇F)
    o = ConjugateGradientOptions(x,stoppingCriterion,retraction,lineSearch[1],lineSearch[2], directionUpdate[1], directionUpdate[2])
    # create default here to check if the user provided a debug and still have the typecheck
    debug::Tuple{Function,Dict{String,Any},Int}= (x::Dict{String,Any}->print(""),Dict{String,Any}(),0);
    kwargs=Dict(kwargs)
    if haskey(kwargs, :debug) # if a key is given -> decorate Options.
        debug = kwargs[:debug]
        o = DebugDecoOptions(o,debug[1],debug[2],debug[3])
    end
    x,r = conjugateGradientDescent(p,o)
    if returnReason
        return x,r;
    else
        return x;
    end
end
"""
    conjugateGradientDescent(problem,options)
performs a conjugateGradientDescent based on a [`GradientProblem`](@ref)
and corresponding [`ConjugateGradientOptions`](@ref).
"""
function conjugateGradientDescent(p::P, o::O) where {P <: GradientProblem, O <: Options}
    stop::Bool = false
    reason::String="";
    iter::Integer = 0
    x = getOptions(o).x0
    s = getOptions(o).lineSearchOptions.initialStepsize
    M = p.M
    ξ = gradF(p,x); # g_k in [HZ06]
    δ = -gradF(p,x); #d_k in [HZ06]
    β = 0;
    while !stop
        s = getStepsize(p,getOptions(o),x,s) # α_k in [HZ06]
        xnew = getOptions(o).retraction(M,x,-s*δ)
        ξnew = gradF(p,xnew) # g_k+1
        βnew = getOptions(o).directionUpdate(x,ξ,δ,xnew,ξnew,
            getOptions(o).directionUpdateOptions)
        δnew = - ξnew + βnew * parallelTransport(M,x,xnew,δ)
        (stop, reason) = evaluateStoppingCriterion(getOptions(o),iter,ξ,x,xnew)
        gradDescDebug(o,iter,x,xnew,ξ,s,reason);
        x=xnew;
        ξ = ξnew;
        δ = δnew;
        β = βnew;
    end
    return x,reason
end
#
#
# Direction Update rules
#
#
"""
    steepestCoefficient(M,x,ξ,δ,xnew,ξnew)
The simplest rule to update is to have no influence of the last direction and
hence return an update β of zero for all last gradients and directions `ξ,δ`,
attached at the last iterate `x` as well as the current gradient `ξnew` and iterate `xnew`

*See also*: [`conjugateGradientDescent`](@ref)
"""
steepestCoefficient(M::mT,
    x::P,ξ::T,δ::T,xnew::P,ξnew::T, o::DirectionUpdateOptions = SimpleDirectionUpdateOptions()) where {mT<:Manifold, P<:MPoint,T<:TVector} = 0.0
@doc doc"""
    HeestenesStiefelCoefficient(M,x,ξ,δ,xnew,ξnew)
Computes an update coefficient for the conjugate gradient method, where
`new` refers to $k+1$ based on

> M.R. Hestenes, E.L. Stiefel, Methods of conjugate gradients for solving linear systems,
> J. Research Nat. Bur. Standards, 49 (1952), pp. 409–436.

adapted to manifolds as follows: let $\nu_k = \xi_{k+1} - P_{x_k\to x_{k+1}}\xi_k$.
Then the update reads

$ \beta_k =
\frac{\langle \xi_{k+1}, \nu_k \rangle_{x_{k+1}} }
{ \langle P_{x_k\to x_{k+1}} \delta_k, \nu_k\rangle_{x_{k+1}} }.$

*See also*: [`conjugateGradientDescent`](@ref)
"""
function HeestenesStiefelCoefficient(M::mT,
    x::P,ξ::T,δ::T,xnew::P,ξnew::T,o::DirectionUpdateOptions = SimpleDirectionUpdateOptions()
        ) where {mT<:Manifold,P<:MPoint,T<:TVector}
    ξtr = parallelTransport(M,x,xnew,ξ)
    δtr = parallelTransport(M,x,xnew,δ)
    νk = ξnew-ξtr #notation from [HZ06]
    β = dot(M,xnew, ξnew,νk)/dot(M,xnew,δtr,νk)
    return max(0,β);
end
@doc doc"""
    FletcherReevesCoefficient(M,x,ξ,δ,xnew,ξnew)
Computes an update coefficient for the conjugate gradient method, where
`new` refers to $k+1$ based on

> R. Fletcher and C. Reeves, Function minimization by conjugate gradients,
> Comput. J., 7 (1964), pp. 149–154.

adapted to manifolds:

$ \beta_k =
\frac{\lVert \xi_{k+1}\rVert_{x_{k+1}}^2}{\lVert \xi_{k}\rVert_{x_{k}}^2}.$

*See also*: [`conjugateGradientDescent`](@ref)
"""
function FletcherReevesCoefficient(M::mT,
    x::P,ξ::T,δ::T,xnew::P,ξnew::T,o::DirectionUpdateOptions = SimpleDirectionUpdateOptions()
    ) where {mT<:Manifold,P<:MPoint,T<:TVector}
    return dot(M,xnew,ξnew,ξnew)/dot(M,x,ξ,ξ)
end
@doc doc"""
    PolyakCoefficient(M,x,ξ,δ,xnew,ξnew)
Computes an update coefficient for the conjugate gradient method, where
`new` refers to $k+1$ based on

> B. T. Polyak, The conjugate gradient method in extreme problems,
> USSR Comp. Math. Math. Phys., 9 (1969), pp. 94–112.

adapted to manifolds: let $\nu_k = \xi_{k+1} - P_{x_k\to x_{k+1}}\xi_k$.
Then the update reads

$ \beta_k =
\frac{ \langle \xi_{k+1}, \nu_k \rangle_{x_{k+1}} }
{\lVert \xi_k \rVert_{x_k} }.$

*See also*: [`conjugateGradientDescent`](@ref)
"""
function PolyakCoefficient(M::mT,
        x::P,ξ::T,δ::T,xnew::P,ξnew::T,o::DirectionUpdateOptions = SimpleDirectionUpdateOptions()
        ) where {mT<:Manifold,P<:MPoint,T<:TVector}
        ξtr = parallelTransport(M,x,xnew,ξ)
        νk = ξnew-ξtr #notation y from [HZ06]
        β = dot(M,xnew, ξnew,νk)/dot(M,x,ξ,ξ);
        return max(0,β);
end
@doc doc"""
    ConjugateDescentCoefficient(M,x,ξ,δ,xnew,ξnew)
Computes an update coefficient for the conjugate gradient method, where
`new` refers to $k+1$ based on

> R. Fletcher, Practical Methods of Optimization vol. 1: Unconstrained Optimization,
> John Wiley & Sons, New York, 1987.

adapted to manifolds:

$ \beta_k =
\frac{ \lVert \xi_{k+1} \rVert_{x_{k+1}}^2 }
{\langle -\delta_k,\xi_k \rangle_{x_k}}.$

*See also*: [`conjugateGradientDescent`](@ref)
"""
function ConjugateDescentCoefficient(M::mT,
        x::P,ξ::T,δ::T,xnew::P,ξnew::T,o::DirectionUpdateOptions = SimpleDirectionUpdateOptions()
    ) where {mT<:Manifold,P<:MPoint,T<:TVector}
    return dot(M,xnew,ξnew,ξnew)/dot(M,x,-δ,ξ)
end
@doc doc"""
    LiuStoreyCoefficient(M,x,ξ,δ,xnew,ξnew)
Computes an update coefficient for the conjugate gradient method, where
`new` refers to $k+1$ based on

> Y. Liu and C. Storey, Efficient generalized conjugate gradient algorithms, Part 1: Theory,
> J. Optim. Theory Appl., 69 (1991), pp. 129–137.

adapted to manifolds: with $\nu_k = \xi_{k+1} - P_{x_k\to x_{k+1}}\xi_k$ it reads

$ \beta_k =
\frac{ \langle \xi_{k+1},\nu_k \rangle_{x_{k+1}} }
{\langle -\delta_k,\xi_k \rangle_{x_k}}.$

*See also*: [`conjugateGradientDescent`](@ref)
"""
function LiuStoreyCoefficient(M::mT,
        x::P,ξ::T,δ::T,xnew::P,ξnew::T,o::DirectionUpdateOptions = SimpleDirectionUpdateOptions()
    ) where {mT<:Manifold,P<:MPoint,T<:TVector}
    ξtr = parallelTransport(M,x,xnew,ξ)
    νk = ξnew-ξtr #notation y from [HZ06]
    return dot(M,xnew, ξnew,νk)/dot(M,x,-δ,ξ)
end
@doc doc"""
    DaiYuanCoefficient(M,x,ξ,δ,xnew,ξnew)
Computes an update coefficient for the conjugate gradient method, where
`new` refers to $k+1$ based on

> Y. H. Dai and Y. Yuan, A nonlinear conjugate gradient method with a strong global convergence property,
> SIAM J. Optim., 10 (1999), pp. 177–182.

adapted to manifolds: with $\nu_k = \xi_{k+1} - P_{x_k\to x_{k+1}}\xi_k$ it reads

$ \beta_k =
\frac{ \lVert \xi_{k+1} \rVert_{x_{k+1}}^2 }
{\langle P_{x_k\to x_{k+1}}\delta_k, \nu_k \rangle_{x_{k+1}}}.$

*See also*: [`conjugateGradientDescent`](@ref)
"""
function DaiYuanCoefficient(M::mT,
        x::P,ξ::T,δ::T,xnew::P,ξnew::T,o::DirectionUpdateOptions = SimpleDirectionUpdateOptions()
    ) where {mT<:Manifold,P<:MPoint,T<:TVector}
    ξtr = parallelTransport(M,x,xnew,ξ);
    νk = ξnew-ξtr #notation y from [HZ06]
    δtr = parallelTransport(M,x,xnew,δ);
    return dot(M,xnew,ξnew,ξnew)/dot(M,x,δtr,νk);
end
@doc doc"""
    HagerZhangCoefficient(M,x,ξ,δ,xnew,ξnew)
Computes an update coefficient for the conjugate gradient method, where
`new` refers to $k+1$ based on

> W. W. Hager and H. Zhang, A new conjugate gradient method with guaranteed descent and an efficient line search,
> SIAM J. Optim, (16), pp. 170-192, 2005.

adapted to manifolds: with $\nu_k = \xi_{k+1} - P_{x_k\to x_{k+1}}\xi_k$ it reads

$ \beta_k = \Bigl\langle\nu_k -
\frac{ 2\lVert \nu_k\rVert_{x_{k+1}}^2 }{ \langle P_{x_k\to x_{k+1}}\delta_k, \nu_k \rangle_{x_{k+1}} }
P_{x_k\to x_{k+1}}\delta_k,
\frac{\xi_{k+1}}{ \langle P_{x_k\to x_{k+1}}\delta_k, \nu_k \rangle_{x_{k+1}} }
\Bigr\rangle_{x_{k+1}}.$

This methods includes a numerical stability proposed by those authors.

*See also*: [`conjugateGradientDescent`](@ref)
"""
function HagerZhangCoefficient(M::mT,
        x::P,ξ::T,δ::T,xnew::P,ξnew::T,o::DirectionUpdateOptions = SimpleDirectionUpdateOptions()
    ) where {mT<:Manifold,P<:MPoint,T<:TVector}
    ξtr = parallelTransport(M,x,xnew,ξ);
    νk = ξnew-ξtr #notation y from [HZ06]
    δtr = parallelTransport(M,x,xnew,δ);
    denom = dot(M,xnew,δtr,νk)
    νknormsq = dot(M,xnew,x,νk,νk)
    β = dot(M,xnew,νk,ξnew)/denom - 2*dot(M,xnew,νk,νk)*dot(M,xnew,δtr,ξnew)/denom^2;
    # Numerical stability from Manopt
    ξn = norm(M,xnew,ξnew)
    η = -1/( ξn*min(0.01,norm(M,x,ξ)) );
    return max(β,η);
end
