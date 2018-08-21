#
#
# For a proximal Problem with at least two proximal maps one can define the
# following douglas rachford algorithm
#
#
export DouglasRachford, DRDebug
@doc doc"""
     DouglasRachford(M, F, proxMaps, x)
Computes the Douglas-Rachford algorithm on the manifold $\mathcal M$, initial
data $x_0$ and the (two) proximal maps `proxMaps`.

For $k>2$ proximal
maps the problem is reformulated using the parallelDouglasRachford: a vectorial
proximal map on the power manifold $\mathcal M^k$ and the proximal map of the
set that identifies all entries again, i.e. the Karcher mean.

For details see
> R. Bergmann, J. Persch, G. Steidl: A Parallel Douglas–Rachford Algorithm for
> Minimizing ROF-like Functionals on Images with Values in Symmetric Hadamard
> Manifolds.
> SIAM J. Imaging Sciences 9.3, pp. 901–937, 2016. doi: 10.1137/15M1052858

# Input
* `M` – a Riemannian Manifold $\mathcal M$
* `F` – a cost function consisting of a sum of cost functions
* `proxes` – functions of the form `(λ,x)->...` performing a proximal map,
  where `⁠λ` denotes the proximal parameter, for each of the summands of `F`.
* `x0` – initial data $x_0\in\mathcal M$

# Optional values
the default parameter is given in brackets
* `λ` – (`(iter) -> 1.0`) function to provide the value for the proximal parameter
  during the calls
* `α` – ('(iter) -> 0.9') relaxation of the step from old to new iterate, i.e.
  $x^{(k+1)} = g(α(k); x^{(k)}, t^{(k)})$, where $t^{(k)}$ is the result
  of the double reflection involved in the DR algorithm
* `R` – (`reflection`) method employed in the iteration
  to perform the reflection of `x` at the prox `p`.
* `returnReason` : ( `false` ) whether or not to return the reason as second return
  value.
* `stoppingCriterion` : ( `(i,x,xnew,λ) -> ...` ) a function indicating when to stop.
  Default is to stop if the norm of the iterates change $d_{\mathcal M}(x,x_{\text{new}})$ is less
  than $10^{-4}$ or the iterations `i` exceed 500.
"""
function DouglasRachford(M::mT, F::Function, x::P, proxes::Array{Function,N} where N;
    λ::Function = (iter) -> 1.0, α::Function = (iter) -> 0.9,
    R = reflection,
    stoppingCriterion::Function = (i,x,xnew,λ) -> (distance(M,x,xnew) < 10.0^-4 || i > 499, (i>499) ? "max Iter $(i) reached." : "Minimal change small enough."),
    returnReason=false,
    kwargs... #especially may contain debug
    ) where {mT <: Manifold, P <: MPoint}
    if length(proxes) < 2
        throw(
         ErrorException("Less than two proximal maps provided, the (parallel) Douglas Rachford requires (at least) two proximal maps.")
        );
    elseif length(proxes==2)
        lM = M;
        prox1 = proxes[1]
        prox2 = proxes[2]
    else # more than 2 -> parallelDouglasRachford
        k=length(proxes)
        lM = Power(M,k)
        prox1 = (λ,x) -> [proxes[i](λ,x[i]) for i in 1:k]
        prox2 = (λ,x) -> fill(mean(M,getValue(x)),k)
    end
    p = ProximalProblem(M,F,[prox1 prox2])
    o = DouglasRachfordOptions(x, stoppingCriterion, reflection, λ, α)
    # create default here to check if the user provided a debug and still have the typecheck
    debug::Tuple{Function,Dict{String,Any},Int}= (x::Dict{String,Any}->print(""),Dict{String,Any}(),0);
    kwargs=Dict(kwargs)
    if haskey(kwargs, :debug) # if a key is given -> decorate Options.
        debug = kwargs[:debug]
        o = DebugDecoOptions(o,debug[1],debug[2],debug[3])
    end
    x,r = DouglasRachford(p,o)
    if returnReason
        return x,r
    else
        return x
    end
end
"""
    DouglasRachford(p,o)
perform a Douglas Rachford algorithm based on the [`ProximalProblem`](@ref)` p`
and the [`DouglasRachfordOptions`](@ref)` o`.
"""
function DouglasRachford(p::ProximalProblem,o::DouglasRachfordOptions)
    if length(p.proximalMaps) != 2
        throw( ErrorException("Douglas-Rachford requires exactely two proximal maps. The problem provides $(length(p.proximalMaps))"))
    end
    x = getOptions(o).x0; newx = x;
    M = p.M
    λ = getOptions(o).λ; α = getOptions(o).α; R = getOptions(o).R
    stop=false; iter=0;
    while !stop
        iter = iter+1;
        # Reflect at the first prox
        p1 = getProximalMap(p,λ(iter),x,1)
        xR = R(M,p1,x);
        # Reflect at second prox
        p2 = getProximalMap(p,λ(iter),xR,2)
        t = R(M,xR,p2)
        # relaxation
        xnew = geodesic(M,x,xnew,α(iter))
        stop, reason = evaluateStoppingCriterion(o,iter,x,xnew,λ)
        # Debug?
        DRDebug(o,iter,x,xnew,reason)
        x = xnew
    end
    return x,reason
end
function DRDebug(o::O,iter::Int,x::MP,xnew::MP,reason::String) where {O <: Options, MP <: MPoint}
    if getOptions(o) != o
        DRDebug(getOptions(o),iter,x,xnew,reason)
    end
end
function DRDebug(o::D,iter::Int,x::MP,xnew::MP,reason::String) where {D <: DebugDecoOptions, MP <: MPoint}
    # decorate
    d = o.debugOptions;
    # Update values for debug
    if haskey(d,"x")
        d["x"] = x;
    end
    if haskey(d,"xnew")
        d["xnew"] = xnew;
    end
    if haskey(d,"Iteration")
        d["Iteration"] = iter;
    end
    o.debugFunction(d);
    if getVerbosity(o) > 2 && length(reason) > 0
        print(reason)
    end
end
