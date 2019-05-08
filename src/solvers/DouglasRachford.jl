export DouglasRachford
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
> SIAM J. Imaging Sciences 9.3, pp. 901–937, 2016.
> doi: [10.1137/15M1052858](https://dx.doi.org/10.1137/15M1052858)

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
* `α` – (`(iter) -> 0.9`) relaxation of the step from old to new iterate, i.e.
  $x^{(k+1)} = g(α(k); x^{(k)}, t^{(k)})$, where $t^{(k)}$ is the result
  of the double reflection involved in the DR algorithm
* `R` – ([`reflection`](@ref)) method employed in the iteration
  to perform the reflection of `x` at the prox `p`.
* `stoppingCriterion` : ( [`stopWhenAny`](@ref)`( `[`stopAtIteration`](@ref)`(200), `[`stopChangeLess`](@ref)`(10.0^-5))` )
  a function `(p,o,i) -> s,r` indicating when to stop and what the reason is.

and the ones that are passed to [`decorateOptions`](@ref) for the decorators.

# Output
* `xOpt` – the resulting point of the Douglas Rachford algorithm
* `record` - if activated (using the `record` key, see [`RecordOptions`](@ref)
  an array containing the recorded values.
"""
function DouglasRachford(M::mT, F::Function, x::P, proxes::Array{Function,N} where N;
    λ::Function = (iter) -> 1.0, α::Function = (iter) -> 0.9,
    R = reflection,
    stoppingCriterion::Function = stopWhenAny( stopAtIteration(200), stopChangeLess(10.0^-5)),
    kwargs... #especially may contain decorator options
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

    o = decorateOptions(o; kwargs...)
    resultO = solve(p,o)
    if hasRecord(resultO)
        return getSolverResult(p,getOptions(resultO)), getRecord(resultO)
    end
    return getSolverResult(p,resultO)
end
function initializeSolver!(p::ProximalProblem,o::DouglasRachfordOptions)
    o.xOld = o.x
    o.mean = o.x[1];
    o.meanOld = o.mean
end
function doSolverStep!(p::ProximalProblem,o::DouglasRachfordOptions,iter)
    o.meanOld = o.mean
    # first prox or parallel Proxes
    pP = getProximalMap(p,o.λ(iter),o.x,1)
    o.x = o.R(p.M,pP,o.x);
    # relaxation
    o.x = geodesic(p.M,o.xOld,o.x,o.α(iter))
    o.xOld = o.x
    # second prox: Mean in parallel
    pP = getProximalMap(p,o.λ(iter),o.x,2)
    o.mean = pP[1] # store mean
    # reflect at mean
    o.x = o.R(p.M,pP,o.x)
 end
function getSolverResult(p::ProximalProblem,o::DouglasRachfordOptions)
    return o.mean
end

# overwrite defaults, since we store the result in the mean field
debug(p::ProximalProblem{M} where {M <: Manifold}, o::DouglasRachfordOptions,::Val{:Change}, iter::Int, out::IO=Base.stdout) =
  print(out,"Change: ",distance(p.M.manifold, o.mean, o.meanOld))
debug(p::ProximalProblem{M} where {M <: Manifold}, o::DouglasRachfordOptions,::Val{:Cost}, iter::Int, out::IO=Base.stdout) =
  print(out,"Cost: ", getCost(p,o.mean))

record(p::ProximalProblem{M} where {M <: Manifold}, o::DouglasRachfordOptions,::Val{:Iterate}, iter::Int) = o.mean
recordType(o::DouglasRachfordOptions,::Val{:Iterate}) = typeof(o.mean)
record(p::ProximalProblem{M} where {M <: Manifold}, o::DouglasRachfordOptions,::Val{:Cost}, iter::Int) = getCost(p,o.mean)
recordType(o::DouglasRachfordOptions,::Val{:Cost}) = Float64
record(p::ProximalProblem{M} where {M <: Manifold}, o::DouglasRachfordOptions, ::Val{:Change}, iter::Int) = distance(p.M.manifold, o.mean, o.meanOld)
recordType(o::DouglasRachfordOptions,::Val{:Change}) = Float64
