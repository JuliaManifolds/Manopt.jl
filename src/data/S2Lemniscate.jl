export S2Lemniscate
@doc doc"""
    S2Lemniscate(p,pts; interval=[0,2π], a=π/2)
generate a Signal on the 2-sphere $\mathbb S^2$ by creating the Lemniscate of
Bernoulli in the tangent space of `p` sampled at `pts` points.

# Input
* `p` – the tangent space the Lemniscate is created in
* `pts` – number of points to sample the Lemniscate

# Optional Values
`* interval` – (`[0,2*π]`) range to sample the lemniscate at, the default value
  refers to one closed curve
 * `a` – (`π/2`) defines a half axis of the Lemniscate to cover a
   half sphere.
"""
function S2Lemniscate(p::SnPoint,pts::Integer;
    interval::Array{Float64,1}=[0.,2.0*π], a::Float64=π/2.)
    return S2Lemniscate.(Ref(p),range(interval[1],stop=interval[2],length=pts); a=a)
end
@doc doc"""
    S2Lemniscate(p,t; interval=[0,2π], a=π/2)
generate a Signal on the 2-sphere $\mathbb S^2$ by creating the Lemniscate of
Bernoulli in the tangent space of `p` sampled at `pts` points.

# Input
* `p` – the tangent space the Lemniscate is created in
* `t` – value to sample the Lemniscate at

# Optional Values
 * `a` – (`π/2`) defines a half axis of the Lemniscate to cover a
   half sphere.
"""
function S2Lemniscate(p::SnPoint,t::Float64; a::Float64=π/2.)
    M = Sphere(2);
    tP = 2.0*Float64(getValue(p)[1]>=0.)-1. # Take north or south pole
    base = SnPoint([0.,0.,tP]);
    xc = a * (cos(t)/(sin(t)^2+1.));
    yc = a * (cos(t)*sin(t)/(sin(t)^2+1.));
    tV = parallelTransport(M,base,p,SnTVector([xc,yc,0.]))
    return exp(M,p,tV)
  end
