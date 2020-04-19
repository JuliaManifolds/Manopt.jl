#
# generate Artificial Data on several manifolds
#
#
#
import LinearAlgebra: I, Diagonal
export artificial_S1_signal, artificial_S1_slope_signal, artificialIn_SAR_image
export artificial_SPD_image, artificial_SPD_image2
export artificial_S2_whirl_image, artificial_S2_whirl_patch
export artificial_S2_rotation_image
export artificial_S2_whirl_patch, artificial_S2_lemniscate

@doc raw"""
    artificialIn_SAR_image([pts=500])
generate an artificial InSAR image, i.e. phase valued data, of size `pts` x
`pts` points.
"""
function artificialIn_SAR_image(pts::Integer)
  # variables
  # rotation of ellipse
  aEll = 35.0; cosE = cosd(aEll); sinE = sind(aEll)
  aStep = 45.0; cosA = cosd(aStep); sinA = sind(aStep)
  # main and minor axis of the ellipse
  axes_inv = [6,25]
  # values for the hyperboloid
  mid_point = [0.275;0.275]
  radius = 0.18
  values = [range(-0.5,0.5,length = pts)...]
  # Steps
  aSteps = 60.0; cosS = cosd(aSteps); sinS = sind(aSteps)
  l = 0.075
  midP = [-.475,-.0625];#.125, .55]
  img = zeros(Float64,pts,pts)
  for j in eachindex(values), i in eachindex(values)
    # ellipse
    Xr = cosE*values[i] - sinE*values[j]
    Yr = cosE*values[j] + sinE*values[i]
    v = axes_inv[1]*Xr^2 + axes_inv[2]*Yr^2
    k1 = v <= 1.0 ? 10.0*pi*Yr : 0.0
    # circle
    Xr = cosA*values[i] - sinA*values[j]
    Yr = cosA*values[j] + sinA*values[i]
    v = ( (Xr-mid_point[1])^2 + (Yr-mid_point[2])^2 )/radius^2
    k2 = v <= 1.0 ? 4.0*pi*(1.0-v) : 0.0
    #
    Xr = cosS*values[i] - sinS*values[j]
    Yr = cosS*values[j] + sinS*values[i]
    k3=0.0
    for m=1:8
      k3 += (abs(Xr+midP[1]+m*l)+abs(Yr+midP[2]+m*l)) <= l ? 2*pi*(m/8) : 0.0
    end
    img[i,j] = mod(k1+k2+k3-pi,2*pi)+pi
  end
  return img
end

@doc raw"""
    artificial_S1_slope_signal([pts=500, slope=4.])

Creates a Signal of (phase-valued) data represented on the
[Circle](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold)` `[Manifold](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold) with increasing slope.

# Optional
* `pts` – (`500`) number of points to sample the function.
* `slope` – (`4.0`) initial slope that gets increased afterwards
"""
function artificial_S1_slope_signal(pts::Integer = 500, slope::Float64=4.)
    t = range(0., 1., length=pts)
    f = zero(t)
    f[ t .<= 1/6] .= -π/2 .+ slope * π/8 * t[ t .<= 1/6]
    # In the following terms, the first max
    f[(1/6 .< t) .& (t .<= 1/3)] .= max(f[f .!= 0]...) .- slope * π/4 * 1/6 .+ slope * π/4 .* t[ (1/6 .< t) .& ( t.<=1/3 )]
    f[(1/3 .< t) .& (t .<= 1/2)] .= max(f[f .!= 0]...) .- slope * π/2 * 1/3 .+ slope * π/2 * t[ (1/3 .< t) .& ( t .<= 1/2)]
    f[(1/2 .< t) .& (t .<= 2/3)] .= max(f[f .!= 0]...) .- slope * π * 1/2 .+ slope * π * t[(1/2 .< t) .& (t .<= 2/3)]
    f[(2/3 .< t) .& (t .<= 5/6)] .= max(f[f .!= 0]...) .- slope * 2*π * 2/3 .+ slope * 2 * π * t[(2/3 .< t) .& (t .<= 5/6)]
    f[ 5/6 .< t] .= max(f[f .!= 0]...) .- slope * 4 * π * 5/6 .+ slope * 4 * π * t[ 5/6 .< t]
    return mod.(f .+ Float64(π), Ref(2*π)) .- Float64(π)
end

@doc raw"""
    artificial_S1_signal([pts=500])

generate a real-valued signal having piecewise constant, linear and quadratic
intervals with jumps in between. If the resulting manifold the data lives on,
is the [Circle](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold) the data is also wrapped
to $[-\pi,\pi)$.

# Optional
* `pts` – (`500`) number of points to sample the function
"""
function artificial_S1_signal(pts::Integer=500)
  t = range(0., 1., length=pts)
  f = artificial_S1_signal.(t)
  return mod.(f .+ Float64(π), Ref(2*π)) .- Float64(π)
end
@doc raw"""
    artificial_S1_signal(x)
evaluate the example signal $f(x), x ∈  [0,1]$,
of phase-valued data introduces in Sec. 5.1 of

> Bergmann, Laus, Steidl, Weinmann, Second Order Differences of Cyclic Data and
> Applications in Variational Denoising, SIAM J. Imaging Sci., 7(4), 2916–2953, 2014.
> doi: [10.1137/140969993](https://dx.doi.org/10.1137/140969993)

for values outside that intervall, this Signal is `missing`.
"""
function artificial_S1_signal(x::Real)
    if x < 0
        y = missing
    elseif x <= 1/4
        y = - 24 * π * (x-1/4)^2  +  3/4 * π
    elseif x <= 3/8
        y = 4 * π * x  -  π / 4
    elseif x <= 1/2
        y = - π * x - 3*π/8;
    elseif x <= (3*0 + 19)/32
        y = - (0+7)/8*π
    elseif x <= (3*1 + 19)/32
        y = - (1+7)/8*π
    elseif x <= (3*2 + 19)/32
        y = - (2+7)/8*π
    elseif x <= (3*3 + 19)/32
        y = - (3+7)/8*π
    elseif x <= 1
        y = 3 / 2 * π * exp(8-1/(1-x))  -  3/4*π
    else
        y = missing
    end
    return y
end
@doc raw"""
    artificial_S2_whirl_image([pts=64])
generate an artificial image of data on the 2 sphere,

# Arguments
* `pts` – (`64`) size of the image in `pts`$\times$`pts` pixel.
"""
function artificial_S2_whirl_image(pts::Int=64)
  M = Sphere(2)
  img = artificial_S2_rotation_image(pts, (0.5,0.5) )
  # Set WhirlPatches
  sc = pts/64
  patchSizes = floor.( sc.* [9,9,9,9,11,11,11,15,15,15,17,21] )
  patchCenters = Integer.( floor.(
    sc.* [ [35,7] [25,41] [32,25] [7,60] [10,5] [41,58] [11,41] [23,56] [38,45] [16,28] [55,42] [51,16] ]
  ) )
  patchSigns = [1,1,-1,1,-1,-1,1,1,-1,-1,1,-1]
  for i=1:length(patchSizes)
    pS = Integer(patchSizes[i])
    pSH = Integer( floor( (patchSizes[i]-1)/2 ) )
    pC = patchCenters[:,i]
    pC = max.(Ref(1), pC.-pS) .+pS
    pSgn = patchSigns[i]
    s = pS%2==0 ? 1 : 0
    r = [ pC[1] .+ ( -pSH:(pSH+s) ), pC[2] .+ ( -pSH:(pSH+s) ) ]
    patch = artificial_S2_whirl_patch(pS)
    if pSgn==-1 # opposite ?
      patch = -patch
    end
    img[ r... ] = patch
  end
  return img
end
@doc raw"""
    artificial_S2_rotation_image([pts=64, rotations=(.5,.5)])
creates an image with a rotation on each axis as a parametrization.

# Optional Parameters
* `pts` – (`64`) number of pixels along one dimension
* `rotations` – (`(.5,.5)`) number of total rotations performed on the axes.
"""
function artificial_S2_rotation_image(pts::Int=64,rotations::Tuple{Float64,Float64}=(.5,.5))
  M = Sphere(2)
  img = fill(zeros(3),pts,pts)
  north = [1.,0.,0.]
  Rxy(a) = [cos(a) -sin(a) 0.; sin(a) cos(a) 0.; 0. 0. 1]
  Rxz(a) = [cos(a) 0. -sin(a); 0. 1. 0.; sin(a) 0. cos(a)]
  for i=1:pts
    for j=1:pts
      x = i/pts*2π*rotations[1];
      y = j/pts*2π*rotations[2];
      img[i,j] =  Rxy(x+y)*Rxz(x-y)*[0,0,1]
    end
  end
  return img
end

@doc raw"""
    artificial_S2_whirl_patch([pts=5])

create a whirl within the `pts`$\times$`pts` patch of
[Sphere](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/sphere.html)(@ref)`(2)`-valued image data.

# Optional Parameters
* `pts` – (`5`) size of the patch. If the number is odd, the center is the north
  pole.
"""
function artificial_S2_whirl_patch(pts::Int=5)
  patch = fill( [0.,0.,-1.], pts, pts)
  scaleFactor = sqrt( (pts-1)^2 / 2 )*3/π;
  for i=1:pts
    for j=1:pts
      if i!=(pts+1)/2 || j != (pts+1)/2
        α = atan( (j-(pts+1)/2), (i-(pts+1)/2) );
        β = sqrt( (j-(pts+1)/2)^2 + (i-(pts+1)/2)^2 )/scaleFactor;
        patch[i,j] = [sin(α)*sin(π/2-β), -cos(α)*sin(π/2-β), cos(π/2-β)]
      end
    end
  end
  return patch
end
@doc raw"""
    artificial_SPD_image([pts=64, stepsize=1.5])

create an artificial image of symmetric positive definite matrices of size
`pts`$\times$`pts` pixel with a jump of size `stepsize`.
"""
function artificial_SPD_image(pts::Int=64, stepsize = 1.5)
  r = range(0, stop = 1-1/pts, length=pts)
  v1 = abs.(2*pi .* r .- pi)
  v2 = pi .* r;
  v3 = range(0, stop = 3*(1-1/pts), length = 2*pts);
  data = fill( Matrix{Float64}(I,3,3), pts, pts )
  for row = 1:pts
    for col = 1:pts
      A = [cos(v1[col]) -sin(v1[col]) 0. ; sin(v1[col]) cos(v1[col]) 0. ; 0. 0. 1.]
      B = [1. 0. 0. ; 0. cos(v2[row]) -sin(v2[row]) ; 0. sin(v2[row]) cos(v2[row]) ]
      C = [ cos(v1[mod(col-row,pts)+1]) 0 -sin(v1[mod(col-row,pts)+1]);
           0. 1. 0.;
           sin(v1[mod(col-row,pts)+1]) 0. cos(v1[mod(col-row,pts)+1]) ]
      scale = [ 1 + stepsize/2 * ( (row + col) > pts ? 1 : 0)
                1 + v3[row + col] - stepsize * ( col > pts/2 ? 1 : 0)
                4 - v3[row + col] + stepsize * ( row > pts/2 ? 1 : 0) ]
      data[row, col] = A * B * C * Diagonal(scale) * C' * B' * A'
    end
  end
  return data
end
@doc raw"""
    artificial_SPD_image2([pts=64, fraction=.66])

create an artificial image of symmetric positive definite matrices of size
`pts`$\times$`pts` pixel with right hand side `fraction` is moved upwards.
"""
function artificial_SPD_image2(pts=64, fraction = 0.66)
  Zl = 4. * Matrix{Float64}(I,3,3)
  # create a first matrix
  α = 2. * π/3;
  β = π/3;
  B = [  1. 0. 0. ; 0. cos(β) -sin(β) ; 0. sin(β) cos(β)  ]
  A = [  cos(α) -sin(α) 0. ; sin(α) cos(α) 0. ; 0. 0. 1.  ]
  Zo = A * B * Diagonal( [2., 4., 8.] ) * B' * A'
  # create a second matrix
  α = -4. * π/3;
  β = -π/3;
  B = [  1. 0. 0. ; 0. cos(β) -sin(β) ; 0. sin(β) cos(β)  ]
  A = [  cos(α) -sin(α) 0. ; sin(α) cos(α) 0. ; 0. 0. 1.  ]
  Zt = A * B * Diagonal( [ 8. / sqrt(2.), 8., sqrt(2.) ] ) * B' * A'
  data = fill( Matrix{Float64}(I,3,3) , pts, pts )
  M = SymmetricPositiveDefinite(3)
  for row = 1 : pts
    for col = 1 : pts
      # (a) from Zo a part to Zt
      C = Zo;
      if ( row > 1) # in X direction
        C = exp(M, C, log(M,C,Zt),
              (row - 1)/(2*(pts-1)) + ( ( row > fraction * pts) ? 1/2 : 0. )
            )
      end
      if ( col > 1) # and then in Y direction
        C = exp(M, C, vector_transport_to(M, Zo, log(M,Zo,Zl), C, ParallelTransport()), (col - 1.)/(pts-1) )
      end
      data[row,col] = C
    end
  end
  return data
end

@doc raw"""
    artificial_S2_lemniscate(p [,pts=128,a=π/2,interval=[0,2π])

generate a Signal on the [Sphere](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/sphere.html) $\mathbb S^2$ by creating the
[Lemniscate of Bernoulli](https://en.wikipedia.org/wiki/Lemniscate_of_Bernoulli)
in the tangent space of `p` sampled at `pts` points and use `exp` to get a
signal on the [Sphere](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/sphere.html).

# Input
* `p` – the tangent space the Lemniscate is created in
* `pts` – (`128`) number of points to sample the Lemniscate
* `a` – (`π/2`) defines a half axis of the Lemniscate to cover a
   half sphere.
* `interval` – (`[0,2*π]`) range to sample the lemniscate at, the default value
  refers to one closed curve
"""
function artificial_S2_lemniscate(p,pts::Integer=128, a::Float64=π/2.,
    interval::Array{Float64,1}=[0.,2.0*π])
    return artificial_S2_lemniscate.(Ref(p),range(interval[1],interval[2],length=pts), a)
end
@doc raw"""
    artificial_S2_lemniscate(p,t; a=π/2)

generate a point from the signal on the [Sphere](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/sphere.html) $\mathbb S^2$ by
creating the [Lemniscate of Bernoulli](https://en.wikipedia.org/wiki/Lemniscate_of_Bernoulli)
in the tangent space of `p` sampled at `t` and use èxp` to obtain a point on
the [Sphere](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/sphere.html).

# Input
* `p` – the tangent space the Lemniscate is created in
* `t` – value to sample the Lemniscate at

# Optional Values
 * `a` – (`π/2`) defines a half axis of the Lemniscate to cover a
   half sphere.
"""
function artificial_S2_lemniscate(p,t::Float64, a::Float64=π/2.)
    M = Sphere(2)
    tP = 2.0*Float64(p[1]>=0.)-1. # Take north or south pole
    base = [0.,0.,tP]
    xc = a * (cos(t)/(sin(t)^2+1.))
    yc = a * (cos(t)*sin(t)/(sin(t)^2+1.))
    tV = vector_transport_to(M,base,[xc,yc,0.],p, ParallelTransport())
    return exp(M,p,tV)
end
