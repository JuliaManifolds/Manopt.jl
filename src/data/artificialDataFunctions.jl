#
# generate Artificial Data on several manifolds
#
#
#
export artificialInSARImage, artificialS2WhirlImage, artificialS2RotationsImage
export artificialS2WhirlPatch

"""
    artificialInSARImage(pts)
generate an artificial InSAR image, i.e. phase valued data, of size `pts` x
`pts` points.

This example data was introduced in the article
> R. Bergmann, F. Laus, G. Steidl, A. Weinmann:
> Second Order Differences of Cyclic Data and Applications in Variational Denoising,
> SIAM Journal on Imaging Sciences, Vol. 7, No. 4, pp. 2916–2953, 2014.
"""
function artificialInSARImage(pts::Integer)::Array{Float64,2}
  # variables
  # rotation of ellipse
  aEll = 35.0; cosE = cosd(aEll); sinE = sind(aEll)
  aStep = 45.0; cosA = cosd(aStep); sinA = sind(aStep)
  # main and minor axis of the ellipse
  axes_inv = [6,25]
  # values for the hyperboloid
  midPoint = [0.275;0.275]
  radius = 0.18
  values = linspace(-0.5,0.5,pts)
  # Steps
  aSteps = 60.0; cosS = cosd(aSteps); sinS = sind(aSteps)
  l = 0.075
  midP = [-.475,-.0625];#.125, .55]
  img = zeros(Float64,pts,pts)
  @inbounds @fastmath for j in eachindex(values), i in eachindex(values)
    # ellipse
    Xr = cosE*values[i] - sinE*values[j]
    Yr = cosE*values[j] + sinE*values[i]
    v = axes_inv[1]*Xr^2 + axes_inv[2]*Yr^2
    k1 = v <= 1.0 ? 10.0*pi*Yr : 0.0
    # circle
    Xr = cosA*values[i] - sinA*values[j]
    Yr = cosA*values[j] + sinA*values[i]
    v = ( (Xr-midPoint[1])^2 + (Yr-midPoint[2])^2 )/radius^2
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

@doc doc"""
    artificialS2WhirlImage()
generate an artificial image of data on the 2 sphere,

# Optional Parameters
* `pts` : (`64`)size of the image in `pts`$\times$`pts` pixel.
"""
function artificialS2WhirlImage(;pts::Int=64)
  M = Sphere(2);
  N = Power(M, (pts,pts) );
  # background - default rotations
  img = artificialS2RotationsImage(;pts=pts, rotations=(0.5,0.5) )
  # Set WhirlPatches
  sc = pts/64;
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
    s = pS%2==0 ? 1 : 0;
    r = [ pC[1] .+ ( -pSH:(pSH+s) ), pC[2] .+ ( -pSH:(pSH+s) ) ]
    patch = artificialS2WhirlPatch(;pts=pS);
    if pSgn==-1 # opposite ?
      patch = PowPoint( opposite.(Ref(M), getValue(patch)) )
    end
    img[ r... ] = patch
  end
  return img
end
@doc doc"""
    artificialS2Rot()
creates an image with a rotation on each axis as a parametrization.

# Optional Parameters
* `pts` : (`64`) number of pixels along one dimension
* `roations` : (`(.5,.5)`) number of total rotations performed on the axes.
"""
function artificialS2RotationsImage(;pts::Int=64,rotations::Tuple{Float64,Float64}=(0.5,0.5))
  M = Sphere(2)
  N = Power(M, (pts,pts) )
  img = PowPoint(Matrix{SnPoint}(undef,pts,pts))
  north = SnPoint([1.,0.,0.])
  Rxy(a) = [cos(a) -sin(a) 0.; sin(a) cos(a) 0.; 0. 0. 1]
  Rxz(a) = [cos(a) 0. -sin(a); 0. 1. 0.; sin(a) 0. cos(a)]
  for i=1:pts
    for j=1:pts
      x = i/pts*2π*rotations[1];
      y = j/pts*2π*rotations[2];
      img[i,j] = SnPoint( Rxy(x+y)*Rxz(x-y)*[0,0,1] )
    end
  end
  return img
end

@doc doc"""
    artificialS2WhirlImage()
create a whirl within the ptsxpts patch

# Optional Parameters
* `pts` : (`5`) size of the patch. If the number is odd, the center is the north pole.
"""
function artificialS2WhirlPatch(;pts::Int=5)
  M = Sphere(2)
  N = Power(M, (pts,pts) )
  patch = PowPoint( fill(SnPoint([0.,0.,-1.]),pts,pts))
  scaleFactor = sqrt( (pts-1)^2 / 2 )*3/π;
  for i=1:pts
    for j=1:pts
      if i!=(pts+1)/2 || j != (pts+1)/2
        # direction within the patch⁠
        α = atan( (j-(pts+1)/2), (i-(pts+1)/2) );
        # scaled length
        β = sqrt( (j-(pts+1)/2)^2 + (i-(pts+1)/2)^2 )/scaleFactor;
        patch[i,j] = SnPoint([sin(α)*sin(π/2-β), -cos(α)*sin(π/2-β), cos(π/2-β)])
      end
    end
  end
  return patch
end
