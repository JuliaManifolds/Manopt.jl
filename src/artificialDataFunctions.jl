#
# generate Artificial Data on several manifolds
#
#
#
export artificialInSARImage

"""
    artificialInSARImage(pts)
  generate an artificial InSAR image, i.e. phase valued data, of size `pts` x
  `pts` points.

  This example was introduced in the article
  > R. Bergmann, F. Laus, G. Steidl, A. Weinmann:
  > Second Order Differences of Cyclic Data and Applications in Variational Denoising,
  > SIAM Journal on Imaging Sciences, Vol. 7, No. 4, pp. 2916–2953, 2014.

~ ManifoldValuedImageProcessing.jl ~ R. Bergmann ~ 2016-12-05.
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
    k1 = v <= 1 ? 10*pi*Yr : 0.0
    # circle
    Xr = cosA*values[i] - sinA*values[j]
    Yr = cosA*values[j] + sinA*values[i]
    v = (Xr-midPoint[1])^2 + (Yr-midPoint[2])^2/radius^2
    k2 = v <= 1 ? 4.0*pi*(1-v) : 0.0
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
