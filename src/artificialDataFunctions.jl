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
  aEll = 35.0/180.0*pi
  aStep = 45.0/180.0*pi
  # main and minor axis of the ellipse
  axes = [1/6,1/25]
  # values for the hyperboloid
  midPoint = 0.275*[1,1]
  radius = 0.18
  values = collect(linspace(-0.5,0.5,pts))
  # Steps
  aSteps::Float64 = 60.0/180.0*pi
  l = 0.075
  # meshes
  gridX = [i for i in values, j in values]
  gridY = [j for i in values, j in values]
  # 1) generate ellipse
  # rotate by alpha
  Xr = cos(aEll).*gridX - sin(aEll).*gridY
  Yr = cos(aEll).*gridY + sin(aEll).*gridX
  img = ((1/axes[1]*Xr.^2 + 1/axes[2]*Yr.^2).<=1).*( 10*pi*Yr + 4.0*pi*(Yr.<0));
  # 2) generate steps
  Xr = cos(aStep).*gridX - sin(aStep).*gridY
  Yr = cos(aStep).*gridY + sin(aStep).*gridX
  img += 4.0*pi/(radius^2)*( ((gridX-midPoint[1]).^2 + (gridY-midPoint[2]).^2) .<= radius^2).*(radius^2 - ((gridX-midPoint[1]).^2 + (gridY-midPoint[2]).^2))
  Xr = cos(aSteps).*gridX - sin(aSteps).*gridY
  Yr = cos(aSteps).*gridY + sin(aSteps).*gridX
  for i=1:8
    midP = [i*l;i*l]+[-.125;-.55]
    img += 2*pi*(i/8)*( (abs(Xr-midP[1])+abs(Yr-midP[2])) .<= l)
  end
  return mod(img-pi,2*pi)+pi
end
