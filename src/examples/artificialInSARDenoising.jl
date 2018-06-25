#
# artificialInSARDenoising.jl
# A script providing an example of denoising phase valued data using the
# `artificialInSARImage(256)` and wrapped Gaussian noise.
#
# This script recreates the examples appeared in
# > R. Bergmann, F. Laus, G. Steidl, A. Weinmann:
# > Second Order Differences of Cyclic Data and Applications in Variational Denoising,
# > SIAM Journal on Imaging Sciences, Vol. 7, No. 4, pp. 2916–2953, 2014.
#
# Manopt.jl ~ R. Bergmsnn ~ 2016-12-09

using Manopt

# Global Names and parameters
dataName = "src/examples/artInSAR/data"
experimentName = "src/examples/artInSAR/experiment1"
σ = 0.3;
loadData = false;

if loadData
  # load phases
  InSARimage = readdlm(string(dataName,"-original.txt"))
  InSARimageNoisy = readdlm(string(dataName,"-noisy.txt"))
  # convert to manifold valued data
  S1image = S1Point.(InSARimage)
  S1imageNoisy = S1Point.(InSARimageNoisy)
else
  InSARImage = artificialInSARImage(256);
  S1image = S1Point.(InSARImage)
  S1imageNoisy = addNoise.(S1image,σ)
  InSARImageNoisy = [p.value for p in S1imageNoisy] #extract noisy data again
  writedlm(string(dataName,"-original.txt"),InSARImage)
  writedlm(string(dataName,"-noisy.txt"),InSARImageNoisy)
end

@time S1imageReconstruction = TV_Regularization_CPPA(S1imageNoisy,[3.0/8.0,1.0/4.0],pi/2; MaxIterations=200)
#@time S1imageReconstruction = TV_Regularization_CPPA(S1imageNoisy,[3.0/8.0,1.0/4.0],pi/2)
#@time S1imageReconstruction = TV_Regularization_CPPA(S1imageNoisy,[3.0/8.0,1.0/4.0],pi/2)
#@code_warntypeTV_Regularization_CPPA(S1imageNoisy,[3.0/8.0,1.0/4.0],pi/2)
#@code_warntypeTV_Regularization_CPPA(S1imageNoisy,[3.0/8.0,1.0/4.0],pi/2; MaxIterations=100)
