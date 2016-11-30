#
#
# CPPAlgorithm based minimizations
#
#
export TV_Regularization_CPPA

"""
    TV_Regularization_CPPA(f,alpha, lambda) - compute the TV regularization model of
 given data array f and paramater alpha and internal operator start lambda.

 # Arguments
 * `f` an d-dimensional array of `ManifoldPoint`s
 * `alpha` parameter of the model
 * `lambda` internal parameter of the cyclic proxximal point algorithm
 # Output
 * `x` the regulraized array
 # Optional Parameters
 * `MinimalChange` (`10.0^(-5)`) minimal change for the algorithm to stop
 * `MaxIterations` (`500`) maximal number of iterations
 ---
 ManifoldValuedImageProcessing 0.8, R. Bergmann, 2016-11-25
"""
function TV_Regularization_CPPA{T <: ManifoldPoint}(
      lambda::Number, alpha::Number, f::Array{T};
      MinimalChange=10.0^(-5), MaxIterations=500)::Array{T}
  x = deepcopy(f)
  xold = deepcopy(x)
  iter = 1
  while (  ( (sum( [ distance(xi,xoldi) for (xi,xoldi) in zip(x,xold) ] ) > MinimalChange)
    && (iter < MaxIterations) ) || (iter==1)  )
    xold = deepcopy(x)
    # First term: d(f,x)^2
    for i in eachindex(x)
      x[i] = proxDistanceSquared(f[i],lambda/i,x[i])
    end
    # TV term
    for d in 1:ndims(f)
      for i in eachindex(f)
        # neighbor index
        i2 = i; i2[d] += 1;
        if ( all(i2 <=size(f)) )
          (x[i], x[i2]) = proxTV(alpha*lambda/i,(x[i], x[i2]))
        end
      end
    end
    iter +=1
  end
  return x;
end
