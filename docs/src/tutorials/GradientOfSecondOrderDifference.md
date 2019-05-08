```@meta
EditURL = "https://github.com/TRAVIS_REPO_SLUG/blob/master/"
```

# Illustration of the Gradient of a Second Order Difference

This example explains how to compute the gradient of the second order
difference mid point model using
[`adjointJacobiField`](@ref)s.

We first have to initialize the Toolbox, set our favourite choice of Manifold
and define the three points we aim to look at

```@example GradientOfSecondOrderDifference
using Manopt
M = Sphere(2)
x = SnPoint([1., 0., 0.])
z = SnPoint([0., 1., 0.])
c = midPoint(M,x,z)
y = geodesic(M, SnPoint([0., 0., 1.]), c, 0.05)
```

