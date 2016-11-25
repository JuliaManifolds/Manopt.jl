using ManifoldValuedImageProcessing
using Base.Test
using Compat
tests = ["testSn"]

for t in tests
  include("$(t).jl")
end
