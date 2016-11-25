using ManifoldValuedImageProcessing
using Base.Test
tests = ["testSn"]

for t in tests
  include("$(t).jl")
end
