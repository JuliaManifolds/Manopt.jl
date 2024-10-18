"""
    ManoptTestSuite.jl

A small module to provide common dummy types and defaults for testing.
"""
module ManoptTestSuite
using Manopt, Test, ManifoldsBase

struct DummyManifold <: AbstractManifold{ManifoldsBase.ℝ} end

end
