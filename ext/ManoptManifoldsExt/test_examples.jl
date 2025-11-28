#
#
# Fill examples in the Tests submodule

function Manopt.Test.Circle_mean_task()
    M = Circle()
    data = [-π / 2, π / 4, 0.0, π / 4]
    p_star = 0.0
    f, grad_f = Manopt.Test.mean_task(M, data)
    return M, f, grad_f, data[1], p_star
end
