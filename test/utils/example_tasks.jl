s = @isdefined _example_tasks_included
if !s
    _example_tasks_included = true

    """
        M, f, grad_f, p0, p_star = Circle_mean_task()

    Create a small mean problem on the circle to test Number-based algorithms
    """
    function Circle_mean_task()
        M = Circle()
        data = [-π / 2, π / 4, 0.0, π / 4]
        p_star = 0.0
        f(M, p) = 1 / 10 * sum(distance.(Ref(M), data, Ref(p)) .^ 2)
        grad_f(M, p) = 1 / 5 * sum(-log.(Ref(M), Ref(p), data))
        return M, f, grad_f, data[1], p_star
    end
end