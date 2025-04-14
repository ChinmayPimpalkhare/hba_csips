using Convex, Clarabel, Random, Statistics, LinearAlgebra
using ForwardDiff, Plots, Dates

# Problem parameters
const n = 10
const m = 1
const d = n * m
const K = 10
const MAX_ITERS = 100

# Objective function: same as in get_x_fang1
function f(x)
    return sum(x ./ (1:length(x)))
end

# Constraint function
function g_u_cvx(x_var, t)
    return sin(t[1]) + sum(-t[1]^(i-1) * x_var[i] for i in 1:n)
end
function get_x_fang1(u, max_iters)
    n = 10
    m = 1
    x_var = Variable(n)
    objective = sum(x_var ./ (1:n))

    u_parts = [u[(i-1)*m+1:i*m] for i in 1:n]
    constraints = [g_u_cvx(x_var, part) <= -1e-4 for part in u_parts]
    push!(constraints, -x_var <= 0.)

    problem = minimize(objective, constraints)

    optimizer_factory = () -> Clarabel.Optimizer(; 
        max_iter = max_iters,
        verbose = false,
        tol_gap_abs = 1e-99,
        tol_gap_rel = 1e-99,
        tol_feas = 1e-99
    )

    start = time()
    solve!(problem, optimizer_factory)
    elapsed = time() - start

    return problem.optval, elapsed
end


# Benchmarking routine
function benchmark_get_x_fang1()
    obj_vals = zeros(K, MAX_ITERS)
    times = zeros(K, MAX_ITERS)

    for k in 1:K
        u_rand = randn(d)  # You can change this to uniform sampling if needed
        for iter in 1:MAX_ITERS
            obj, t = get_x_fang1(u_rand, iter)
            obj_vals[k, iter] = obj
            times[k, iter] = t
        end
    end

    return obj_vals, times
end

# Plotting results
function plot_results(obj_vals, times)
    iters = 1:MAX_ITERS
    mean_obj = vec(mean(obj_vals, dims=1))
    std_obj = vec(std(obj_vals, dims=1))

    mean_time = vec(mean(times, dims=1))
    std_time = vec(std(times, dims=1))

    p1 = plot(iters, mean_obj, ribbon=std_obj,
        xlabel="Max Iterations", ylabel="Objective Value",
        title="Objective Value vs Iterations", legend=false)

    p2 = plot(iters, mean_time, ribbon=std_time,
        xlabel="Max Iterations", ylabel="Runtime (s)",
        title="Runtime vs Iterations", legend=false)
    display(p1)
    #display(p2)
    return p1, p2
end

# Run full benchmark
obj_vals, times = benchmark_get_x_fang1()
plot_results(obj_vals, times)
