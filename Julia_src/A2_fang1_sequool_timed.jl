using Convex, Random ,PProf ,Zygote ,BenchmarkTools ,Plots ,JuMP ,CSV ,DataFrames, Dates,Profile ,ProfileView ,MathOptInterface ,Clarabel ,Logging ,ForwardDiff ,PyFormattedStrings, DataStructures 
Profile.Allocs.clear()
optimizer = Convex.MOI.OptimizerWithAttributes(Clarabel.Optimizer,
    "tol_gap_abs" => 1e-2,          # Absolute duality gap tolerance (default: 1e-8)
    "tol_gap_rel" => 1e-2,          # Relative duality gap tolerance (default: 1e-8)
    "tol_feas" => 1e-3,             # Feasibility tolerance (default: 1e-8)
    "max_iter" => 100,              # Maximum iterations (default: 200),
    "equilibrate_enable" => false, 
    "static_regularization_enable" => true, 
    "iterative_refinement_enable" => true, 
    "verbose" => true               # Enable logging for solver details
)
global_logger(ConsoleLogger(stderr, Logging.Error))
timing_results = []



function time_and_store(identifier, func)
    start_time = time_ns()
    result = func()
    end_time = time_ns()
    elapsed_time = (end_time - start_time) / 1e9  # Convert to seconds
    push!(timing_results, Dict("Operation" => identifier, "Time (s)" => elapsed_time))
    return result
end



#//[x]: Change these functions (f and g) 
function f(x)
    return sum(x ./ (1:length(x)))
end


function g_u_cvx(x_var, t)
    return sin(t[1]) + sum(-t[1]^(i-1) * x_var[i] for i in 1:50)
end




function get_x_fang1(u)
    n = 50
    x_var = Variable(n)
    objective = sum(x_var ./ (1:length(x_var)))
    m = 1
    u_parts = [u[(i-1)*m+1:i*m] for i in 1:n]
    constraints = [g_u_cvx(x_var, part) <= -1e-4 for part in u_parts]
    push!(constraints, -x_var <= 0.)

    # Solve convex optimization problem and capture output
    problem = minimize(objective, constraints)
    
    open("solver_log.txt", "a") do io
        println(io, "======================================")
        println(io, "Solving at: ", now())
        println(io, "======================================")

        # Redirect stdout and stderr to log file
        redirect_stdout(io) do
            redirect_stderr(io) do
                solve!(problem, optimizer)
            end
        end

        println(io, "======================================\n")
    end

    # Extract results
    optimal_point = vec(evaluate(x_var))
    lagrange_multipliers_temp = [c.dual for c in constraints[1:n]]
    lagrange_multipliers = repeat(lagrange_multipliers_temp, inner = m)

    # Compute gradient
    grad_constraints = [
        ForwardDiff.gradient(v -> g_u_cvx(optimal_point, v), part) for part in u_parts
    ]
    
    final_grad = vcat(grad_constraints...)
    return optimal_point, lagrange_multipliers, final_grad
end


# function get_x_fang1(u)
#     n = 50
#     x_var = Variable(n)
#     objective = sum(x_var ./ (1:length(x_var)))
#     m = 1
#     u_parts = [u[(i-1)*m+1:i*m] for i in 1:n]
#     constraints = [g_u_cvx(x_var, part) <= -1e-4 for part in u_parts]
#     push!(constraints, -x_var <= 0.)

#     # Time the convex optimization step
#     problem = minimize(objective, constraints)
#     time_and_store("1. Solving convex optimization", () -> solve!(problem, optimizer))

#     # Extract results
#     lagrange_multipliers_temp = [c.dual for c in constraints[1:n]]
#     lagrange_multipliers = repeat(lagrange_multipliers_temp, inner = m)
#     optimal_point = vec(evaluate(x_var))

#     # Time gradient computation
#     grad_constraints = time_and_store("2. Computing gradient", () -> [
#         ForwardDiff.gradient(v -> g_u_cvx(optimal_point, v), part) for part in u_parts
#     ])
    
#     final_grad = vcat(grad_constraints...)
#     return optimal_point, lagrange_multipliers, final_grad
# end


mutable struct Node
    low_domain::Vector{Float64}
    up_domain::Vector{Float64}
    c_point::Vector{Float64}
    reward::Float64
    gradient::Vector{Float64}
    depth::Int
    visited::Bool
    id::Int
    index::Int
    parent::Int
    children::Vector{Int}
    is_leaf::Bool

    function Node(low_domain::Vector{Float64}, up_domain::Vector{Float64}, depth::Int, id::Int, index::Int, parent::Int)
        c_point = (low_domain .+ up_domain) ./ 2

        # Time the oracle call
        x_opt, lag_mul, fin_grad = time_and_store("3. Node evaluation (calling oracle)", () -> get_x_fang1(c_point))
        reward = f(x_opt)
        gradient = lag_mul .* fin_grad
        
        new(low_domain, up_domain, c_point, reward, gradient, depth, false, id, index, parent, Int[], true)
    end
end



# mutable struct Node
#     low_domain::Vector{Float64}
#     up_domain::Vector{Float64}
#     c_point::Vector{Float64}
#     reward::Float64
#     gradient::Vector{Float64}
#     depth::Int
#     visited::Bool
#     id::Int
#     index::Int
#     parent::Int
#     children::Vector{Int}
#     is_leaf:: Bool 

#     function Node(low_domain::Vector{Float64}, up_domain::Vector{Float64}, depth::Int, id::Int, index::Int, parent::Int)
#         c_point = (low_domain .+ up_domain) ./ 2  # Midpoint of domain
#         #// BUG: Defining direct computation of rewards 
#         #// [x] Requires redefinition of the Node struct for every problem 
#         x_opt, lag_mul, fin_grad = get_x_fang1(c_point) # CALL THE ORACLE 
#         reward = f(x_opt)                               # SET REWARD 
#         gradient = lag_mul .* fin_grad                  # SET GRADIENT     
#         visited = false
#         children = Int[]
#         is_leaf = true
#         new(low_domain, up_domain, c_point, reward, gradient, depth, visited, id, index, parent, children, is_leaf)
#     end
# end

mutable struct Partition
    low_domain::Vector{Float64}
    up_domain::Vector{Float64}
    root::Node
    nodes_by_depth::Dict{Int, BinaryMaxHeap{Tuple{Float64, Int64, Node}}}
    height_tree::Int
    current_node::Node 
    global_node_id::Int

    function Partition(low_domain::Vector{Float64}, up_domain::Vector{Float64})
        root = Node(low_domain, up_domain, 0, 1, 1, -1)
        heap0 = BinaryMaxHeap{Tuple{Float64, Int64, Node}}()
        push!(heap0, (root.reward, root.id, root))
        new(low_domain, up_domain, root, Dict(0 => heap0), 0, root, 2)
    end
end

function reset_partition!(partition::Partition)
    partition.height_tree = 0        # Reset tree height
    root = Node(partition.low_domain, partition.up_domain, 0, 1, 1, -1)
    partition.root = root 
    # Create a new heap for depth 0 with the new root.
    partition.nodes_by_depth = Dict(0 => BinaryMaxHeap{Tuple{Float64, Node}}())
    push!(partition.nodes_by_depth[0], (root.reward, root))
    partition.current_node = root
end

# # Modified make_children! function
# function make_children!(partition::Partition, parent::Node, add_layer::Bool, split_dim::Int)
#     if split_dim < 1 || split_dim > length(parent.low_domain)
#         throw(ArgumentError("Invalid split_dim=$split_dim. Must be in range 1 to $(length(parent.low_domain))."))
#     end

#     depth = parent.depth
#     new_depth = depth + 1

#     # Compute split point (midpoint along the selected dimension)
#     mid = (parent.low_domain[split_dim] + parent.up_domain[split_dim]) / 2

#     # Create copies for child domains
#     low1, up1 = copy(parent.low_domain), copy(parent.up_domain)
#     low2, up2 = copy(parent.low_domain), copy(parent.up_domain)

#     # Adjust domains for the new children:
#     up1[split_dim]  = mid  # First child gets the lower half
#     low2[split_dim] = mid   # Second child gets the upper half

#     # Assign unique IDs and indices
#     child1_id = partition.global_node_id
#     partition.global_node_id += 1 
#     child2_id = partition.global_node_id
#     partition.global_node_id += 1

#     child1 = Node(low1, up1, new_depth, child1_id, 2 * parent.index - 1, parent.id)
#     child2 = Node(low2, up2, new_depth, child2_id, 2 * parent.index, parent.id)

#     # Update parent-child relationships
#     parent.children = [child1_id, child2_id]
#     parent.is_leaf = false

#     # Insert children into the heap for new_depth.
#     if !haskey(partition.nodes_by_depth, new_depth)
#         # Create a new BinaryMaxHeap for this depth if it does not exist.
#         partition.nodes_by_depth[new_depth] = BinaryMaxHeap{Tuple{Float64, Int64, Node}}()
#     end

#     # Push both children onto the heap.
#     push!(partition.nodes_by_depth[new_depth], (child1.reward, child1.id, child1))
#     push!(partition.nodes_by_depth[new_depth], (child2.reward, child2.id, child2))

#     # If add_layer flag is true, update the tree height.
#     if add_layer
#         partition.height_tree += 1
#     end
# end

function make_children!(partition::Partition, parent::Node, add_layer::Bool, split_dim::Int)
    time_and_store("4. Splitting node", () -> begin
        depth = parent.depth
        new_depth = depth + 1

        mid = (parent.low_domain[split_dim] + parent.up_domain[split_dim]) / 2
        low1, up1 = copy(parent.low_domain), copy(parent.up_domain)
        low2, up2 = copy(parent.low_domain), copy(parent.up_domain)
        up1[split_dim] = mid
        low2[split_dim] = mid

        child1_id = partition.global_node_id
        partition.global_node_id += 1
        child2_id = partition.global_node_id
        partition.global_node_id += 1

        child1 = Node(low1, up1, new_depth, child1_id, 2 * parent.index - 1, parent.id)
        child2 = Node(low2, up2, new_depth, child2_id, 2 * parent.index, parent.id)

        parent.children = [child1_id, child2_id]
        parent.is_leaf = false

        if !haskey(partition.nodes_by_depth, new_depth)
            partition.nodes_by_depth[new_depth] = BinaryMaxHeap{Tuple{Float64, Int64, Node}}()
        end

        push!(partition.nodes_by_depth[new_depth], (child1.reward, child1.id, child1))
        push!(partition.nodes_by_depth[new_depth], (child2.reward, child2.id, child2))

        if add_layer
            partition.height_tree += 1
        end
    end)
end
function get_top_m_nodes_heap!(partition::Partition, depth::Int, m::Int, result::Dict{Int, Node}, iter_count::Int)
    if !haskey(partition.nodes_by_depth, depth)
        println("No nodes found at depth $depth")
        return iter_count
    end

    heap = partition.nodes_by_depth[depth]
    count = m

    while count > 0
        current_node = pop!(heap)[3]
        split_dim = argmax(abs.(current_node.gradient))
        
        # Time node expansion
        time_and_store("5. Expanding node", () -> make_children!(partition, current_node, true, split_dim))

        result[iter_count] = current_node
        iter_count += 1
        count -= 1
    end
    return iter_count
end



function run_algorithm_SequOOL!(partition::Partition, T::Int64)
    df = CSV.read("allocations_dataframe.csv", DataFrame)
    row = df[df.N .== T, :]
    h_max = row[1, "h_max"]
    result = Dict{Int64, Node}()
    iter_count = 1
    root_node = partition.root
    split_dim = argmax(abs.(root_node.gradient))

    # Time root node split
    time_and_store("6. Initial root split", () -> make_children!(partition, root_node, true, split_dim))

    for i in 1:h_max
        budget_h = row[1, Symbol("h_$i")]
        
        # Time heap expansion
        iter_count = time_and_store("7. Processing heap expansion", () -> get_top_m_nodes_heap!(partition, i, budget_h, result, iter_count))
    end

    return result
end


function save_timing_results()
    df = DataFrame(timing_results)
    CSV.write("timing_results.csv", df)
    return df
end

# function run_algorithm_SequOOL!(partition::Partition, T::Int64)
#     # Let's expand this in detail 
#     # The code should always take in a partition which is empty 
#     # Then it should pull the row from the dataframe
#     # It should then define a node_stack --> Dict()
#     df = CSV.read("allocations_dataframe.csv", DataFrame)
#     row = df[df.N .== T, :]
#     h_max = row[1, "h_max"]
#     result = Dict{Int64, Node}()
#     iter_count = 1
#     root_node = partition.root
#     split_dim = argmax(abs.(root_node.gradient))
#     make_children!(partition, root_node, true, split_dim)    
#     for i in 1:h_max
#         budget_h = row[1, Symbol("h_$i")]
#         #println(f"Budget at {i} is {budget_h}")
#         iter_count = get_top_m_nodes_heap!(partition, i, budget_h, result, iter_count)
#         #println(f"Iter count is {iter_count}")
#         #println(f"Number of nodes expanded until now is {partition.global_node_id}")
#     end
#     return result
# end

function plot_rewards(node_stack_expand)
    rewards = []
    node_ids = []
    max_reward = -Inf
    best_node = nothing

    # Iterate over the node stack and collect rewards and node IDs
    for (iter, node) in node_stack_expand
        push!(rewards, node.reward)
        push!(node_ids, node.id)
        
        # Update the node with the maximum reward
        if node.reward > max_reward
            max_reward = node.reward
            best_node = node
        end
    end

    # Plot the rewards
    plot_obj = plot(1:length(rewards), rewards, xlabel="Iteration", ylabel="Reward", 
                    title="Reward vs Iteration", legend=false, lw=2, color=:blue)

    # Force plot to show
    display(plot_obj)

    return best_node
end



#// [x] Change these for every problem
global low_domain = [0.1 for _ in 1:50]
global up_domain  = [1.0 - 0.1*sqrt(pi) for _ in 1:50]
global partition = Partition(low_domain, up_domain)
global T = 100
open("solver_log.txt", "w") do io
    close(io)  # Ensure file is emptied
end
node_stack_expand = @time run_algorithm_SequOOL!(partition, T)
save_timing_results()
best_node = plot_rewards(node_stack_expand)


sorted_keys = sort(collect(keys(node_stack_expand)))

# Extract nodes in sorted order
sorted_nodes = [node_stack_expand[k] for k in sorted_keys]

df = DataFrame(
    id = [node.id for node in sorted_nodes],
    low_domain = [join(node.low_domain, ",") for node in sorted_nodes],
    up_domain = [join(node.up_domain, ",") for node in sorted_nodes],
    c_point = [join(node.c_point, ",") for node in sorted_nodes],
    reward = [node.reward for node in sorted_nodes],
    gradient = [join(node.gradient, ",") for node in sorted_nodes],
    depth = [node.depth for node in sorted_nodes],
    visited = [node.visited for node in sorted_nodes],
    index = [node.index for node in sorted_nodes],
    parent = [node.parent for node in sorted_nodes],
    children = [join(node.children, ",") for node in sorted_nodes],
    is_leaf = [node.is_leaf for node in sorted_nodes]
)

# Store to CSV
CSV.write("tree_nodes_sequOOL.csv", df)

