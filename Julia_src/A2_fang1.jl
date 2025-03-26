using Convex
using Random
using PProf
using Zygote
using BenchmarkTools
using Plots
using JuMP
using CSV
using DataFrames
using Profile
Profile.Allocs.clear()
using ProfileView
using MathOptInterface
using Clarabel
using Logging
using ForwardDiff
using PyFormattedStrings 
optimizer = Convex.MOI.OptimizerWithAttributes(Clarabel.Optimizer, 
                                                "verbose" => false,
                                              )
global_logger(ConsoleLogger(stderr, Logging.Error))
#// TODO Do not remove from here. 
#// TODO Check if we can change to immutable struct
# Define a Node struct with simpler types
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
    is_leaf:: Bool 

    function Node(low_domain::Vector{Float64}, up_domain::Vector{Float64}, depth::Int, id::Int, index::Int, parent::Int)
        # Initialize missing fields with default values
        c_point = (low_domain .+ up_domain) ./ 2  # Midpoint of domain
        reward = 0.0
        gradient = zeros(length(low_domain))
        visited = false
        children = Int[]
        is_leaf = true

        new(low_domain, up_domain, c_point, reward, gradient, depth, visited, id, index, parent, children, is_leaf)
    end
end

# Define Partition struct
mutable struct Partition
    low_domain::Vector{Float64}
    up_domain::Vector{Float64}
    root::Node
    nodes_by_depth::Dict{Int, Vector{Node}}
    height_tree::Int
    current_node::Node 

    function Partition(low_domain::Vector{Float64}, up_domain::Vector{Float64})
        root = Node(low_domain, up_domain, 0, 1, 1, -1)
        new(low_domain, up_domain, root, Dict(0 => [root]), 0, root)
    end
end

function reset_partition!(partition::Partition)
    partition.height_tree = 0        # Reset tree height
    root = Node(partition.low_domain, partition.up_domain, 0, 1, 1, -1)
    partition.root = root 
    partition.nodes_by_depth = Dict(0 => [root])
    partition.current_node = root
end

function make_children!(partition::Partition, parent::Node, add_layer::Bool, split_dim::Int)
    # Validate split_dim
    if split_dim < 1 || split_dim > length(parent.low_domain)
        throw(ArgumentError("Invalid split_dim=$split_dim. Must be in range 1 to $(length(parent.low_domain))."))
    end

    depth = parent.depth
    new_depth = depth + 1

    # Compute split point (midpoint along the selected dimension)
    mid = (parent.low_domain[split_dim] + parent.up_domain[split_dim]) / 2

    # Create low and high domain copies for child nodes
    low1, up1 = copy(parent.low_domain), copy(parent.up_domain)
    low2, up2 = copy(parent.low_domain), copy(parent.up_domain)

    # Adjust domains for the new children
    up1[split_dim] = mid  # First child gets the lower half
    low2[split_dim] = mid  # Second child gets the upper half

    # Assign unique IDs and indices
    child1_id = 1 + sum(length, values(partition.nodes_by_depth))
    child2_id = child1_id + 1

    child1 = Node(low1, up1, new_depth, child1_id, 2 * parent.index - 1, parent.id)
    child2 = Node(low2, up2, new_depth, child2_id, 2 * parent.index, parent.id)

    # Update parent-child relationships
    parent.children = [child1_id, child2_id]
    parent.is_leaf = false

    # Add children to the partition
    #// BUG This code doesn't seem to be correct so let us change it 
    # if add_layer
    #     if !haskey(partition.nodes_by_depth, new_depth)  # Fix here!
    #         partition.nodes_by_depth[new_depth] = []
    #     end
    #     push!(partition.nodes_by_depth[new_depth], child1, child2)
    #     partition.height_tree += 1
    # end

    #// FIXME Experimenting with the actual code here. 
    if add_layer
        if !haskey(partition.nodes_by_depth, new_depth)  # Fix here!
            partition.nodes_by_depth[new_depth] = []
        end
        push!(partition.nodes_by_depth[new_depth], child1, child2)
        partition.height_tree += 1
    else 
        push!(partition.nodes_by_depth[new_depth], child1, child2)
    end

end


function return_pull!(partition::Partition, h_t::Int, l_t::Int, v_max::Float64)
        # 0 --> FUNCTION EVALUATION (return c_point) 
        # 1 --> NODE EXPANSION (return root)
        # 2 --> CANNOT OVERCOME NODE CONDITION (return root)
        # 3 --> NO LEAVES IN GIVEN DEPTH (return root)
    root = partition.nodes_by_depth[0][1]
    node_list = partition.nodes_by_depth[h_t]
    max_value = -Inf
    max_node = nothing
    println(f"Number of nodes at the given depth is {size(node_list)}")
    for node in node_list
        if node.is_leaf == true
            if node.visited == false
                node.visited = true
                partition.current_node = node
                return node.c_point, 0, v_max
            end
            if node.reward >= max_value
                max_value = node.reward
                max_node = node 
            end 
        end 
    end
    if max_node === nothing
        partition.current_node = root 
        return root.c_point, 3, v_max
    elseif max_value >= v_max
        v_max = max_value
        partition.current_node = max_node
        add_layer = (h_t == l_t)
        #// FIXME try softmax and then sample 
        optimal_split_dim = argmax(abs.(partition.current_node.gradient))
        make_children!(partition, partition.current_node, add_layer, optimal_split_dim)
        return root.c_point, 1, v_max
    else 
        partition.current_node = root 
        return root.c_point, 2, v_max
    end 
end 

#//[x]: Change these functions (f and g) 
function f(x)
    return sum(x ./ (1:length(x)))
end


function g_u_cvx(x_var, t)
    return sin(t[1]) + sum(-t[1]^(i-1) * x_var[i] for i in 1:50)
end

#// TODO: Remove types later in the function declarations. 

function get_x_fang1(u)
    #//[x] Change these for every problem 
    n = 50
    x_var = Variable(n)
    objective = sum(x_var ./ (1:length(x_var)))
    m = 1
    u_parts = [u[(i-1)*m+1:i*m] for i in 1:n]
    constraints = [g_u_cvx(x_var, part) <= -1e-4 for part in u_parts]
    push!(constraints,-x_var <= 0.)
    #// [ ] Keep these same for all  
    problem = minimize(objective, constraints)
    solve!(problem, optimizer)
    lagrange_multipliers_temp = [c.dual for c in constraints[1:n]]
    lagrange_multipliers = repeat(lagrange_multipliers_temp, inner = m)
    optimal_point = vec(evaluate(x_var)) 
    #// [x] Change the call to the gradient function in ForwardDiff 
    grad_constraints = [ForwardDiff.gradient(v -> g_u_cvx(optimal_point, v), part) for part in u_parts]
    final_grad = vcat(grad_constraints...)
    return optimal_point, lagrange_multipliers, final_grad
end

#//[x] Define these for every problem 

function run_algorithm_SequOOL!(partition::Partition, T::Int64)
    node_stack_eval = Dict() 
    node_stack_expand = Dict() 
    l_t = 0 
    h_t = 0
    v_max = -Inf
    t = 0
    N = 0
    l_update = true
    h_max = floor(Int, sqrt(T))
    while(t <= T)
        #// BUG: Task --> Change the definition of l_t. Please check if this is indeed what you want 
        
        #// BUG: Add an update rule for l_t
        if l_update == true
            l_t = partition.height_tree
            l_update = false
        end
        point, action, v_max = return_pull!(partition, h_t, l_t, v_max)
        println(f"h_t is {h_t}, l_t is {l_t}, action is {action}")
        if action == 0 # FUNCTION EVALUATION
            #//[x] Change function name 
            x_opt, lag_mul, fin_grad = get_x_fang1(point) # CALL THE ORACLE 
            partition.current_node.reward = f(x_opt) # SET REWARD 
            partition.current_node.gradient = lag_mul .* fin_grad # SET GRADIENT 
            node_stack_eval[N] = partition.current_node # UPDATE EVALUATION NODE STACK 
            N += 1 
            #println(f"N is {N}, f is {f(x_opt)}")
        elseif action == 1 # NODE EXPANSION 
            node_stack_expand[t] = partition.current_node # UPDATE EXPANSION NODE STACK 
            t += 1
            if h_t == l_t 
                h_t = 0
                v_max = -Inf
                l_update = true
            else 
                h_t += 1
            end 
        elseif action == 2 # CANNOT OVERCOME NODE CONDITION
            if h_t == l_t 
                h_t = 0
                v_max = -Inf
                l_update = true
            else 
                h_t += 1
            end 
        elseif action == 3 # NO LEAF AT THE GIVEN LAYER 
            h_t += 1 
        end 
    end 
    #// FIXME Not sure if this works, remove if needed 
    reset_partition!(partition)
    return node_stack_eval, node_stack_expand 
end


function run_algorithm_SOO!(partition::Partition, T::Int64)
    node_stack_eval = Dict() 
    node_stack_expand = Dict() 
    l_t = 0 
    h_t = 0
    v_max = -Inf
    t = 0
    N = 0
    l_update = true
    h_max = floor(Int, sqrt(T))
    while(t <= T)
        #// BUG: Task --> Change the definition of l_t. Please check if this is indeed what you want 
        
        #// BUG: Add an update rule for l_t
        if l_update == true
            l_t = partition.height_tree
            l_update = false
        end
        point, action, v_max = return_pull!(partition, h_t, l_t, v_max)
        println(f"h_t is {h_t}, l_t is {l_t}, action is {action}")
        if action == 0 # FUNCTION EVALUATION
            #//[x] Change function name 
            x_opt, lag_mul, fin_grad = get_x_fang1(point) # CALL THE ORACLE 
            partition.current_node.reward = f(x_opt) # SET REWARD 
            partition.current_node.gradient = lag_mul .* fin_grad # SET GRADIENT 
            node_stack_eval[N] = partition.current_node # UPDATE EVALUATION NODE STACK 
            N += 1 
            #println(f"N is {N}, f is {f(x_opt)}")
        elseif action == 1 # NODE EXPANSION 
            node_stack_expand[t] = partition.current_node # UPDATE EXPANSION NODE STACK 
            t += 1
            if h_t == l_t 
                h_t = 0
                v_max = -Inf
                l_update = true
            else 
                h_t += 1
            end 
        elseif action == 2 # CANNOT OVERCOME NODE CONDITION
            if h_t == l_t 
                h_t = 0
                v_max = -Inf
                l_update = true
            else 
                h_t += 1
            end 
        elseif action == 3 # NO LEAF AT THE GIVEN LAYER 
            h_t += 1 
        end 
    end 
    #// FIXME Not sure if this works, remove if needed 
    reset_partition!(partition)
    return node_stack_eval, node_stack_expand 
end

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
global T = 10
#node_stack_eval, node_stack_expand = @time run_algorithm_SOO!(partition, T)
Profile.Allocs.@profile sample_rate=0.001 run_algorithm_SOO!(partition, T)
PProf.Allocs.pprof()
best_node = plot_rewards(node_stack_expand)

using DataFrame
using CSV
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
CSV.write("tree_nodes.csv", df)

