using Convex, Random ,PProf ,Zygote ,BenchmarkTools ,Plots ,JuMP ,CSV ,DataFrames ,Profile ,ProfileView ,MathOptInterface ,Clarabel ,Logging ,ForwardDiff ,PyFormattedStrings, DataStructures 
Profile.Allocs.clear()
optimizer = Convex.MOI.OptimizerWithAttributes(Clarabel.Optimizer, "verbose" => false,)
global_logger(ConsoleLogger(stderr, Logging.Error))



#//[x]: Change these functions (f and g) 
function f(x)
    return sum(x ./ (1:length(x)))
end


function g_u_cvx(x_var, t)
    return sin(t[1]) + sum(-t[1]^(i-1) * x_var[i] for i in 1:50)
end

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
        c_point = (low_domain .+ up_domain) ./ 2  # Midpoint of domain
        #// BUG: Defining direct computation of rewards 
        #// [x] Requires redefinition of the Node struct for every problem 
        x_opt, lag_mul, fin_grad = get_x_fang1(c_point) # CALL THE ORACLE 
        reward = f(x_opt)                               # SET REWARD 
        gradient = lag_mul .* fin_grad                  # SET GRADIENT     
        visited = false
        children = Int[]
        is_leaf = true
        new(low_domain, up_domain, c_point, reward, gradient, depth, visited, id, index, parent, children, is_leaf)
    end
end

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

# Modified make_children! function
function make_children!(partition::Partition, parent::Node, add_layer::Bool, split_dim::Int)
    if split_dim < 1 || split_dim > length(parent.low_domain)
        throw(ArgumentError("Invalid split_dim=$split_dim. Must be in range 1 to $(length(parent.low_domain))."))
    end

    depth = parent.depth
    new_depth = depth + 1

    # Compute split point (midpoint along the selected dimension)
    mid = (parent.low_domain[split_dim] + parent.up_domain[split_dim]) / 2

    # Create copies for child domains
    low1, up1 = copy(parent.low_domain), copy(parent.up_domain)
    low2, up2 = copy(parent.low_domain), copy(parent.up_domain)

    # Adjust domains for the new children:
    up1[split_dim]  = mid  # First child gets the lower half
    low2[split_dim] = mid   # Second child gets the upper half

    # Assign unique IDs and indices
    child1_id = partition.global_node_id
    partition.global_node_id += 1 
    child2_id = partition.global_node_id
    partition.global_node_id += 1

    child1 = Node(low1, up1, new_depth, child1_id, 2 * parent.index - 1, parent.id)
    child2 = Node(low2, up2, new_depth, child2_id, 2 * parent.index, parent.id)

    # Update parent-child relationships
    parent.children = [child1_id, child2_id]
    parent.is_leaf = false

    # Insert children into the heap for new_depth.
    if !haskey(partition.nodes_by_depth, new_depth)
        # Create a new BinaryMaxHeap for this depth if it does not exist.
        partition.nodes_by_depth[new_depth] = BinaryMaxHeap{Tuple{Float64, Int64, Node}}()
    end

    # Push both children onto the heap.
    push!(partition.nodes_by_depth[new_depth], (child1.reward, child1.id, child1))
    push!(partition.nodes_by_depth[new_depth], (child2.reward, child2.id, child2))

    # If add_layer flag is true, update the tree height.
    if add_layer
        partition.height_tree += 1
    end
end



function get_top_m_nodes_heap!(partition::Partition, depth::Int, m::Int, result::Dict{Int, Node}, iter_count::Int)
    # Check if there are nodes at the given depth.
    if !haskey(partition.nodes_by_depth, depth)
        println("No nodes found at depth $depth")
        return []
    end

    # 2. Use a max-heap to select the top m nodes.
    heap = partition.nodes_by_depth[depth]  # Max-heap based on reward
    count = m
    # Extract elements from the heap (they come out in descending order by reward).
    while count > 0
        current_node = pop!(heap)[3]
        split_dim = argmax(abs.(current_node.gradient))
        make_children!(partition, current_node, true, split_dim)
        result[iter_count] = current_node 
        iter_count += 1
        count -= 1
    end
    return iter_count
end

function run_algorithm_SequOOL!(partition::Partition, T::Int64)
    # Let's expand this in detail 
    # The code should always take in a partition which is empty 
    # Then it should pull the row from the dataframe
    # It should then define a node_stack --> Dict()
    df = CSV.read("allocations_dataframe.csv", DataFrame)
    row = df[df.N .== T, :]
    h_max = row[1, "h_max"]
    result = Dict{Int64, Node}()
    iter_count = 1
    root_node = partition.root
    split_dim = argmax(abs.(root_node.gradient))
    make_children!(partition, root_node, true, split_dim)    
    for i in 1:h_max
        budget_h = row[1, Symbol("h_$i")]
        println(f"Budget at {i} is {budget_h}")
        iter_count = get_top_m_nodes_heap!(partition, i, budget_h, result, iter_count)
        println(f"Iter count is {iter_count}")
    end
    return result
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
global T = 200
node_stack_expand = @time run_algorithm_SequOOL!(partition, T)
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

