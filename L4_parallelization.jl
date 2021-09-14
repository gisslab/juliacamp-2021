####parallelization example
using Distributed

#add processes
workers()
addprocs(2)

# Want to draw a Normal(0, σ) sample of n observations for X times
# First need to write down this function and broadcast it to all processors using @everywhere
@everywhere using Distributions, Statistics
@everywhere function draw(σ::Int64, n::Int64)
    dist = Normal(0, σ)
    draws = rand(dist, n)
    return [mean(draws) std(draws)]
end

## Method 1: using @distributed for with ShareArrays
# Sequential execution
println("Sequential execution")
temp = zeros(10,2);
@time for i = 1:10
    temp[i,:] =  draw(i, 100000000)
end

# parallelized execution
println("Parallelized execution")
# Use ShareArrays to collect results from different cores
using SharedArrays
temp = SharedArray{Float64}(10,2);
@time @sync @distributed for i = 1:2
    temp[i,:] =  draw(i, 100000000)
end

## Method 2: using pmap (which stands for parallel map)
println("parallel map")
@time temp = pmap(i->draw(i,100000000),[i for i = 1:10])
# For each process i = 1:10, the command execute draw(i, 100000000) and the output is a vector.
# The i-th element in temp is the result from process i.

# You can also try sequential execution.
println("non-parallel map")
@time temp = map(i->draw(i,100000000),[i for i = 1:10])
# Even without parallelization, map() is still powerful because you can write a function in terms of a scalar.
# Then propagate the function to an array.
# Could do this in Matlab but very slow...

println("Finished!")

#### Parallelization for value function iteration
## Optimal Savings
using Distributed, SharedArrays
@everywhere using Parameters, Plots

# Household preference: u(c) = log(c)
# Production technology: y = k^θ
# Capital depreciation rate: δ
# Budget constraint: c + k' = k^θ + (1-δ)*k

# Dynamic programming problem:
# V(k) = max_{k'} u(c) + β V(k') s.t. c + k' = k^θ + (1-δ)*k
# Solve the problem and plot the policy function k' = k'(k)
# Guess V(k'), get V(k) (function Bellman() below)
# Iterate until the function converge (function Solve_model() below)

# struct to hold model primitives (global constants)
@everywhere @with_kw struct Primitives
    β::Float64 = 0.9 #discount factor
    θ::Float64 = 0.36 #production
    δ::Float64 = 0.025 #depreciation
    k_grid::Array{Float64,1} = collect(range(0.1, length = 1800, stop= 45.0)) #capital grid
    nk::Int64 = length(k_grid) #number of capital grid states
end

# struct to hold model outputs
@everywhere mutable struct Results
    val_func::Array{Float64,1} #value function
    pol_func::Array{Float64,1} #policy function
end

#Bellman operator
function Bellman(prim::Primitives, res::Results)
    @unpack β, δ, θ, nk, k_grid = prim #unpack primitive structure
    v_next = SharedArray{Float64}(nk) #preallocate next guess of value function

    for i_k = 1:nk #loop over state space
        max_util = -1e10
        k = k_grid[i_k] #value of capital
        budget = k^θ + (1-δ)*k #budget

        @sync @distributed for i_kp = 1:nk #loop over choice set, find max {log(c) + β * res.val_func[i_kp]}
            kp = k_grid[i_kp] #value of k'
            c = budget - kp #consumption
            if c>0 #check if postiive
                val = log(c) + β * res.val_func[i_kp] #compute utility
                if val > max_util #wow new max!
                    max_util = val #reset max value
                    res.pol_func[i_k] = kp #update policy function
                end
            end
        end
        v_next[i_k] = max_util #update value function
    end
    v_next
end

#function to solve the model
function Solve_model()
    #initialize primitives and results
    prim = Primitives()
    val_func, pol_func = zeros(prim.nk), zeros(prim.nk)
    res = Results(val_func, pol_func)

    error, n = 100, 0
    while error>0.0001 #loop until convergence
        n+=1
        v_next = Bellman(prim, res) #next guess of value function
        error = maximum(abs.(v_next .- res.val_func)) #check for convergence
        res.val_func = v_next #update
        println("Current error: ", error)
    end
    println("Value function converged in ", n, " iterations")
    prim, res
end

@elapsed prim, res = Solve_model() #solve the model.
Plots.plot(prim.k_grid, res.val_func) #plot value function

#### In Linstat, here is the command to submit jl file and make a log
# julia L4_parallelization.jl | & tee L4.log
