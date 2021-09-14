import Pkg; Pkg.add("Optim");
using Optim, Plots

## Brent's method
f(x, y) = (x-y)^2
# syntax: optimize(function, min_x, max_x)
opt = optimize(x->f(x, 1.0), -5.0, 5.0)

## Rosenbrock
# global minimum (x1 = 1, x2 = 1)
function Rosenbrock(vec::Vector{Float64})
    x1, x2 = vec[1], vec[2]
    val = (1 - x1)^2 + 100 * (x2 - x1^2)^2
end

# evaluate function at a bunch of points
x_grid = collect(-3:0.01:3)
nx = length(x_grid)
z_grid = zeros(nx, nx)

for i = 1:nx, j = 1:nx
    guess = [x_grid[i], x_grid[j]]
    z_grid[i,j] = Rosenbrock(guess)
end

# plot
Plots.surface(x_grid, x_grid, z_grid, seriescolor=:viridis, camera = (50,50))
Plots.contourf(x_grid, x_grid, log.(1 .+ z_grid), seriescolor=:inferno)

### Nelder-Mead
guess = [0.0, 0.0]
# syntax: optimize(function, initial guess)
opt = optimize(Rosenbrock, guess)

### LBFGS
# syntzx: optimize(function, initial guess, optimization options)
opt = optimize(Rosenbrock, guess, LBFGS())

## Define Gradient
# Seems that optimize() requires gradient & Hessian be defined with in-place function
function g(G, guess::Vector{Float64})
    x, y = guess[1], guess[2]
    G[1] = -2.0 * (1.0 - x) - 400.0 * (y - x^2) * x
    G[2] = 200.0 * (y - x^2)
    G #return
end

# Hessian
function h(H, guess::Vector{Float64})
    x, y = guess[1], guess[2]
    H[1,1] = 2.0 - 400.0 * y + 1200.0 * x^2
    H[1,2] = -400.0 * x
    H[2,1] = -400.0 * x
    H[2,2] = 200.0
    H #retturn
end

# Nelder_Mead
guess = [0.0, 0.0]
@time opt = optimize(Rosenbrock, g, h, guess)
@time opt = optimize(Rosenbrock, guess,BFGS())

## Lots of local minima
function Greiwank(x::Array{Float64,2})
    return (1/4000)*sum(x.^2) - prod(cos.(x./sqrt(length(x)))) + 1
end

### evaluate function at a bunch of points
x_grid = collect(-5:0.01:5)
nx = length(x_grid)
z_grid = zeros(nx, nx)

for i = 1:nx, j = 1:nx
    guess_val = [x_grid[i], x_grid[j]]
    z_grid[i,j] = Greiwank(guess_val)
end

### plots
Plots.surface(x_grid, x_grid, z_grid, seriescolor=:viridis, camera = (50,50))
Plots.contourf(x_grid, x_grid, z_grid, seriescolor=:inferno)

### global optimum at (0,0)
guess_init = [3.0, 3.0]
opt = optimize(Greiwank, guess_init) #this fails!
println(opt.minimizer)

#now this works!
guess_init = [2.0, 2.0]
opt = optimize(Greiwank, guess_init) #this works!
println(opt.minimizer)

#try multiple starts!
function Multistart()
    x_grid = collect(-5:0.5:5)
    nx = length(x_grid)
    minimum, minimizers = 100, [100, 100] # initial (bad) guesses for minimum and minimizes

    for i = 1:nx, j = 1:nx
        guess = [x_grid[i], x_grid[j]] #starting guess
        opt = optimize(Greiwank, guess) #nelder-mead with new starting guess
        if opt.minimum<minimum #new minimum!
            minimum = opt.minimum #update
            minimizers = opt.minimizer #update
        end
    end
    minimum, minimizers #return
end
min, minimizers = Multistart()

## OLS example
using Distributions, Random

# run the same OLS as first class
dist = Normal(0,1)
β_0 = 1.0; β_1 = 2.0; β_2 = 3.0;
n = 10000;
x = rand(n).*10;
x2 = x.^2;
ϵ = rand(dist, n) ## generate random shocks
Y_data = β_0 .+ β_1.*x + β_2.*x2 .+ ϵ
X = hcat(ones(n), x, x2)
β_ols = inv(X' * X) * X' * Y_data

### ols-nelder function
function OLS_Nelder(β::Array{Float64,1})
    β_0, β_1, β_2 = β[1], β[2], β[3] #unpack β
    ϵ = Y_data - (β_0 .+ β_1.*x + β_2.*x2)
    return error = sum(ϵ.^2) #sum of squared error
end

#do OLS with nelder-mead
guess_init = [0.0, 0.0, 0.0]
opt = optimize(OLS_Nelder, guess_init) #it works
println("Estimate")
println(opt.minimizer)
println(propertynames(opt)) # show argument names in opt
println("Analytical Solution")
println(β_ols)

# Let's try a tighter tolerance level
opt = optimize(OLS_Nelder, guess_init, BFGS(), Optim.Options(x_tol=1e-17,f_tol=1e-17,g_tol=1e-17))
println("Estimate")
println(opt.minimizer)
println(propertynames(opt)) # show argument names in opt
println("Analytical Solution")
println(β_ols)
