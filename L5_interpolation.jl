import Pkg; Pkg.add("Interpolations"); Pkg.add("Polynomials")
using Plots, Polynomials

## Polynomial Interpolation
### Runge's function/phenomenon
Runge(x) = 1 /(1 + 25*x^2)
x_fine = collect(-1.0:0.01:1.0)
Plots.plot(x_fine, Runge.(x_fine))

#5-degree polynomial fit
x_coarse = collect(range(-1.0, length = 6, stop = 1.0))
poly_fit = fit(x_coarse, Runge.(x_coarse), 5)
Plots.plot!(x_fine, poly_fit.(x_fine))

#try a higher-degree -- surely that will help!
x_coarse_2 = collect(range(-1.0, length = 10, stop = 1.0))
poly_fit_2 = fit(x_coarse_2, Runge.(x_coarse_2), 9)
Plots.plot!(x_fine, poly_fit_2.(x_fine))

## Linear/spline interpolation: Univariate Case
using Interpolations
# Evaluate the function at x_range
x_range = -1.0:0.25:1.0
runge_range = Runge.(x_range)
runge_linear = LinearInterpolation(x_range, runge_range)
runge_spline = CubicSplineInterpolation(x_range, runge_range)
# Approximate the function over x_fine
Plots.plot(x_fine, [Runge.(x_fine), runge_linear.(x_fine), runge_spline.(x_fine)])

## Linear/spline interpolation: Multivariate Case (interpolate())
### Scaled Interpolation
# interpolate() assumes x_range = 1:N, or its multidimensional equivalent.

# Therefore, the syntax is something like interpolate(y_range, BSpline(Linear())). No x_range is submitted.

# You need an additional step to map the interpolating results to the true domain.
x_range = 0.1:0.1:1
y_range = log.(x_range)
log_interp = interpolate(y_range, BSpline(Cubic(Line(OnGrid()))))
log_interp(1)
slog_interp = scale(log_interp, x_range)
slog_interp(1)

### Bivariate linear interpolation
f(x, y) = x^2 + y^2
grid_coarse = 0.0:0.5:5.0
grid_fine = 0.0:0.01:5.0
ncoarse, nfine = length(grid_coarse), length(grid_fine)
z_grid = [f(x,y) for x in grid_fine, y in grid_fine]
Plots.contourf(grid_fine, grid_fine, z_grid)

### Bi-variate interpolation
# Evaluate the function at grid_coarse
func_grid_coarse = [f(x,y) for x in grid_coarse, y in grid_coarse]

# create the linear interpolation
f_grid_interp = interpolate(func_grid_coarse, BSpline(Linear()))

# scale the interpolating result
sf_grid_interp = scale(f_grid_interp, grid_coarse, grid_coarse)
z_interp = sf_grid_interp(grid_fine, grid_fine)

#compare contour plots
Plots.contourf(grid_fine, grid_fine, z_grid)
Plots.contourf(grid_fine, grid_fine, z_interp) #slightly more jagged, but otherwise pretty good!

## Optimal Growth with Interpolation / Optimization
# Household preference: u(c) = log(c)
# Production technology: y = k^θ
# Capital depreciation rate: δ
# Budget constraint: c + k' = k^θ + (1-δ)*k

# Dynamic programming problem:
# V(k) = max_{k'} u(c) + β V(k') s.t. c + k' = k^θ + (1-δ)*k
# Solve the problem and plot the policy function k' = k'(k)
# Guess V(k'), get V(k) (function Bellman() below)
# Iterate until the function converge (function Solve_model() below)

using Parameters, Optim, Interpolations

#a struct to hold model primitives.
@with_kw struct Primitives
    β::Float64 = 0.99 #discount rate.
    θ::Float64 = 0.36 #capital share
    δ::Float64 = 0.025 #capital depreciation
    k_grid::Array{Float64,1} = collect(range(0.1, length = 50, stop = 45.0)) #capital grid. Much more coarse.
    nk::Int64 = length(k_grid) #number of capital elements
end

#mutable struct to hold model results
mutable struct Results
    val_func::Array{Float64,1}
    pol_func::Array{Float64,1}
end

#Bellman operator
function Bellman(prim::Primitives, res::Results)
    @unpack β, δ, θ, nk, k_grid = prim #unpack parameters from prim struct. Improves readability.
    v_next = zeros(nk)
    # k_interp = interpolate(k_grid, BSpline(Linear()))
    val_interp = scale(interpolate(res.val_func, (BSpline(Linear()))),k_grid[1]:k_grid[2]-k_grid[1]:k_grid[nk])
    # k_grid[1]:k_grid[2]-k_grid[1]:k_grid[nk] transforms k_grid array to a range, as is required by the scale() syntax.

    for i_k = 1:nk #loop over state space
        k= k_grid[i_k] #convert state indices to state values
        budget = k^θ + (1-δ)*k

        val(kp) = log(budget- kp) + β * val_interp(kp)
        obj(kp) = - val(kp)

        opt = optimize(obj, k_grid[1], min(budget,k_grid[nk])) #find optimal value
        res.pol_func[i_k] = opt.minimizer #policy function
        v_next[i_k] = -opt.minimum #update next guess of value function
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

@elapsed prim, res = Solve_model() #solve for policy and value functions #
Plots.plot(prim.k_grid, res.pol_func)
Plots.plot(prim.k_grid, res.val_func)
