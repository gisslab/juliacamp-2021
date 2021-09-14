using Random, Plots, Distributions, Statistics, Parameters

######birthday problem
function birthday(n::Int64, sims::Int64)
    results = zeros(sims) #preallocate monte-carlo results vector
    for i = 1:sims #loop over simulations
        days = rand(1:365, n) #draw birthdays
        results[i] = length(unique(days))
        # unique(days) returns the unique elements in days
        # length(unique(days)) returns the number of unique elements in days
    end
    results #return
end
res_20, res_50, res_70 = birthday(20, 10000), birthday(50, 10000), birthday(70, 10000)
Plots.histogram(res_20)
Plots.histogram(res_50)
Plots.histogram(res_70)

#####Average distance between two random points in a cube
function Point_distance(sims::Int64)
    results = zeros(sims)
    for i = 1:sims #loop over simulations
        p1, p2 = rand(3), rand(3) #two points!
        results[i] = sqrt(sum((p1.-p2).^2))
    end
    return mean(results)
end
Point_distance(10000)

#### Gibbs Sampler for Correlated Normal
# Note: for illustrative purpose only. Improve this by:
# changing mean and variance beyond (0,1)
# try different seeds then calculate average

function bigibbs(T::Int64, rho::Float64)
    x = zeros(T+1)
    y = zeros(T+1)
    for t = 1:T
        x[t+1] = randn() * sqrt(1-rho^2) + rho*y[t]
        y[t+1] = randn() * sqrt(1-rho^2) + rho*x[t+1]
    end
    return x, y
end

x,y = bigibbs(100000, 0.8)
x = x[10000:end];
y = y[10000:end];
using Statistics
mean(x), var(x), mean(y), var(y), cov(x,y)

#### Importance Sampling
Random.seed!(12345)
p = Normal(0,5); # target distribution
n = 1000; # sample size
q1 = Normal(0, 10); # auxiliary distribution 1 (wide coverage of p's support)
q2 = Normal(0, 3); # auxiliary distribution 2 (wide coverage of p's support)
x1 = rand(q1, n); x2 = rand(q2,n);
# E[Y^2]
E_y_sq1 = mean(x1.^2 .* pdf.(p,x1) ./ pdf.(q1,x1));
E_y_sq2 = mean(x2.^2 .* pdf.(p,x2) ./ pdf.(q2,x2));
# That's why we want Halton sequence

#### Halton Sequence
using HaltonSequences
s = Halton(3,length=n*3);
s = s[2n+1:end];
using Distributions
x2 = quantile.(q2, s);
E_y_sq2 = mean(x2.^2 .* pdf.(p,x2) ./ pdf.(q2,x2));

#### Quadrature
using FastGaussQuadrature
### Simulate a Normal Integral
    # Transform Gauss-Hermite weights to ω_k/\sqrt{π} and abscissa to μ + sqrt(2)*σ*ζ_k.

μ = 0; σ = 1;
ζ, ω=gausshermite(5);
ζ = μ .+ sqrt(2) * σ * ζ;
ω = ω ./ sqrt(pi);
# to approximate E(x), f(x) = x
sum(ω .* ζ)

# to approximate E(x^2), f(x) = x^2
sum(ω .* (ζ.^2))


####something more involved: computing expected value of college factoring in wage offer shocks
@with_kw struct Primitives
    β_0::Float64 = 2.7 #wage constant
    β_1::Float64 = 0.47 #college premium
    σ::Float64 = 0.597 #wage offer SD
    α::Float64 = 1.0 #leisure
    B::Float64 = 5.0 #base consumption
end

mutable struct Results
    emax::Array{Float64,1} #first for no college, second for college
    lfp::Array{Float64,1} #lfp probabilities
end

function Compute_emax(prim::Primitives, res::Results, sims::Int64)
    #housekeeping
    @unpack β_0, β_1, σ, α, B = prim
    dist = Normal(0, σ)
    val_nwork = α + log(B)
    utils, lfps = zeros(2, sims), zeros(2, sims)

    for s = 0:1 #loop over schooling levels
        for i = 1:sims #loop over simulations
            ε = rand(dist) #draw shock and compute resultant wage
            wage = exp(β_0 + β_1*s + ε)
            utils[s+1, i] = max(log(wage), val_nwork)
            lfps[s+1, i] = (log(wage)>val_nwork) #update working decision
        end
    end

    res.emax[1], res.emax[2] = mean(utils[1,:]), mean(utils[2,:]) #expected max utilities
    res.lfp[1], res.lfp[2] = mean(lfps[1,:]), mean(lfps[2,:]) #LFP probabilities
end

#initializes model primitives and executes solution
function Solve_model(sims::Int64)
    prim = Primitives() #initialize primitives
    res = Results(zeros(2), zeros(2)) #initialize resutls
    Compute_emax(prim, res, sims) #solve model
    prim, res #return
end


prim, res = Solve_model(100) #run the code
