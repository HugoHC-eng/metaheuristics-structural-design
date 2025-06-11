# Pure Genetic Algorithm (GA) in Julia for I‐beam design optimization

using Random
using Printf
using Plots

# === PROBLEM PARAMETERS ===
const LB = (h=10.0, b=10.0, tw=0.9, tf=0.9)
const UB = (h=80.0, b=50.0, tw=5.0, tf=5.0)

# === GA PARAMETERS ===
const POP_SIZE          = 30
const N_GENERATIONS     = 10_000
const CROSSOVER_RATE    = 0.8
const MUTATION_RATE     = 0.2
const MUTATION_STRENGTH = 0.1

# === AUXILIARY FUNCTIONS ===

"""
    constraints(h, b, tw, tf)

Compute the moment of inertia I and the two constraint values g1 (area)
and g2 (stress) for an I‐beam section.
"""
function constraints(h, b, tw, tf)
    I    = (tw*(h - 2*tf)^3)/12 + (b*tf^3)/6 + 2*b*tf*((h - tf)/2)^2
    g1   = 2*b*tf + tw*(h - 2*tf)
    denom = tw*(h - 2*tf)^3 +
            2*b*tf*(4*tf^2 + 3*h*(h - 2*tf)) +
            tw^3*(h - 2*tf) +
            2*tw*b^3
    g2   = (18_000*h + 15_000*b) / denom
    return I, g1, g2
end

"""
    objective(x)

Evaluate the objective function for a candidate solution tuple x=(h,b,tw,tf).
Returns 1e6 if any constraint is violated, otherwise deflection proxy = 5000/I.
"""
function objective(x::NTuple{4,Float64})
    h, b, tw, tf = x
    I, g1, g2     = constraints(h, b, tw, tf)
    f             = 5000 / I
    return (g1 > 300 || g2 > 6) ? 1e6 : f
end

"""
    tournament_selection(pop, fits)

Binary tournament selection: pick two random individuals and return
the one with the better (lower) fitness.
"""
function tournament_selection(pop, fits)
    i1, i2 = rand(1:length(pop)), rand(1:length(pop))
    return fits[i1] < fits[i2] ? pop[i1] : pop[i2]
end

"""
    mutate(x)

Gaussian mutation on each gene with probability MUTATION_RATE,
clamped to the [LB,UB] bounds.
"""
function mutate(x::NTuple{4,Float64})
    h, b, tw, tf = x
    vals = (; h=h, b=b, tw=tw, tf=tf)
    for (i, field) in enumerate(keys(vals))
        if rand() < MUTATION_RATE
            Δ   = MUTATION_STRENGTH * randn()
            val = getfield(x, i) + Δ
            lb  = getfield(LB, field)
            ub  = getfield(UB, field)
            val = clamp(val, lb, ub)
            x   = Base.setindex(x, val, i)
        end
    end
    return x
end

"""
    initialize_population()

Generate the initial population of POP_SIZE individuals uniformly
within the variable bounds.
"""
function initialize_population()
    return [(
        LB.h  + rand()*(UB.h  - LB.h),
        LB.b  + rand()*(UB.b  - LB.b),
        LB.tw + rand()*(UB.tw - LB.tw),
        LB.tf + rand()*(UB.tf - LB.tf)
    ) for _ in 1:POP_SIZE]
end

# === MAIN FUNCTION ===

function run_ga()
    # initialize population and fitness
    population    = initialize_population()
    fitness       = [objective(ind) for ind in population]
    best_history  = Float64[]

    # main loop
    for gen in 1:N_GENERATIONS
        new_pop = NTuple{4,Float64}[]
        for i in 1:2:POP_SIZE
            # selection
            p1 = tournament_selection(population, fitness)
            p2 = tournament_selection(population, fitness)
            # uniform crossover
            if rand() < CROSSOVER_RATE
                α       = rand(4)
                child1  = (
                    α[1]*p1[1] + (1-α[1])*p2[1],
                    α[2]*p1[2] + (1-α[2])*p2[2],
                    α[3]*p1[3] + (1-α[3])*p2[3],
                    α[4]*p1[4] + (1-α[4])*p2[4]
                )
                child2  = (
                    α[1]*p2[1] + (1-α[1])*p1[1],
                    α[2]*p2[2] + (1-α[2])*p1[2],
                    α[3]*p2[3] + (1-α[3])*p1[3],
                    α[4]*p2[4] + (1-α[4])*p1[4]
                )
            else
                child1, child2 = p1, p2
            end
            # mutation
            push!(new_pop, mutate(child1))
            if length(new_pop) < POP_SIZE
                push!(new_pop, mutate(child2))
            end
        end

        # evaluate and replace
        population = new_pop
        fitness    = [objective(ind) for ind in population]
        push!(best_history, minimum(fitness))
    end

    # final result
    idx_best = findmin(fitness)[2]
    best     = population[idx_best]
    fval     = best_history[end]
    I, g1, g2 = constraints(best...)
    println("\n✅ BEST SOLUTION (Pure GA):")
    @printf("h   = %.4f cm\n", best[1])
    @printf("b   = %.4f cm\n", best[2])
    @printf("tw  = %.4f cm\n", best[3])
    @printf("tf  = %.4f cm\n", best[4])
    @printf("Deflection proxy  = %.6f\n", fval)
    @printf("g1 (area)         = %.4f (≤ 300)\n", g1)
    @printf("g2 (stress)       = %.4f (≤ 6)\n", g2)

    # convergence plot
    plot(best_history;
        lw=1.5,
        xlabel="Generation",
        ylabel="Best Objective Value",
        title="Convergence of Pure GA",
        legend=false,
        grid=true
    )
end

# execute the GA
run_ga()
g