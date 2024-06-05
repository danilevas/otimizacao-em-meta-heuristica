import pyswarms as ps
import numpy as np
from pyswarms.backend.operators import compute_pbest, compute_objective_function

def optimize(objective_func, maxiters, oh_strategy, start_opts, end_opts):
    opt = ps.single.GlobalBestPSO(n_particles=30, dimensions=2, options=start_opts, oh_strategy=oh_strategy)

    swarm = opt.swarm
    opt.bh.memory = swarm.position
    opt.vh.memory = swarm.position
    swarm.pbest_cost = np.full(opt.swarm_size[0], np.
    inf)

    for i in range(maxiters):
        # Compute cost for current position and personal best
        swarm.current_cost =  compute_objective_function(swarm, objective_func)
        swarm.pbest_pos, swarm.pbest_cost = compute_pbest(swarm)
        

        # Set best_cost_yet_found for ftol
        best_cost_yet_found = swarm.best_cost
        swarm.best_pos, swarm.best_cost = opt.top.compute_gbest(swarm)
        # Perform options update
        swarm.options = opt.oh( opt.options, iternow=i, itermax=maxiters, end_opts=end_opts )
        print("Iteration:", i," Options: ", swarm.options)    # print to see variation
        # Perform velocity and position updates
        swarm.velocity = opt.top.compute_velocity(
            swarm, opt.velocity_clamp, opt.vh, opt.bounds
        )
        swarm.position = opt.top.compute_position(
            swarm, opt.bounds, opt.bh
        )
    # Obtain the final best_cost and the final best_position
    final_best_cost = swarm.best_cost.copy()
    final_best_pos = swarm.pbest_pos[
        swarm.pbest_cost.argmin()
    ].copy()
    return final_best_cost, final_best_pos

# optimize(rosenbrock)