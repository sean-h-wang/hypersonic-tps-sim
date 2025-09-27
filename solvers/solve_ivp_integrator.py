import numpy as np
from scipy.integrate import solve_ivp

def solve_ivp_integrator(rhs_func, params, strategy: str):
    
    """
    Solve the heat equation using Finite Difference Method of Lines

    Args: 
        rhs_func: function defining the right-hand side of the ODE system
        y0: initial state vector [T0, T1, ..., TN, L]
        t_span: time span for the integration [t0, tf]
        params: dictionary containing all physical parameters
    
    Returns:
        t_sol: time array
        T_sol: temperature solution array (time x space)
        L_sol: leading edge length array
    """

    # Setup outputs:
    t_sol = []
    T_sol = []
    L_sol = []

    # Retrieve Variables
    T0 = params['T0']
    L0 = params['L0']
    N = params['N']
    t_max = params['t_max']

    # Initial condition
    y0 = np.concatenate([T0 * np.ones(N + 1), [L0]])   # Combined initial state
    tspan = [0, t_max]

    def ablation_complete(t, y, params):
        return y[-1] - L0/10  # L - L0/10 = 0

    ablation_complete.terminal = True
    ablation_complete.direction = -1  # Trigger when decreasing

    try:
        sol = solve_ivp(rhs_func, tspan, y0, method=strategy, events=ablation_complete, dense_output=True, args=(params, ), rtol=1e-8, atol=1e-10)

        if sol.success and len(sol.t) > 0:
            t_sol = sol.t
            y_sol = sol.y.T

            T_sol = y_sol[:, :-1]
            L_sol = y_sol[:, -1]
        else:
            print(f"Radau failed: {sol.message}")
            return None, None, None
        
    except Exception as e:
        print(f"Radau encountered an error: {e}")
        return None, None, None
    
    return t_sol, T_sol, L_sol