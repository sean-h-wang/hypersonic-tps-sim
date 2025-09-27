import numpy as np

def solve_ivp_integrator(rhs_func, y0, t_span, params):
    
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

    def ablation_complete(t, y):
        return y[-1] - L0/10  # L - L0/10 = 0

    ablation_complete.terminal = True
    ablation_complete.direction = -1  # Trigger when decreasing