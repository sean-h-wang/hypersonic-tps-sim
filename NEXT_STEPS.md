# NEXT WORK SESSION PLAN
*Project: 1D Hypersonic Ablation Solver*  
*Repo: hypersonic-tps-sim*

---

## 1. Modularization Reminders
- [ ] **Solver module (`fd_solver.py`)**:  
    - Ensure it only integrates the governing equations and returns results.  
    - No plotting or file writing inside this module.  
- [ ] **Data I/O module (`hdf5_io.py`)**:  
    - Responsible for saving solver outputs to HDF5 files.  
    - Include metadata: material, Mach, altitude, N, dt, Picard iteration counts.  
- [ ] **Visualization module (`viz_analysis.ipynb` or `plot_utils.py`)**:  
    - Reads HDF5 files, generates plots/animations without rerunning solver.  
    - Interpolation can be applied here for smoother curves, but **do not store interpolated points in HDF5**.  

---

## 2. Solver Output Strategy
- [ ] **Uniform temporal storage**:
    - Store all results at **1 ms intervals**.  
    - Use `solve_ivp` dense output or loop over `sol(t)` at the desired time steps.  
    - **Do not store the `sol` object or polynomial coefficients**.  
- [ ] **Spatial storage**:
    - Store only the `N+1 = 201` grid points used by the solver.  
    - **No interpolated points**; perform interpolation on-the-fly in visualization module if needed.  
- [ ] Ensure consistency for future expansions (spectral/DG):
    - Same HDF5 structure so all solvers’ outputs can be compared directly.  
