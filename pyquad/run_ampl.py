import io, pickle
from amplpy import AMPL
from contextlib import redirect_stdout
import numpy as np

def solve_model_6dof(params, ampl_mdl_path, hide_solver_output=True, solver="snopt"):
    ampl = AMPL() # ampl installation directory should be in system search path
    # .mod file
    ampl.read(ampl_mdl_path)

    # set parameter values
    for lbl, val in params.items():
        ampl.getParameter(lbl).set(val)

    # set the solver
    ampl.eval("option solver "+solver+";")


    # solve
    if hide_solver_output:
        with redirect_stdout(None):
            ampl.solve()
    else:
        ampl.solve()

    optimal_sol_found = _ampl_optimal_sol_found(ampl)

    traj = None
    objective = None
    if optimal_sol_found:
        traj = _extract_trajectory_from_solver_6dof(ampl)
        objective = ampl.getObjective('myobjective').value()

    ampl.close()
    return optimal_sol_found, traj, objective

def solve_model_5dof(params, ampl_mdl_path, hide_solver_output=True, solver="snopt"):
    ampl = AMPL() # ampl installation directory should be in system search path
    # .mod file
    ampl.read(ampl_mdl_path)

    # set parameter values
    for lbl, val in params.items():
        ampl.getParameter(lbl).set(val)

    # set the solver
    ampl.eval("option solver "+solver+";")

    # solve
    if hide_solver_output:
        with redirect_stdout(None):
            ampl.solve()
    else:
        ampl.solve()

    optimal_sol_found = _ampl_optimal_sol_found(ampl)

    traj = None
    objective = None
    if optimal_sol_found:
        traj = _extract_trajectory_from_solver_5dof(ampl)
        objective = ampl.getObjective('myobjective').value()

    ampl.close()
    return optimal_sol_found, traj, objective





def _ampl_optimal_sol_found(ampl):
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        ampl.display('solve_result')

    result_status = stdout.getvalue().rstrip()
    optimal_sol_found = (result_status == 'solve_result = solved')
    return optimal_sol_found


def _extract_trajectory_from_solver_6dof(ampl):
    # extract required variables from ampl
    nodes = int(ampl.getParameter('n').get())
    timegrid = [val for (_,val) in ampl.getVariable('timegrid').getValues().toList()]
    dt = ampl.getVariable('dt').getValues().toList()[0]

    y_arr = [val for (_,val) in ampl.getVariable('y').getValues().toList()]
    ym_arr = [val for (_,val) in ampl.getVariable('ym').getValues().toList()]
    vy_arr = [val for (_,val) in ampl.getVariable('vy').getValues().toList()]
    vym_arr = [val for (_,val) in ampl.getVariable('vym').getValues().toList()]
    z_arr = [val for (_,val) in ampl.getVariable('z').getValues().toList()]
    zm_arr = [val for (_,val) in ampl.getVariable('zm').getValues().toList()]
    vz_arr = [val for (_,val) in ampl.getVariable('vz').getValues().toList()]
    vzm_arr = [val for (_,val) in ampl.getVariable('vzm').getValues().toList()]
    theta_arr = [val for (_,val) in ampl.getVariable('theta').getValues().toList()]
    thetam_arr = [val for (_,val) in ampl.getVariable('thetam').getValues().toList()]
    omega_arr = [val for (_,val) in ampl.getVariable('omega').getValues().toList()]
    omegam_arr = [val for (_,val) in ampl.getVariable('omegam').getValues().toList()]
    ul_arr = [val for (_,val) in ampl.getVariable('ul').getValues().toList()]
    ulm_arr = [val for (_,val) in ampl.getVariable('ulm').getValues().toList()]
    ur_arr = [val for (_,val) in ampl.getVariable('ur').getValues().toList()]
    urm_arr = [val for (_,val) in ampl.getVariable('urm').getValues().toList()]

    # number of points along trajectory = 2*nodes-1
    traj_arr = np.zeros(shape=(2*nodes-1, 9))
    for i in np.arange(nodes-1):
        traj_arr[2*i,:] = np.asarray([timegrid[i],
                                      y_arr[i],
                                      vy_arr[i],
                                      z_arr[i],
                                      vz_arr[i],
                                      theta_arr[i],
                                      omega_arr[i],
                                      ul_arr[i],
                                      ur_arr[i]])
        traj_arr[2*i+1,:] = np.asarray([timegrid[i] + dt/2.0,
                                        ym_arr[i],
                                        vym_arr[i],
                                        zm_arr[i],
                                        vzm_arr[i],
                                        thetam_arr[i],
                                        omegam_arr[i],
                                        ulm_arr[i],
                                        urm_arr[i]])
    traj_arr[-1,:] = np.asarray([timegrid[-1],
                                 y_arr[-1],
                                 vy_arr[-1],
                                 z_arr[-1],
                                 vz_arr[-1],
                                 theta_arr[-1],
                                 omega_arr[i],
                                 ul_arr[-1],
                                 ur_arr[-1]])
    return traj_arr

def _extract_trajectory_from_solver_5dof(ampl):
    # extract required variables from ampl
    nodes = int(ampl.getParameter('n').get())
    timegrid = [val for (_,val) in ampl.getVariable('timegrid').getValues().toList()]
    dt = ampl.getVariable('dt').getValues().toList()[0]

    y_arr = [val for (_,val) in ampl.getVariable('y').getValues().toList()]
    ym_arr = [val for (_,val) in ampl.getVariable('ym').getValues().toList()]
    vy_arr = [val for (_,val) in ampl.getVariable('vy').getValues().toList()]
    vym_arr = [val for (_,val) in ampl.getVariable('vym').getValues().toList()]
    z_arr = [val for (_,val) in ampl.getVariable('z').getValues().toList()]
    zm_arr = [val for (_,val) in ampl.getVariable('zm').getValues().toList()]
    vz_arr = [val for (_,val) in ampl.getVariable('vz').getValues().toList()]
    vzm_arr = [val for (_,val) in ampl.getVariable('vzm').getValues().toList()]
    theta_arr = [val for (_,val) in ampl.getVariable('theta').getValues().toList()]
    thetam_arr = [val for (_,val) in ampl.getVariable('thetam').getValues().toList()]
    u1_arr = [val for (_,val) in ampl.getVariable('u1').getValues().toList()]
    u1m_arr = [val for (_,val) in ampl.getVariable('u1m').getValues().toList()]
    u2_arr = [val for (_,val) in ampl.getVariable('u2').getValues().toList()]
    u2m_arr = [val for (_,val) in ampl.getVariable('u2m').getValues().toList()]

    # number of points along trajectory = 2*nodes-1
    traj_arr = np.zeros(shape=(2*nodes-1, 8))
    for i in np.arange(nodes-1):
        traj_arr[2*i,:] = np.asarray([timegrid[i],
                                      y_arr[i],
                                      vy_arr[i],
                                      z_arr[i],
                                      vz_arr[i],
                                      theta_arr[i],
                                      u1_arr[i],
                                      u2_arr[i]])
        traj_arr[2*i+1,:] = np.asarray([timegrid[i] + dt/2.0,
                                        ym_arr[i],
                                        vym_arr[i],
                                        zm_arr[i],
                                        vzm_arr[i],
                                        thetam_arr[i],
                                        u1m_arr[i],
                                        u2m_arr[i]])
    traj_arr[-1,:] = np.asarray([timegrid[-1],
                                 y_arr[-1],
                                 vy_arr[-1],
                                 z_arr[-1],
                                 vz_arr[-1],
                                 theta_arr[-1],
                                 u1_arr[-1],
                                 u2_arr[-1]])
    return traj_arr


'''
    n_repeats is the number of OCPs that will be attempted
    final number of optimal trajectories will be <= n_repeats
    n_jobs is number of CPUs to use
'''
def generate_trajectories(n_repeats, init_conds_range, ampl_mdl_path,
                            quad_params={}, n_jobs=1, get_trajectory=solve_model_6dof):
    # rnd number generators use half-open interval
    # instead use next float after upper limit to give U[low, high]
    get_upper_limit = lambda high : np.nextafter(high, high+1)

    # sample initial conditions
    x0 = np.random.uniform(low=init_conds_range['x'][0],
                           high=get_upper_limit(init_conds_range['x'][1]),
                           size=(n_repeats,1))
    z0 = np.random.uniform(low=init_conds_range['z'][0],
                           high=get_upper_limit(init_conds_range['z'][1]),
                          size=(n_repeats,1))
    vx0 = np.random.uniform(low=init_conds_range['vx'][0],
                            high=get_upper_limit(init_conds_range['vx'][1]),
                           size=(n_repeats,1))
    vz0 = np.random.uniform(low=init_conds_range['vz'][0],
                            high=get_upper_limit(init_conds_range['vz'][1]),
                           size=(n_repeats,1))
    theta0 = np.random.uniform(low=init_conds_range['theta'][0],
                               high=get_upper_limit(init_conds_range['theta'][1]),
                              size=(n_repeats,1))

    # construct list of params
    init_conds_arr = np.concatenate((x0,vx0,z0,vz0,theta0), axis=1)
    init_conds_lst = [init_conds_arr[i,:] for i in range(n_repeats)]
    ampl_params_lst = [{**{'x0':x0, 'z0':z0, 'vx0':vx0, 'vz0':vz0, 'theta0':theta0},
                        **quad_params}
                        for (x0,vx0,z0,vz0,theta0) in init_conds_lst]

    # solve trajectories in parallel
    sol_lst = Parallel(n_jobs)(delayed(get_trajectory)(params, ampl_mdl_path) for params in ampl_params_lst)

    # indices where solution found
    opt_sol_found_lst = [sol[0] for sol in sol_lst]
    indices_lst = [idx for idx, elem in enumerate(opt_sol_found_lst) if elem==True]

    # filter out elems where no solution found
    init_conds_arr_filtered = init_conds_arr[indices_lst]
    trajs = [sol[1] for sol in sol_lst]
    traj_arr = np.array([trajs[idx] for idx in indices_lst])
    objs = [sol[2] for sol in sol_lst]
    objs_arr = np.array([objs[idx] for idx in indices_lst])

    return traj_arr, init_conds_arr_filtered, objs_arr

