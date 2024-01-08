import numpy as np
import contextlib
import joblib
import sys
import io
import os
import glob
import copy
import shutil
import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.art3d as art3d

from matplotlib.patches import Circle, PathPatch, Rectangle
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from progressbar import progressbar
from amplpy import AMPL

__spatial_cmap_vals = [
    (0.19607843, 0.6, 0.8627451),
    (0.6, 0.19607843, 0.8627451),
]

__spatial_cmap_vals = {key: [(val1, val2, val2) for val1,val2 in zip([0.0, 1.0], val)] for key, val in zip(
    "red green blue".split(" "),
    np.transpose(__spatial_cmap_vals)
)}

__temporal_cmap_vals = [
    (0.52941176, 0.80784314, 0.8627451),
    (0.80784314, 0.52941176, 0.8627451),
] 

__temporal_cmap_vals = {key: [(val1, val2, val2) for val1,val2 in zip([0.0, 1.0], val)] for key, val in zip(
    "red green blue".split(" "),
    np.transpose(__temporal_cmap_vals)
)}

__control_cmap_vals = [
    (0.07843137, 0.23529412, 0.8627451),
    (0.23529412, 0.07843137, 0.8627451),
]

__control_cmap_vals = {key: [(val1, val2, val2) for val1,val2 in zip([0.0, 1.0], val)] for key, val in zip(
    "red green blue".split(" "),
    np.transpose(__control_cmap_vals)
)}

# matplotlib.cm.register_cmap(name='spatial_vars', data=__spatial_cmap_vals, lut=1024)
# matplotlib.cm.register_cmap(name='temporal_vars', data=__temporal_cmap_vals, lut=1024)
# matplotlib.cm.register_cmap(name='control_vars', data=__control_cmap_vals, lut=1024)

# matplotlib.use('agg') # allow importing within shell
import matplotlib.pyplot as plt

plot_vars = [
    ("t", "y"),
    ("t", "vy"),
    ("t", "z"),
    ("t", "vz"),
    ("t", "theta"),
    ("t", "omega"),
    ("y", "z"),
    ("t", "ul"),
    ("t", "ur"),
]

axis_labels = [
    ("t (s)", "y (m)"),
    ("t (s)", r"$v_y$ (m/s)"),
    ("t (s)", "z (m)"),
    ("t (s)", r"$v_z$ (m/s)"),
    ("t (s)", r"$\theta$ (rad)"),
    ("t (s)", r"$\omega$ (rad/s)"),
    ("y (m)", "z (m)"),
    ("t (s)", r"$u_L$ (F%)"),
    ("t (s)", r"$u_R$ (F%)"),
]

plot_vars_12dof = [
    ('t', 'x'),
    ('t', 'vx'),
    ('t', 'y'),
    ('t', 'vy'),
    ('t', 'z'),
    ('t', 'vz'),
    ('t', 'phi'), 
    ('t', 'theta'), 
    ('t', 'psi'), 
    ('t', 'dphi'), 
    ('t', 'dtheta'), 
    ('t', 'dpsi'), 
    ('y', 'z'),
    ('x', 'y'),
    ('z', 'x'),
]

axis_labels_12dof = [
    ('t (s)', 'x (m)'),
    ('t (s)', '$v_x$ (m/s)'), 
    ('t (s)', 'y (m)'),
    ('t (s)', '$v_y$ (m/s)'), 
    ('t (s)', 'z (m)'),
    ('t (s)', '$v_z$ (m/s)'), 
    ('t (s)', '$\\phi$ (rad)'),
    ('t (s)', '$\\theta$ (rad)'),
    ('t (s)', '$\\psi$ (rad)'),
    ('t (s)', '$\\mathrm{d}\\phi/\\mathrm{d}t$ (rad)'),
    ('t (s)', '$\\mathrm{d}\\theta/\\mathrm{d}t$ (rad)'),
    ('t (s)', '$\\mathrm{d}\\psi/\\mathrm{d}t$ (rad)'),
    ('y (m)', 'z (m)'),
    ('x (m)', 'y (m)'),
    ('z (m)', 'x (m)'),
]

def numpy_multiand(*args):
    if len(args) == 2:
        return np.logical_and(args[0], args[1])
    elif len(args)%2 == 1:
        return np.logical_and(args[0], numpy_multiand(*args[1:]))
    else:
        return np.logical_and(numpy_multiand(*args[:len(args)//2]), numpy_multiand(*args[len(args)//2:]))
    
def numpy_multior(*args):
    if len(args) == 2:
        return np.logical_or(args[0], args[1])
    elif len(args)%2 == 1:
        return np.logical_or(args[0], numpy_multior(*args[1:]))
    else:
        return np.logical_or(numpy_multior(*args[:len(args)//2]), numpy_multior(*args[len(args)//2:]))

def flatten_nested_list(nested_list):
    flat_list = []
    for i in nested_list:
        if not isinstance(i, str):
            try:
                len(i)
                flat_list.extend(flatten_nested_list(i))
            except:
                flat_list.append(i)
        else:
            flat_list.append(i)
    return flat_list

def __rotation_matrix_from_xyz(phi, theta, psi):
    c1 = np.cos(phi)
    s1 = np.sin(phi)
    c2 = np.cos(theta)
    s2 = np.sin(theta)
    c3 = np.cos(psi)
    s3 = np.sin(psi)
    
    return np.array([
        [c2 * c3,               -c2 * s3,                -s2     ],
        [c1 * s3 - c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
        [s1 * s3 - c1 * c3 * s2, c3 * s1 - c1 * s2 * s3,  c1 * c2]
    ])

rotation_matrix_from_xyz = np.vectorize(__rotation_matrix_from_xyz, signature="(),(),()->(3,3)")

def rotation_matrix(d):
    """
    Calculates a rotation matrix given a vector d. The direction of d
    corresponds to the rotation axis. The length of d corresponds to 
    the sin of the angle of rotation.

    Variant of: http://mail.scipy.org/pipermail/numpy-discussion/2009-March/040806.html
    """
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        return np.identity(3)

    d = d/sin_angle

    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array([[    0,  d[2],  -d[1]],
                  [-d[2],     0,  d[0]],
                  [d[1], -d[0],    0]], dtype=np.float64)

    M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return M

def pathpatch_2d_to_3d(pathpatch, z = 0, normal = 'z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str: #Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1,0,0), index)

    normal = normal/np.linalg.norm(normal) #Make sure the vector is normalised

    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path) #Apply the transform
    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color    

    verts = path.vertices #Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1)) #Obtain the rotation vector   
    M = rotation_matrix(d) #Get the rotation matrix
    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])
    
def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta

def _get_cmap(var_names):
    if "u" in var_names[1]:
        return "control_vars"
    elif "timegrid" in var_names or "t" in var_names:
        return "temporal_vars"
    else:
        return "spatial_vars"

def _create_fig_axes(fig=None,axes=None,var_names=plot_vars):
    num_subplots = len(var_names)
    num_subplots_side = np.sqrt(num_subplots)
    if np.ceil(num_subplots_side) * np.floor(num_subplots_side) >= num_subplots:
        num_subplots_side = (np.ceil(num_subplots_side), np.floor(num_subplots_side))
    else:
        num_subplots_side = (np.ceil(num_subplots_side), np.ceil(num_subplots_side))

    num_subplots_side = tuple((int(i) if i > 0 else i+1 for i in num_subplots_side))
    
    if fig is None and axes is None:
        fig = plt.figure()
    elif fig is None and axes is not None:
        fig = axes[0].get_figure()
    
    if axes is None or len(axes) != len(var_names):
        fig.clf()
        axes = []
        for i in range(num_subplots_side[0]):
            for j in range(num_subplots_side[1]):
                var_idx = i*num_subplots_side[1] + j
                if var_idx >= len(var_names):
                    break
                axes.append(fig.add_subplot(num_subplots_side[0],num_subplots_side[1],var_idx + 1))

    return fig, axes, num_subplots_side

def _add_axis_lbl(axes, var_names, axis_labels, var_idx):
    if axis_labels and var_idx < len(axis_labels):
        if len(axis_labels[var_idx]) == 0:
            axes[var_idx].set_xlabel(var_names[var_idx][0])
            axes[var_idx].set_ylabel(var_names[var_idx][1])
        elif len(axis_labels[var_idx]) == 1:
            axes[var_idx].set_xlabel(axis_labels[var_idx][0])
            axes[var_idx].set_ylabel(var_names[var_idx][1])
        elif len(axis_labels[var_idx]) == 2:
            axes[var_idx].set_xlabel(axis_labels[var_idx][0])
            axes[var_idx].set_ylabel(axis_labels[var_idx][1])
        elif len(axis_labels[var_idx]) == 3:
            axes[var_idx].set_xlabel(axis_labels[var_idx][0])
            axes[var_idx].set_ylabel(axis_labels[var_idx][1])
            axes[var_idx].set_title(axis_labels[var_idx][2])
    else:
        axes[var_idx].set_xlabel(var_names[var_idx][0])
        axes[var_idx].set_ylabel(var_names[var_idx][1])
    
# Taken and modified from https://stackoverflow.com/a/50029441/6170161
def multiline(xs, ys, c=None, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
#     print(xs.shape, np.atleast_2d(xs).shape)
#     print(ys.shape, np.atleast_2d(ys).shape)
#     print(xs, ys)
    segments = [np.column_stack([x, y]) for x, y in zip(np.atleast_2d(xs), np.atleast_2d(ys))]
    
    num_segs = len(segments)
    
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    if c is not None:
        lc.set_array(np.asarray(np.atleast_1d(c)))
    else:
        lc.set_array(np.linspace(0, 1, num_segs))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc

def ampl_trajectory_plotter(var_dict, var_names=plot_vars, fig=None, axes=None, axis_labels=axis_labels, **plot_kwargs):
    fig, axes, num_subplots_side = _create_fig_axes(fig, axes, var_names)
    for i in range(num_subplots_side[0]):
        for j in range(num_subplots_side[1]):
            var_idx = i*num_subplots_side[1] + j
            if var_idx < len(var_names):
                cmap = _get_cmap(var_names[var_idx])

                x_vals = var_dict[var_names[var_idx][0]]
                y_vals = var_dict[var_names[var_idx][1]]

                lc = multiline(
                    x_vals,
                    y_vals,
                    cmap=cmap,
                    ax=axes[var_idx],
                    **plot_kwargs
                )
                _add_axis_lbl(axes, var_names, axis_labels, var_idx)
        else:
            continue
        break
    plt.tight_layout()
    return fig, axes

# default arguments for bebop_6dof
# frame delay is time in milliseconds between frames (or rows in traj_arr)
def trajectory_animate(traj_arr, pos_lim=20, thrust_min=1.76, thrust_max=2.35, frame_delay=25):
    fig = plt.figure(figsize=(9,5))

    ys = traj_arr['y']
    zs = traj_arr['z']
    vys = traj_arr['vy']
    vzs = traj_arr['vz']
    thetas = traj_arr['theta']
    omegas = traj_arr['omega']
    times = traj_arr['t']
    u1s = traj_arr['ul']
    u2s = traj_arr['ur']
    
    max_time = traj_arr['tf']
    
    # coordinates for quad visualisation
    vis_quad_len = 1
    ys_r = ys + vis_quad_len * np.cos(thetas)
    ys_l = ys - vis_quad_len * np.cos(thetas)
    zs_r = zs + vis_quad_len * np.sin(thetas)
    zs_l = zs - vis_quad_len * np.sin(thetas)

    ax = fig.add_subplot(121, autoscale_on=False, xlim=(-pos_lim-1.0, pos_lim+1.0), ylim=(-pos_lim-1.0, pos_lim+1.0))
    ax.grid()

    quad, = ax.plot([], [], 'o-', lw=2)
    traj, = ax.plot([], [], '-', lw=1.2)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    ax = fig.add_subplot(222, autoscale_on=False, xlim=(0, np.ceil(max_time)), ylim=(thrust_min-0.1, thrust_max+0.5))
    Lthrust, = ax.plot([], [], '-')
    ax.set_xlabel('t')
    ax.set_ylabel('left thrust (N)')

    ax = fig.add_subplot(224, autoscale_on=False, xlim=(0, np.ceil(max_time)), ylim=(thrust_min-0.1, thrust_max+0.5))
    Rthrust, = ax.plot([], [], '-')
    ax.set_xlabel('t')
    ax.set_ylabel('right thrust (N)')

    fig.tight_layout()

    def next_i():
        while anim.fr_num < len(times):
            anim.fr_num += anim.direction
            anim.fr_num = abs(anim.fr_num%len(times))
            if anim.back2beginning:
                anim.fr_num = 0
                anim.back2beginning = False
                if anim.running:
                    anim.event_source.stop()
                anim.running = False
            yield anim.fr_num
    
    def init():
        quad.set_data([], [])
        traj.set_data([],[])
        Lthrust.set_data([], [])
        Rthrust.set_data([], [])
        time_text.set_text('')
        return quad,traj,Lthrust,Rthrust,time_text

    pause = False

    def animate(i):
        quad.set_data([ys_l[i], ys_r[i]], [zs_l[i], zs_r[i]])
        traj.set_data(ys[:i], zs[:i])
        Lthrust.set_data(times[:i], u1s[:i]*(thrust_max - thrust_min) + thrust_min)
        Rthrust.set_data(times[:i], u2s[:i]*(thrust_max - thrust_min) + thrust_min)
        time_text.set_text('time = {:.2f}s'.format(times[i])) # not working
        return quad,traj,Lthrust,Rthrust,time_text

    def on_press(event):
        if event.key.isspace():
            if anim.running:
                anim.event_source.stop()
            else:
                anim.event_source.start()
            anim.running ^= True
        elif event.key == 'left':
            anim.direction = -1
        elif event.key == 'right':
            anim.direction = +1
        elif event.key == 'r':
            anim.back2beginning = True

        # Manually update the plot
        if event.key in ['left','right']:
            t = anim.frame_seq.next()
            update_plot(t)
            plt.draw()

    fig.canvas.mpl_connect('key_press_event', on_press)

    anim = animation.FuncAnimation(fig, animate, next_i,
                                   interval=frame_delay, blit=True, repeat=False, init_func=init)
    
    anim.running = True
    anim.direction = 1
    anim.back2beginning = False
    anim.fr_num = 0

    return anim

def eprint(*args, **kwargs):
    # Prints errors to the stderr stream in order to allow
    # error messages to be printed to the right place.
    print(*args, file=sys.stderr, **kwargs)

def str_shorthand(x):
    # Creates an acronym for a string by combining the uppercase letters
    # of the string.
    return "".join([i for i in x if i.isupper()])

def extract_path_ext(filepath, default_ext=".csv"):
    # Extracts and returns the path, the name, and the extensions of
    # the file specified by the string filepath.
    return "/".join(filepath.split("/")[:-1]) + "/", filepath.split("/")[-1], "." + filepath.split("/")[-1].split(".")[-1] if len(filepath.split("/")[-1].split(".")) > 1 else default_ext

def atleast_nd(x, n=1):
    # Extension of numpy's atleast_1d,atleast_2d, and atleast_3d
    # to arbitrary dimensions. Appends dimensions based on the
    # parity of the existing dimensions of the array.
    if x.ndim >= n:        return x
    elif n == 1:
        return np.atleast_1d(x)
    elif n == 2:
        return np.atleast_2d(x)
    elif n == 3:
        return np.atleast_3d(x)
    else:
        return np.expand_dims(atleast_nd(x, n=n-1), axis=(0 if (n-x.ndim)%2 == 0 else -1))

class AMPLModel(object):
    def __init__(self, ampl_mod_path):
        self.ampl = AMPL()
        self.ampl.read(ampl_mod_path)
        self.ampl_mod_path = ampl_mod_path
        
        self.__parameters_that_have_been_set = dict()
        self.__pristine = True
        self.__ampl_output = None
        self.auto_reset = True
        self.reset()
        
    def getObjectives(self):
        return self.ampl.getObjectives()
        
    def getObjective(self, name):
        return self.ampl.getObjective(name)
        
    def getVariables(self):
        return self.ampl.getVariables()
        
    def getVariable(self, name):
        return self.ampl.getVariable(name)
    
    def getParameters(self):
        return self.ampl.getParameters()
    
    def getParameter(self, name):
        return self.ampl.getParameter(name)
    
    def getAMPL(self):
        return self.ampl
    
    def display(self, *args):
        return self.ampl.display(*args)
    
    def reset(self):
        if not self.__pristine or True:
            self.ampl.reset()
            self.ampl.read(self.ampl_mod_path)
            self.parameter_names = {i[0]:i[1] for i in self.ampl.getParameters()}
            self.variable_names  = {i[0]:i[1] for i in self.ampl.getVariables()}

            if "timegrid" in self.variable_names:
                self.variable_names = {"t": self.variable_names['timegrid'], **self.variable_names}
            self.objective_names = {i[0]:i[1] for i in self.ampl.getObjectives()}

            self.special_variables = {key: (key, key+"m") for key in self.variable_names if key+"m" in self.variable_names}
            self._special_vars_key_set = {val[0] for _,val in self.special_variables.items()} | {val[1] for _,val in self.special_variables.items()}

            self._keys = list(self.keys())
            self.solved = False
            self.__pristine = True
            self.__updateUnderlyingValues()
    
    def checkSolved(self):
        if not self.solved:
            ampl_out = io.StringIO()
            with contextlib.redirect_stdout(ampl_out):
                self.ampl.display("solve_result")
            result_status = ampl_out.getvalue().rstrip()
            if result_status == "solve_result = solved":
                self.solved = True
            else:
                self.solved = False
            self.__pristine = False
        return self.solved
    
    def solve(self, hide_solver_output=True):
        try:
            if not self.checkSolved() or not self.auto_reset:
                self.__ampl_output = io.StringIO()
                if hide_solver_output:
                    with contextlib.redirect_stdout(self.__ampl_output):
                        self.ampl.solve()
                else:
                    self.ampl.solve()
            return self.checkSolved()
        except KeyboardInterrupt:
            return self.checkSolved()
        except:
            raise
    
    def solveAsync(self, callback, hide_solver_output=True):
        if not self.checkSolved():
            self.__ampl_output = io.StringIO()
            if hide_solver_output:
                with contextlib.redirect_stdout(self.__ampl_output):
                    return self.ampl.solveAsync(callback)
            else:
                return self.ampl.solveAsync(callback)
            
        
    def showAMPLOutput(self):
        return self.__ampl_output.getvalue().rstrip()
        
    def getSolutionValues(self, name_list=None):
        if not self.solved:
            self.solve()
        if name_list is not None:
            if isinstance(name_list, str):
                name_list = [name_list]
            name_list = {i for i in name_list if i in self.variable_names}
        else:
            name_list = set(self.variable_names.keys())
        name_is_special = name_list & set(self.special_variables.keys())
        return self[name_list]
        
    def getVariableValues(self, name_list=None):
        if name_list is not None:
            if isinstance(name_list, str):
                name_list = [name_list]
            name_list = {i for i in name_list if i in self.variable_names}
        else:
            name_list = set(self.variable_names.keys())
        name_is_special = name_list & set(self.special_variables.keys())
        return {
            **self.getSpecialVariable(name_is_special),
            **{key: self.getVariableValue(key) for key in name_list - name_is_special}
        }
        
    def getParameterValues(self, name_list=None):
        if name_list is not None:
            if isinstance(name_list, str):
                name_list = [name_list]
            return {key: self.getParameterValue(key) for key in name_list if key in self.parameter_names}
        else:
            return {key: self.getParameterValue(key) for key in self.parameter_names}
        
    def getObjectiveValues(self, name_list=None):
        if name_list is not None:
            if isinstance(name_list, str):
                name_list = [name_list]
            return {key: self.getObjectiveValue(key) for key in name_list if key in self.objective_names}
        else:
            return {key: self.getObjectiveValue(key) for key in self.objective_names}
        
    def getVariableValue(self, var_name):
        if self.variable_names[var_name].indexarity() == 0:
            return self.variable_names[var_name].getValues().toList()[0]
        else:
            # temp = np.array(self.variable_names[var_name].getValues().toList())
            # temp = temp[:, self.variable_names[var_name].indexarity():]
            temp = np.array([instance.value() for index, instance in self.variable_names[var_name].instances()])
            return temp.reshape(self.getVarShape(var_name))
        
    def getParameterValue(self, param_name):
        if self.parameter_names[param_name].indexarity() == 0:
            return self.parameter_names[param_name].getValues().toList()[0]
        else:
            # temp = np.array(self.parameter_names[param_name].getValues().toList())
            # temp = temp[:, self.parameter_names[param_name].indexarity():]
            temp = np.array([value for index, value in self.parameter_names[param_name].instances()])
            return temp.reshape(self.getParamShape(param_name))
    
    def getObjectiveValue(self, obj_name):
        return np.array(self.objective_names[obj_name].getValues().toList()).reshape(self.getObjectiveShape(obj_name))
        
    def getSpecialVariable(self, name_list=None):
        return_dict = dict()
        if name_list is not None:
            if isinstance(name_list, str):
                if name_list in self.special_variables:
                    return self.coerceSpecialVariable(name_list)
            for i in name_list:
                if i in self.special_variables:
                    return_dict.update({i: self.coerceSpecialVariable(i)})
        else:
            for i in self.special_variables:
                return_dict.update({i: self.coerceSpecialVariable(i)})
        return return_dict
    
    def coerceSpecialVariable(self, var_name):
        endpoint_vals = self.getVariableValue(self.special_variables[var_name][0])
        midpoint_vals = self.getVariableValue(self.special_variables[var_name][1])
        assert(endpoint_vals.shape[1:] == midpoint_vals.shape[1:])
        final_shape_tuple = list(endpoint_vals.shape[1:])
        temp = np.empty((endpoint_vals.shape[0] + midpoint_vals.shape[0], *final_shape_tuple), dtype=endpoint_vals.dtype)
        temp[::2] = endpoint_vals
        temp[1::2] = midpoint_vals
        return temp
    
    def getVarShape(self, var_name):
        ret_tuple = []
        if self.getVariable(var_name).indexarity() == 0:
            return tuple()
        else:
            for i in self.getVariable(var_name).getIndexingSets():
                ret_tuple.append(self.getSetSize(i.split("in")[-1].strip()))
        return tuple(ret_tuple)
    
    def getParamShape(self, param_name):
        ret_tuple = []
        if self.getParameter(param_name).indexarity() == 0:
            return tuple()
        else:
            for i in self.getParameter(param_name).getIndexingSets():
                ret_tuple.append(self.getSetSize(i.split("in")[-1].strip()))
        return tuple(ret_tuple)
    
    def getObjectiveShape(self, obj_name):
        ret_tuple = []
        if self.getObjective(obj_name).indexarity() == 0:
            return tuple()
        else:
            for i in self.getObjective(obj_name).getIndexingSets():
                ret_tuple.append(self.getSetSize(i.split("in")[-1].strip()))
        return tuple(ret_tuple)
    
    def getSetSize(self, set_name):
        return len(self.ampl.getSet(set_name).getValues().toList())
    
    def setParameterValue(self, param_name, param_value):
        if param_name in self.parameter_names:
            if self.auto_reset:
                if param_name not in self.__parameters_that_have_been_set:
                    self.reset()
                elif not np.all(np.isclose(self.__parameters_that_have_been_set[param_name], param_value, rtol=1e-10, atol=1e-12)):
                    self.reset()
            self.__parameters_that_have_been_set[param_name] = param_value
        return self.__updateUnderlyingValues()
            
    def setParameterValues(self, param_dict=dict(), **kwargs):
        for key,val in {**param_dict, **kwargs}.items():
            self.setParameterValue(key, val)
        return self
    
    def __updateUnderlyingValue(self, param_name, param_value):
        if param_name in self.parameter_names:
            self.ampl.param[param_name] = param_value
        return self
    
    def __updateUnderlyingValues(self):
        for key, val in self.__parameters_that_have_been_set.items():
            self.__updateUnderlyingValue(key, val)
        return self
            
    def __getitem__(self, key):
        if isinstance(key, int):
            return self._keys[key], self[self._keys[key]]
        elif isinstance(key, str):
            if key in self.parameter_names:
                return self.getParameterValue(key)
            elif key == "timegrid" or key == "t":
                num_nodes = int(self['n'])
                x_vals = np.empty((num_nodes*2 - 1,), dtype=np.float64)
                x_vals[::2] = self.getVariableValue("timegrid")
                x_vals[1::2] = self.getVariableValue("timegrid")[:-1] + self.getVariableValue("dt") / 2.0
                return x_vals
            elif key in self.special_variables:
                return self.getSpecialVariable(key)
            elif key in self.variable_names:
                return self.getVariableValue(key)
            elif key in self.objective_names:
                return self.getObjectiveValue(key)
            else:
                raise KeyError("key {} not in model".format(key))
        else:
            return {_key: self[_key] for _key in key}
            
    def __setitem__(self, key, val):
        if isinstance(key, str):
            if key in self.parameter_names:
                return self.setParameterValue(key, val)
            else:
                raise KeyError("Key not in parameters!")
        else:
            if isinstance(val, dict):
                if (set(key) & set(val.keys())) != set(key):
                    raise ValueError("Specified keys not in dict of values to assign.")
                for _key in key:
                    self[_key] = val[_key]
            else:
                if len(key) != len(val):
                    raise ValueError("Number of parameters does not match number of values. Received {} values, expected {}.".format(len(val), len(key)))
                for _key,_val in zip(key,val):
                    self[_key] = _val
                
    def items(self):
        return ((key, self[key]) for key in self.keys())
    
    def keys(self):
        return sorted(self.parameter_names.keys()) + \
               sorted(self.special_variables.keys()) + \
               sorted(set(self.variable_names.keys()) - self._special_vars_key_set) + \
               sorted(self.objective_names.keys())
            
    def __iter__(self):
        return (key for key in self.keys())
        
    def __len__(self):
        return len(self.keys())
    
def run_ampl_model(ampl_mod_path, param_dict):
    model = AMPLModel(ampl_mod_path)
    model.setParameterValues(param_dict)
    model.solve()
    soln_vals = model.getSolutionValues()
    obj_vals = model.getObjectiveValues()
    return {"Success": model.checkSolved(), "t": soln_vals['timegrid'], **soln_vals, **obj_vals, **param_dict}

        
class AMPLTrajectoryManager(object):
    def __init__(self, ampl_mod_path, default_param_samples=10):
        temp = AMPLModel(ampl_mod_path)
        
        self.n = temp.getParameterValue("n")
        self.parameter_names = set(i[0] for i in temp.getParameters())        
        self.output_variables = set(temp.special_variables.keys())
        self.output_variables = self.output_variables | set(temp.objective_names.keys())
        self.output_variables = self.output_variables | (set(temp.variable_names.keys()) - temp._special_vars_key_set)
        if "timegrid" in self.output_variables:
            self.output_variables = self.output_variables | {"t",}
        
        self.ampl_model_path = ampl_mod_path
        self.param_ranges = dict()
        self.num_trajectories = None
        self._params_to_update = set()
        self._param_vals = dict()
        self._default_param_samples = default_param_samples
        self.optimised_solution_frame = dict()
        self._keys = self.keys()
        self.solved = False
        
    def setParameterRange(self, param_name, param_bounds):
        if param_name in self.parameter_names:
            if self.solved:
                self.solved = False
            self.param_ranges.update({param_name: tuple(param_bounds)})
            self._params_to_update.add(param_name)
        return self
        
    def setParameter(self, param_name, param_val):
        if not isinstance(param_val, (float, int, complex)):
            raise TypeError("Parameter value should have numeric type. Received type: ".format(type(param_val)))
        return self.setParameterRange(param_name, [param_val, param_val])
    
    def setParameterRanges(self, param_ranges_dict=dict(), **kwargs):
        for key,val in {**param_ranges_dict, **kwargs}.items():
            self.setParameterRange(key, val)
        return self
                
    def generateTrajectories(self, num_trajectories, regenerate_all=False, **kwargs):
        if isinstance(num_trajectories, int):
            if num_trajectories != self.num_trajectories:
                regenerate_all = True
            if not self._params_to_update or regenerate_all:
                for key,val in self.param_ranges.items():
                    self._param_vals.update({key: np.random.uniform(val[0], val[1], num_trajectories)})
            else:
                for key,val in self.param_ranges.items():
                    if key in self._params_to_update:
                        self._param_vals.update({key: np.random.uniform(val[0], val[1], num_trajectories)})
                        self._params_to_update.remove(key)
            if regenerate_all and self._params_to_update:
                self._params_to_update = set()
        else:
            raise TypeError("Expected num_trajectories to have type int but got " + str(type(num_trajectories)))
        self.num_trajectories = num_trajectories
        self.solved = False
        return self
    
    def solveTrajectories(self, njobs=2, Parallel_kwargs={"backend": "loky"}, force_rerun=False):
        if self.num_trajectories is None:
            raise ValueError("You have not generated any trajectories! Call generateTrajectories to generate trajectories to solve in Parallel.")
        if self.solved and not force_rerun:
            return self
        # solve trajectories in parallel
        sol_lst = joblib.Parallel(njobs, **Parallel_kwargs)(
            joblib.delayed(run_ampl_model)(self.ampl_model_path, {key: val[idx] for key,val in self._param_vals.items()}) for idx in progressbar(range(self.num_trajectories)))
        
        self.filtered_solutions = sol_lst
        
        self.optimised_solution_frame = {i: np.stack([val[i] for val in self.filtered_solutions]) for i in self.filtered_solutions[0].keys()}
        
        self.num_trajectories = len(self.filtered_solutions)
        
        if len(self.filtered_solutions) == 0:
            eprint("No trajectories were successful. Run with more trajectories or change paramere ranges.")
        else:
            self.solved = True
        
        return self
    
    def __call__(self, var_names=None, traj_idx=None, **kwargs):
        if not self.solved:
            self.solveTrajectories(**kwargs)
        var_list = []
        if var_names is not None:
            if isinstance(var_names, str):
                var_list.append(var_names)
            else:
                var_list.extend(list(var_names))
        else:
            var_list = list(self.output_variables)
            
        if traj_idx is not None:
            return {key: self[key][traj_idx] for key in self.optimised_solution_frame if key in var_list}
        else:
            return {key: self[key] for key in self.optimised_solution_frame if key in var_list}
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self._keys[key], self[self._keys[key]]
        elif isinstance(key, str):
            if key in self.param_ranges:
                return self.param_ranges[key]
        if not self.solved:
            raise ValueError("Cannot get solution when trajectories have not been solved for! Use __call__ interface to autosolve if trajectories have not been computed.")
        if key not in self.optimised_solution_frame.keys():
            raise KeyError("key {} not in valid set of solution variables".format(key))
        else:
            if isinstance(key, str):
                return self.optimised_solution_frame[key]
            else:
                return {i: self[i] for i in key}
        
    def __setitem__(self, key, val):
        if isinstance(key, str):
            if key in self.parameter_names:
                if isinstance(val, (float, int, complex)):
                    self.setParameterRange(key, [val, val])
                elif len(val) == 2:
                    self.setParameterRange(key, val)
                else:
                    raise ValueError("value must either be a single number or a list of 2 numbers representing value bounds")
            else:
                raise KeyError("key {} not in valid set of parameters".format(key))
        elif isinstance(key, (tuple, list)):
            for _key, _val in zip(key, val):
                if _key in self.parameter_names:
                    if isinstance(_val, (float, int, complex)):
                        self.setParameterRange(_key, [_val, _val])
                    elif len(_val) == 2:
                        self.setParameterRange(_key, _val)
                    else:
                        raise ValueError("value must either be a single number or a list of 2 numbers representing value bounds")
                else:
                    raise KeyError("key {} not in valid set of parameters".format(_key))
                
    def items(self):
        return ((key, self[key]) for key in self.keys())
                
    def keys(self):
        return sorted(self.parameter_names) + \
               sorted(self.optimised_solution_frame.keys())
                
    def __iter__(self):
        return (key for key in self.keys())
        
    def __len__(self):
        return len(self.keys())
    
    def saveAsNumpy(self, filepath, dataset_modifier=None):
        if not self.solved:
            raise ValueError("Cannot save nonexistent data. Call solveTrajectories first.")
            
        save_frame = None
        
        if dataset_modifier is not None:
            if hasattr(dataset_modifier, "__call__"):
                save_frame = dataset_modifier(self.optimised_solution_frame)
        if save_frame is None:
            save_frame = self.optimised_solution_frame
        return np.savez_compressed(filepath, **save_frame)

    
def run_ampl_model_continuation(model, param_dict):
    # only reset model when previous solution failed
    if not model.checkSolved():
        model.reset()
    model.setParameterValues(param_dict)
    model.solve()
    soln_vals = model.getSolutionValues()
    obj_vals = model.getObjectiveValues()
    return {"Success": model.checkSolved(), "t": soln_vals['timegrid'], **soln_vals, **obj_vals}

def run_ampl_model_random_walk(ampl_model_path, fixed_params, param_dict, num, filepath):
    ampl_model = AMPLModel(ampl_model_path)
    ampl_model.setParameterValues(fixed_params)
    ampl_model.auto_reset = False
    trajectories = []
    for idx in progressbar(range(num)):
        traj = run_ampl_model_continuation(ampl_model, {key: val[idx] for key, val in param_dict.items()})
        file = filepath+'/traj'+str(idx) + '.npz'
        trajectories.append(traj)
        np.savez_compressed(file, **traj)
    return trajectories
        

    
def random_walk(minimum, maximum, num, step_size):
#     steps = np.random.normal(scale=step_var, size=(num-1))
    steps = np.random.choice([-step_size, step_size], num-1)
    rwalk = np.zeros(num)
    rwalk[0] = np.random.uniform(minimum, maximum)
    for i in range(1, num):
        new = rwalk[i-1] + steps[i-1]
        if new > maximum or new < minimum:
            rwalk[i] = rwalk[i-1] - steps[i-1]
        else:
            rwalk[i] = new
    return rwalk

    
class AMPLTrajectoryManagerContinuation(object):
    def __init__(self, ampl_mod_path, default_param_samples=10):
        temp = AMPLModel(ampl_mod_path)
        
        self.n = temp.getParameterValue("n")
        self.parameter_names = set(i[0] for i in temp.getParameters())        
        self.output_variables = set(temp.special_variables.keys())
        self.output_variables = self.output_variables | set(temp.objective_names.keys())
        self.output_variables = self.output_variables | (set(temp.variable_names.keys()) - temp._special_vars_key_set)
        if "timegrid" in self.output_variables:
            self.output_variables = self.output_variables | {"t",}
        
        self.ampl_model_path = ampl_mod_path
        self.param_ranges = dict()
        self.fixed_params = dict()
        
        self.num_trajectories = None
        self.num_walks = None
        self.num_steps = None
        
        self._params_to_update = set()
        self._param_vals = dict()
        self._default_param_samples = default_param_samples
        self.optimised_solution_frame = dict()
        self._keys = self.keys()
        self.solved = False
        
    def setFixedParameters(self, param_dict):
        self.fixed_params.update(param_dict)
        
    def getFixedParameters(self):
        params = AMPLModel(self.ampl_model_path).getParameterValues()
        params.update(self.fixed_params)
        return {key: params[key] for key in params.keys() if key not in self._param_vals.keys()}
    
    def setParameterRange(self, param_name, param_bounds):
        if param_name in self.parameter_names:
            if self.solved:
                self.solved = False
            self.param_ranges.update({param_name: tuple(param_bounds)})
            self._params_to_update.add(param_name)
        return self
        
    def setParameter(self, param_name, param_val):
        if not isinstance(param_val, (float, int, complex)):
            raise TypeError("Parameter value should have numeric type. Received type: ".format(type(param_val)))
        return self.setParameterRange(param_name, [param_val, param_val])
    
    def setParameterRanges(self, param_ranges_dict=dict(), **kwargs):
        for key,val in {**param_ranges_dict, **kwargs}.items():
            self.setParameterRange(key, val)
        return self
                
    def generateTrajectories(self, num_trajectories, regenerate_all=False, **kwargs):
        if isinstance(num_trajectories, int):
            if num_trajectories != self.num_trajectories:
                regenerate_all = True
            if not self._params_to_update or regenerate_all:
                for key,val in self.param_ranges.items():
                    self._param_vals.update({key: np.random.uniform(val[0], val[1], num_trajectories)})
            else:
                for key,val in self.param_ranges.items():
                    if key in self._params_to_update:
                        self._param_vals.update({key: np.random.uniform(val[0], val[1], num_trajectories)})
                        self._params_to_update.remove(key)
            if regenerate_all and self._params_to_update:
                self._params_to_update = set()
        else:
            raise TypeError("Expected num_trajectories to have type int but got " + str(type(num_trajectories)))
        self.num_trajectories = num_trajectories
        self.solved = False
        return self
    
    def solveTrajectories(self, njobs=2, Parallel_kwargs={"backend": "loky"}, force_rerun=False):
        if self.num_trajectories is None:
            raise ValueError("You have not generated any trajectories! Call generateTrajectories to generate trajectories to solve in Parallel.")
        if self.solved and not force_rerun:
            return self
        # solve trajectories in parallel
        sol_lst = joblib.Parallel(njobs, **Parallel_kwargs)(
            joblib.delayed(run_ampl_model)(self.ampl_model_path, {key: val[idx] for key,val in self._param_vals.items()}) for idx in progressbar(range(self.num_trajectories)))
        
        
        self.filtered_solutions = sol_lst
        
        self.optimised_solution_frame = {i: np.stack([val[i] for val in self.filtered_solutions]) for i in self.filtered_solutions[0].keys()}
        
        self.num_trajectories = len(self.filtered_solutions)
        
        if len(self.filtered_solutions) == 0:
            eprint("No trajectories were successful. Run with more trajectories or change paramere ranges.")
        else:
            self.solved = True
        
        return self
    
    def generateTrajectoriesRandomWalk(self, num_walks, num_steps, step_sizes, regenerate_all=False, **kwargs):
        if isinstance(num_walks, int) and isinstance(num_steps, int):
            if num_walks*num_steps != self.num_trajectories:
                regenerate_all = True
            if not self._params_to_update or regenerate_all:
                for key,val in self.param_ranges.items():
                    self._param_vals.update({key: np.stack([
                        random_walk(val[0], val[1], num_steps, step_sizes[key]) 
                        for i in range(num_walks)
                    ])})
            else:
                for key,val in self.param_ranges.items():
                    if key in self._params_to_update:
                        self._param_vals.update({key: np.stack([
                            random_walk(val[0], val[1], num_steps, step_sizes[key]) 
                            for i in range(num_walks)
                        ])})
                        self._params_to_update.remove(key)
            if regenerate_all and self._params_to_update:
                self._params_to_update = set()
        else:
            raise TypeError("Expected num_walks and num_steps to have type int but got " + str(type(num_trajectories)))
        self.num_trajectories = num_walks*num_steps
        self.num_walks = num_walks
        self.num_steps = num_steps
        self.solved = False
        return self
    
    def solveTrajectoriesRandomWalk(self, njobs=1, Parallel_kwargs={"backend": "loky"}, force_rerun=False):
        if self.num_trajectories is None:
            raise ValueError("You have not generated any trajectories! Call generateTrajectoriesRandomWalk to generate trajectories to solve in Parallel.")
        if self.solved and not force_rerun:
            return self
        
        # each trajectory will be saved in a file
        filepaths = ['temp_folder/randomwalk' + str(i) for i in range(self.num_walks)]
        if not os.path.exists('temp_folder'):
            os.mkdir('temp_folder')
        for path in filepaths:
            if not os.path.exists(path):
                os.mkdir(path)
        print('solutions will be temporarily saved in temp_folder/..')
                
        # each job has a progressbar
        sol_lst = joblib.Parallel(njobs, **Parallel_kwargs)(
            joblib.delayed(run_ampl_model_random_walk)(
                self.ampl_model_path,
                self.fixed_params,
                {key: val[idx] for key,val in self._param_vals.items()},
                self.num_steps,
                filepaths[idx]
            )
            for idx in range(self.num_walks)
        )
        sol_lst = sum(sol_lst, [])
        
        self.filtered_solutions = sol_lst
        
        self.optimised_solution_frame = {
            i: np.stack([val[i] for val in self.filtered_solutions]) for i in self.output_variables | {"Success"}
        }
        
        self.num_trajectories = len(self.filtered_solutions)
        
        if len(self.filtered_solutions) == 0:
            eprint("No trajectories were successful. Run with more trajectories or change paramere ranges.")
        else:
            self.solved = True
        
        return self
    
    def __call__(self, var_names=None, traj_idx=None, **kwargs):
        if not self.solved:
            self.solveTrajectories(**kwargs)
        var_list = []
        if var_names is not None:
            if isinstance(var_names, str):
                var_list.append(var_names)
            else:
                var_list.extend(list(var_names))
        else:
            var_list = list(self.output_variables)
            
        if traj_idx is not None:
            return {key: self[key][traj_idx] for key in self.optimised_solution_frame if key in var_list}
        else:
            return {key: self[key] for key in self.optimised_solution_frame if key in var_list}
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self._keys[key], self[self._keys[key]]
        elif isinstance(key, str):
            if key in self.param_ranges:
                return self.param_ranges[key]
        if not self.solved:
            raise ValueError("Cannot get solution when trajectories have not been solved for! Use __call__ interface to autosolve if trajectories have not been computed.")
        if key not in self.optimised_solution_frame.keys():
            raise KeyError("key {} not in valid set of solution variables".format(key))
        else:
            if isinstance(key, str):
                return self.optimised_solution_frame[key]
            else:
                return {i: self[i] for i in key}
        
    def __setitem__(self, key, val):
        if isinstance(key, str):
            if key in self.parameter_names:
                if isinstance(val, (float, int, complex)):
                    self.setParameterRange(key, [val, val])
                elif len(val) == 2:
                    self.setParameterRange(key, val)
                else:
                    raise ValueError("value must either be a single number or a list of 2 numbers representing value bounds")
            else:
                raise KeyError("key {} not in valid set of parameters".format(key))
        elif isinstance(key, (tuple, list)):
            for _key, _val in zip(key, val):
                if _key in self.parameter_names:
                    if isinstance(_val, (float, int, complex)):
                        self.setParameterRange(_key, [_val, _val])
                    elif len(_val) == 2:
                        self.setParameterRange(_key, _val)
                    else:
                        raise ValueError("value must either be a single number or a list of 2 numbers representing value bounds")
                else:
                    raise KeyError("key {} not in valid set of parameters".format(_key))
                
    def items(self):
        return ((key, self[key]) for key in self.keys())
                
    def keys(self):
        return sorted(self.parameter_names) + \
               sorted(self.optimised_solution_frame.keys())
                
    def __iter__(self):
        return (key for key in self.keys())
        
    def __len__(self):
        return len(self.keys())
    
    def saveAsNumpy(self, filepath, dataset_modifier=None):
        if not self.solved:
            raise ValueError("Cannot save nonexistent data. Call solveTrajectories first.")
            
        save_frame = None
        
        if dataset_modifier is not None:
            if hasattr(dataset_modifier, "__call__"):
                save_frame = dataset_modifier(self.optimised_solution_frame)
        if save_frame is None:
            save_frame = self.optimised_solution_frame
            
        # add fixed params to frame
        save_frame.update(self.getFixedParameters())
        
        # save as compressed file
        np.savez_compressed(filepath, **save_frame)
        print('saved trajectories to ' + filepath)
        if os.path.exists('temp_folder'):
            shutil.rmtree('temp_folder')
            print('removed temp_folder')
        return None