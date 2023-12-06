import pickle
import numpy as np
from pyaudi import gdual_double, tanh, exp, log

class Controller:
    def __init__(self, path_to_pickle):
        self.input_scaler_params, self.output_scaler_params, self.config, self.weights \
            = pickle.load(open(path_to_pickle, 'rb'))
        if 'layers' in self.config:
            self.config = self.config['layers']
            
    def preprocess_input(self, state):
        return (state - self.input_scaler_params[0]) / self.input_scaler_params[1]

    def postprocess_output(self, pred):
        out = (pred - self.output_scaler_params[0]) / self.output_scaler_params[1]
        return (out * self.output_scaler_params[2]) + self.output_scaler_params[3]

    def nn_predict(self, model_input):
        vector = model_input
        dense_layer_count = 0
        for layer_config in self.config:
            if layer_config['class_name'] == 'Dense':
                wgts, biases = self.weights[dense_layer_count*2 : (dense_layer_count+1)*2]
                vector = wgts.T.dot(vector) + biases
                dense_layer_count += 1
            elif layer_config['class_name'] == 'Activation':
                if layer_config['config']['activation'] == 'relu':
                    vector[convert_gdual_to_float(vector) < 0] = 0
                elif layer_config['config']['activation'] == 'tanh':
                    vector = np.vectorize(lambda x : tanh(x))(vector)
                elif layer_config['config']['activation'] == 'softplus':
                    # avoiding overflow with exp; | log(1+exp(x)) - x | < 1e-10   for x>=30
                    # floatize = lambda x : x.constant_cf if type(x) == gdual_double else x
                    # softplus = lambda x : x if floatize(x) > 30.0 else log(exp(x)+1)
                    softplus = lambda x : log(exp(x)+1)
                    vector = np.vectorize(softplus)(vector)
        return vector


    def compute_control(self, state):
        model_input = self.preprocess_input(state)
        model_pred = self.nn_predict(model_input)
        control_out = self.postprocess_output(model_pred)
        return control_out
    
    def convert_gdual_to_float(gdual_array):
        floatize = lambda x : x.constant_cf if type(x) == gdual_double else x
        convert_to_float = np.vectorize(floatize, otypes=[np.float64])
        return convert_to_float(gdual_array)

class UnbiasedController(Controller):
    def __init__(self, path_to_pickle, bias):
        super().__init__(path_to_pickle=path_to_pickle)
        self.bias = np.array(bias)
    def compute_control(self, state):
        return super().compute_control(state + self.bias)
    
def get_unbiased_controller(path_to_pickle, g0, m, Fmin, Fmax, Ixx, L_q, theta_max):
    import pygmo as pg
    
    # This pygmo problem is solved by the equilibrium point of the system
    class my_prob:
        def __init__(self, path = path_to_pickle):
            self.Controller = Controller(path_to_pickle=path_to_pickle)
        def fitness(self, x):
            hover_thrust = (m*g0 - 2*Fmin) / (2*(Fmax-Fmin))
            return [np.linalg.norm(self.Controller.compute_control(x) - np.array([hover_thrust]*2))]
        def get_bounds(self):
            return ([-1,0,-1,0,0,0],[1,0,1,0,0,0])
        
    udp = my_prob()
    prob = pg.problem(udp)
    algo = pg.algorithm(pg.de(gen=500, xtol=0, ftol=0))
    #uda = pg.nlopt(solver="neldermead")
    #algo = pg.algorithm(pg.nlopt(solver="neldermead"))
    algo.set_verbosity(1)
    pop = pg.population(prob,19)
    pop.push_back([0,0,0,0,0,0])
    pop = algo.evolve(pop)
    
    bias = pop.champion_x
    
    return UnbiasedController(path_to_pickle, bias)