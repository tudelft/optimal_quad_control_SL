#include "nn_parameters.h"

// NN network parameters -- atribute values
const float weights_in[NUM_NODES][NUM_STATES] = ${weights_in};

const float bias_in[NUM_NODES] = ${bias_in};

const float weights_out[NUM_CONTROLS][NUM_NODES] = ${weights_out};

const float bias_out[NUM_CONTROLS] = ${bias_out};

const float in_norm_mean[NUM_STATES] = ${in_norm_mean};

const float in_norm_std[NUM_STATES] = ${in_norm_std};

const float out_scale_min = ${out_scale_min};

const float out_scale_max = ${out_scale_max};

const float weights_hid[NUM_HIDDEN_LAYERS-1][NUM_NODES][NUM_NODES] = ${weights_hidden};

const float bias_hid[NUM_HIDDEN_LAYERS-1][NUM_NODES] = ${bias_hidden};