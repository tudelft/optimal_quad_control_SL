#ifndef NN_PARAMETERS_H
#define NN_PARAMETERS_H

#define NUM_STATES ${num_states}
#define NUM_CONTROLS ${num_controls}
#define NUM_LAYERS ${num_layers}
#define NUM_HIDDEN_LAYERS ${num_hidden_layers}
#define NUM_NODES ${num_nodes}


// NN network parameters -- define variables
extern const float weights_in[NUM_NODES][NUM_STATES];

extern const float bias_in[NUM_NODES];

extern const float weights_out[NUM_CONTROLS][NUM_NODES];

extern const float bias_out[NUM_CONTROLS];

extern const float in_norm_mean[NUM_STATES];

extern const float in_norm_std[NUM_STATES];

extern const float out_scale_min;

extern const float out_scale_max;

extern const float weights_hid[NUM_HIDDEN_LAYERS-1][NUM_NODES][NUM_NODES];

extern const float bias_hid[NUM_HIDDEN_LAYERS-1][NUM_NODES];

#endif
