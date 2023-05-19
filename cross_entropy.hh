#pragma once
#include "nn_layer.hh"

class CECost {
public:
	float cost(host_vec predictions, int target);
	host_vec dCost(host_vec preds, int target, host_vec dY);
};