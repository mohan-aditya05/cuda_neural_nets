#include "cross_entropy.hh"

#include <math.h>
#include <iostream>


float CECost::cost(host_vec preds, int target)
{
    float cost_value=0.0;
	cost_value += -log(preds[target]);

	// std::cout << "Cost value: "<< cost_value <<std::endl;
    return cost_value;
}


host_vec CECost::dCost(host_vec preds, int target, host_vec dY) {
	
	dY[target] = -1.0/preds[target];
	return dY;
}