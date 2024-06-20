#include ".\myfitness.h"


myFitnessFunction::myFitnessFunction():fitnessfunction()
{
	// constructor
}

double myFitnessFunction::calcFitness(vector<double> X)
{
	unsigned int dim = X.size(); // dimension of search space
	double fit=0.0;
	unsigned int i;
	for (i=0; i<dim; i++)
	{
		fit=fit+X[i];
	}
	return fit;
}

