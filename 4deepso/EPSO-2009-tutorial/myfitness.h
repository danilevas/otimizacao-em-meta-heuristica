#pragma once
#include "fitnessfunction.h"

using namespace std;

class myFitnessFunction : public fitnessfunction
{
public:

	myFitnessFunction();

	//the most important member function is the following
	double calcFitness(vector<double> X); // takes the vector and returns the value of fitness function double 
};


