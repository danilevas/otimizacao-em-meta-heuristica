#pragma once
#include "particle.h"
#include <vector>

class particle;

// type that defines pointer to comparison funciton depending whether the problem
// is minimization or maximization
typedef bool (* CompareFunction)(particle&, particle&); 


using namespace std;

class fitnessfunction
{
private:
	bool Minimize; // sense of optimization - maximize or minimize
	unsigned int dim;

public:

	fitnessfunction(void);
	//fitnessfunction(unsigned int newDim);
	// ~fitnessfunction(void);

	CompareFunction particleCompare; // pointer to predicate function for comparison of two particles, used for sort



	//the most important member function is the following
	virtual double calcFitness(vector<double> X) = 0; // takes the vector and returns the value of fitness function double 
	// needs to be redefined in inherited classes




	void SetMinimize(bool MinimizeFlag); // sets the flag regarding the optimization sense- minimize or maximize
	bool fitnessCompare(double fit1, double fit2); // comparison predicate function for doubles

};


bool particle_compare_minimize(particle &pP1, particle &pP2); // comparison predicate function - minimization
bool particle_compare_maximize(particle &pP1, particle &pP2); // comparison predicate function - maximization