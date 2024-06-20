#pragma once
#include "fitnessfunction.h"


// A collection of example fitness functions
// This can almost be templated.

using namespace std;


// Rosenbrock
class Rosenbrock : public fitnessfunction
{
public:
	Rosenbrock():fitnessfunction(){ }; // constructor
	double calcFitness(vector<double> X)
	{
		unsigned int dim = X.size(); // dimension of search space
		double fit;
		unsigned int i;
		// Rosenbrock
		fit=0.0;
		for (i=0; i<dim-1; i++)
		{
			fit=fit + (100*pow((X[i+1]-X[i]*X[i]),2)+pow((X[i]-1),2));
		}
		return fit;
	}
};


// Sphere
class Sphere : public fitnessfunction
{
public:
	Sphere():fitnessfunction(){ }; // constructor
	double calcFitness(vector<double> X) 
	{
		unsigned int dim = X.size(); // dimension of search space
		double fit;
		unsigned int i;
		fit=0.;
		for (i=0; i<dim; i++)
		{
			fit=fit+ (X[i] - 1.0)*(X[i] - 1.0); // sphere, sum (xi - 1)^2
		}
		return fit;
	}
};



class Alpine : public fitnessfunction
{
public:
	Alpine():fitnessfunction(){ }; // constructor
	double calcFitness(vector<double> X)
	{
		unsigned int dim = X.size(); // dimension of search space
		double fit;
		unsigned int i;
		
		double mult=1.0;
		fit=1.0;
		for (i=0; i<dim; i++)
		{
			fit=fit * sin(X[i]);
			mult = mult*X[i];
		}
		fit=fit*sqrt(mult);

		return fit;
	}
			
};



class Schaffer : public fitnessfunction
{
public:
	Schaffer():fitnessfunction(){ }; // constructor
	double calcFitness(vector<double> X)
	{
		unsigned int dim = X.size(); // dimension of search space
		double fit;
		double x1 = X[0];
		double x2 = X[1];

		fit=0.5 + ((std::sin(std::sqrt(x1*x1+x2*x2)))*(std::sin(std::sqrt(x1*x1+x2*x2))) - 0.5) / ((1.0 + 0.001*(x1*x1+x2*x2))*(1.0 + 0.001*(x1*x1+x2*x2)));

		return fit;
	}

};



class Griewank : public fitnessfunction
{
public:
	Griewank():fitnessfunction(){ }; // constructor
	double calcFitness(vector<double> X)
	{
		unsigned int dim = X.size(); // dimension of search space
		double fit;
		unsigned int i;
		
		fit = 0.;
		double p = 1;
		for (i=0; i<dim; i++)
		{  
			fit = fit + X[i]*X[i];
			p = p * (cos(X[i]/sqrt((double) i+1)));
		}
		fit = fit / 4000 - p + 1;
		return fit;
	}
};


