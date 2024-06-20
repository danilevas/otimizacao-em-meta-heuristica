#include ".\fitnessfunction.h"

fitnessfunction::fitnessfunction()
{
	this->Minimize = true;
	this->particleCompare = particle_compare_minimize; // default : to minimize
	// this->dim = newDim;
}
//
//fitnessfunction::~fitnessfunction(void)
//{
//}


void fitnessfunction::SetMinimize(bool MinimizeFlag)
{
	if (MinimizeFlag == false)
	{
		this->particleCompare = particle_compare_maximize;
		this->Minimize = false;
	}
	else 
	{
		this->particleCompare = particle_compare_minimize;
		this->Minimize = true;
	}
}



bool fitnessfunction::fitnessCompare(double fit1, double fit2)
{

	if (Minimize)
	{
		if (fit1 < fit2) // minimizing
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	else 
	{
		if (fit1 > fit2) // maximizing
		{
			return true;
		}
		else
		{
			return false;
		}
	}
}



bool particle_compare_minimize(particle &Particle1, particle &Particle2) // predicate function for comparison in sort algorithm
{
	if (Particle1.GetFitness() < Particle2.GetFitness()) 
	{
		return true;
	}
	else
	{
		return false;
	}
}


bool particle_compare_maximize(particle &Particle1, particle &Particle2) // predicate function for comparison in sort algorithm
{
	if (Particle1.GetFitness() > Particle2.GetFitness()) 
	{
		return true;
	}
	else
	{
		return false;
	}
}



