// cppepso.cpp : main EPSO C++ application
// Hrvoje Keko, INESC Porto, 09/2006 and 10/2008, and 03/2009


// STL headers
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

// EPSO header
#include "swarm.h"

#include "exampleFitness.h" // Examples of fitness functions



int main(int argc, char* argv[]) // todo : iterations without improvement
{
	unsigned int i; // number of iterations
	unsigned int numParticles;
	unsigned int maxIter;

	unsigned int dim;
	bool minimize; // minimizing or maximizing

	// setup the general problem parameters
	numParticles=20;
	maxIter=500;
	dim = 2;
	minimize=true;
	Rosenbrock myFf; 
	
	// setup search space limits
	vector <double> minPos, maxPos;

	minPos.resize(dim); maxPos.resize(dim);
	for (i=0; i<dim; i++)
	{
		minPos[i]=-50;
		maxPos[i]=50;
	}
	
	// now declare the swarm
	swarm epsoSwarm (numParticles, minPos, maxPos, dim, minimize, myFf);

	cout << setprecision(11);
	//cout << scientific;
	i=0; 

    // ***  main algorithm loop  ***
	// stopping criterion should be decided here
	while (epsoSwarm.GetBestFitness() > 0.0001)		
	//for (i=0; i<maxIter;) 
	{
		i++;
		cout << " i:" << i << " f:" << epsoSwarm.GetBestFitness() << endl; 
		epsoSwarm.NextIteration();
	}

	// print the number of iterations
	cout << "iterations: " << i << " fitness evals: " << epsoSwarm.GetNumEvals() << " fitbest: " << epsoSwarm.GetBestFitness() << endl;
	
	// display the final best solution on the screen
	cout << endl << endl << " ==== best solution found ==== " << endl;
	for (i=0; i<dim; i++)
	{
		cout << "x[" << i << "]=" << epsoSwarm.GetBestPos()[i] << "  ";
		if (((i+1)%3)==0) cout << endl;
	}
	cout<<endl;
	return 0;
	
}




// best fitness output to .csv file
// ofstream outputFile;
// outputFile.open("output.txt", ios_base::out | ios_base::app );
// outputFile.precision(11);
// outputFile << scientific;
// outputFile << i << ", " << numEvals << ", " << gBestFitness << endl;






		



