
// EPSO 2009
// Swarm class
// Class that encapsulates the EPSO algorithm itself
// Things that can (and should) be redesigned in real-use: 
// - constructor
// - iteration

// Other things are basically non-existent.


#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <list>
#include <vector>
#include <boost/random.hpp>


#include "particle.h"
#include "fitnessfunction.h"



using namespace boost;



class swarm
{
public:
	//swarm(void);
	
	swarm(int numParticles, double minPos, double maxPos, unsigned int dim, bool minmize, fitnessfunction &ff, double commProb=0.2, double tau=0.2); // constructor with the limits same for each dimension - common use
	swarm(int numParticles, vector<double> minPos, vector<double> maxPos, unsigned int dim, bool minimize, fitnessfunction &ff, double commProb=0.2, double tau=0.2); // constructor

	~swarm(void); // destructor
	void NextIteration();
	
	unsigned int GetNumEvals();
	unsigned int GetNumParticles();


	double GetBestFitness();
	vector<double> GetBestPos();

	void SetTau(double newTau);

	// note: if designed, swarm's copy constructor needs to copy the state of RNG generator!




private:

	double gBestFitness; // global best's fitness
	vector<double > gBestPos; // global best position

	unsigned int numParticles; // number of particles
	list<particle> listParticles;  // list holding the swarm

	unsigned int numEvals; // number of fitness function evaluations performed
	boost::mt19937 rng; // unique random number generator used for initializing all the particles
	fitnessfunction &ff;


	unsigned int dim;
	bool minimize; // minimizing or maximizing

	double tau; // mutate strategic parameters

};
