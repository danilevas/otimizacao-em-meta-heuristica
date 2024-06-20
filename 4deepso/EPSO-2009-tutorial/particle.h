#pragma once
#include <vector>
#include <iostream>
#include <boost/random.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/lognormal_distribution.hpp>
#include <ctime>
#include "fitnessfunction.h"

using namespace std;


class fitnessfunction;

class particle
{
public:
	particle(void); // constructor, default

	//particle(int dim, double minPos, double maxPos, double minVel, double maxVel, boost::mt19937& rng);


	// constructor for new particles - needs reference to RNG so that it's always the same mersenne twister RNG
	particle(int dim, vector<double> minPos, vector<double> maxPos, vector<double> minVel, vector<double> maxVel, boost::mt19937& rng, fitnessfunction *pFf, double commProb=0.2);
	// the limits for each coordinate of the solution vector can be different - hence the usage of vectors

	~particle(void); // destructor
	
	void Print(); // dump the information about the particle, can be done by overloading the <<operator, for now this will do
	double GetFitness(); // gets the current fitness value but does not recalculate fitness each time
	void UpdateFitness(); // update the fitness

	vector <double > GetBestPos(); //get the particle's own previous best position (pBest)
	void SetBestPos(vector<double > newGlobalBestPos); //set global best position gBest for this particle
	vector <double > GetPos(); //get particle's current position (location, vector X)
	vector <double > GetVel(); //get particle's current velocity

	void Mutate(boost::mt19937& rng, double tau=0.2); // mutate the internal parameters (EPSO)
	//rng is parameter so that lognormal mutation can be called
	
	void Movement(boost::mt19937& rng); // perform the movement of particles according to EPSO rules

	void SetCommunicationProbability(double newCommProb);

private:
	unsigned int dim; // dimension of the search space
	double Fitness; // stores current fitness
	vector<double > Pos; // position
	vector<double > Vel; // velocity
	vector<double > myBestPos; // particle previous best, pBest
	vector<double > globalBestPos; //global best, gBest
	double myBestFitness; //particle's best fitness in myBestPos (pBest)

	double inertia;
	double memory;
	double cooperation;
	double perturbation; 
	// EPSO weights

	fitnessfunction *ff; // pointer to fitness function

	vector <double> maxPos;
	vector <double> minPos;
	vector <double> maxVel;
	vector <double> minVel; // limits for position and velocity


	double commProbability;
};

