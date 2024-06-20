// swarm.cpp - implementation of swarm class


#include "swarm.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <boost/random.hpp>
#include <ctime>
#include <list>
#include <vector>
#include "particle.h"
#include "fitnessfunction.h"

#define EPSO_RANDOM_SEED 12345 // seed for repeatable experiments
// comment (undefine) this to have swarm initialized based on system clock

swarm::~swarm(void)
{
}



swarm::swarm(int newNumParticles, double minPos, double maxPos, unsigned int newDim, bool newMinimize, fitnessfunction &newFf, double newCommProb, double newTau) : ff(newFf)
{
	unsigned int i;

	this->ff = newFf; // !


	this->numParticles = newNumParticles;
	this->minimize = newMinimize;
	this->tau = newTau; // default value
	this->numEvals = 0;
	this->dim = newDim;
	
#ifndef EPSO_RANDOM_SEED
	rng.seed(static_cast<unsigned> (std::time(0))); //initialize random number generator with current time
#else
	rng.seed(EPSO_RANDOM_SEED); // fixed seed for RNG - use to have a repeatable experiment
#endif
	
	

	ff.SetMinimize(minimize);  

	// fill in the vectors for the constructor of particles.
	// for some functions, all hyperspace dimensions have the same boundaries

	// particles ask for vector of hyperspace boundaries
	// creation of these temporary vectors
	vector <double> tmpMinPos, tmpMaxPos, tmpMinVel, tmpMaxVel;
	for (i=0; i<dim; i++)
	{
		tmpMinPos.push_back(minPos);
		tmpMaxPos.push_back(maxPos);
		tmpMinVel.push_back(-fabs(maxPos - minPos));
		tmpMaxVel.push_back(fabs(maxPos - minPos));
	}

	for (i=0; i<numParticles; i++) // initialize list of particles
	{
		listParticles.push_back(particle(dim, tmpMinPos, tmpMaxPos, tmpMinVel, tmpMaxVel, rng, &ff, newCommProb));
	}

	list<particle>::iterator pParticle;

	listParticles.sort(ff.particleCompare); // may be stupid: sort list of particles according to their fitness

	this->gBestFitness = listParticles.begin()->GetFitness();
	this->gBestPos = listParticles.begin()->GetPos();  // get the global best for the beginning
}




swarm::swarm(int newNumParticles, vector<double> minPos, vector<double> maxPos, unsigned int newDim, bool newMinimize, fitnessfunction &newFf, double newCommProb, double newTau) : ff(newFf)
{
	unsigned int i;

	this->numParticles = newNumParticles;
	this->minimize = newMinimize;
	this->tau = newTau;
	this->numEvals = 0;
	this->dim = newDim;


#ifndef EPSO_RANDOM_SEED
	rng.seed(static_cast<unsigned> (std::time(0))); //initialize random number generator with current time
#else
	rng.seed(EPSO_RANDOM_SEED); // fixed seed for RNG - use to have a repeatable experiment
#endif

	ff.SetMinimize(minimize);  

	vector <double> tmpMinVel, tmpMaxVel; // velocity limits
	for (i=0; i<dim; i++)
	{
		tmpMinVel.push_back(-fabs(maxPos[i] - minPos[i]));
		tmpMaxVel.push_back(fabs(maxPos[i] - minPos[i]));
	}

	for (i=0; i<numParticles; i++) // initialize list of particles
	{
		listParticles.push_back(particle(dim, minPos, maxPos, tmpMinVel, tmpMaxVel, rng, &ff, newCommProb));
	}




	list<particle>::iterator pParticle;

	listParticles.sort(ff.particleCompare); // may be stupid: sort list of particles according to their fitness

	this->gBestFitness = listParticles.begin()->GetFitness();
	this->gBestPos = listParticles.begin()->GetPos();  // get the global best for the beginning
}







unsigned int swarm::GetNumEvals()
{
	return this->numEvals;
}

unsigned int swarm::GetNumParticles()
{
	return this->numParticles;
}

double swarm::GetBestFitness()
{
	return gBestFitness;
}

vector<double> swarm::GetBestPos()
{
	return gBestPos;
}


void swarm::SetTau(double newTau)
{
	tau = newTau;
}




void swarm::NextIteration()
{

	list<particle>::iterator pParticle;
	particle tempParticle;  // auxilliary variable: holding replicated particle
	// easy to make this a vector  - so particle might be repeatedly replicated



	// check whether the new best is found (remember list of particles is sorted!)
	pParticle=listParticles.begin();	// this is the best particle

	if (ff.fitnessCompare(pParticle->GetFitness(), gBestFitness))
	{
		gBestPos = pParticle->GetPos(); // new gBest found - update.
		gBestFitness = pParticle->GetFitness(); 
	}


	while(pParticle!=listParticles.end())
	{
		pParticle->SetBestPos(gBestPos); // new gBest set
		tempParticle=*pParticle; // replicate particle, make a copy

		tempParticle.Mutate(rng, tau); //  tau = 0.2, mutate internal parameters
		tempParticle.Movement(rng); // move the replicated particle
		(*pParticle).Movement(rng); // move the original particle

		numEvals++;
		numEvals++;

		if ( ff.particleCompare(*pParticle, tempParticle) )
		{ // original particle chosen - nothing to do
		}
		else
		{ // mutated particle chosen
			(*pParticle)=tempParticle;
		}	
		pParticle++; // iterate through all the particles
	}
	listParticles.sort(ff.particleCompare); // order the list of particles

	// iteration finished
}










//
//void swarm::dumpFitness()
//{
//	// print out the fitnesses of the whole population
//	//pParticle=listParticles.begin();  
//	//while(pParticle!=listParticles.end())
//	//{
//	//	cout << pParticle->GetFitness() << "; ";
//	//	pParticle++;
//	//}
//	//cout << endl;
//	// this should (actually) call particle's printout method.
//}