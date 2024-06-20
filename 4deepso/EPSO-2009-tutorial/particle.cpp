#include ".\particle.h"

particle::particle(void)
{
}

particle::~particle(void)
{
}


particle::particle(int dim, vector<double> minPos, vector<double> maxPos, vector<double> minVel, vector<double> maxVel, boost::mt19937& rng, fitnessfunction *pFf, double commProb)
{
	//define distribution for weights
	boost::uniform_real<double> uniform_weight(0,1);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<double> >  uniform_sampler_weights(rng, uniform_weight); 

	int i;
	for (i=0; i<dim; i++)
	{
		// define distribution for positions according to [minPos, maxPos] for i-th dimension
		boost::uniform_real<double> uniform_pos(minPos[i], maxPos[i]);  
		// bind variate generator to random number sampler
		boost::variate_generator<boost::mt19937&, boost::uniform_real<double> >  uniform_sampler_position(rng, uniform_pos); 

		//define distribution for velocities according to [minVel, maxVel] for i-th dimension
		boost::uniform_real<double> uniform_vel(minVel[i], maxVel[i]);  
		// bind variate generator to random number sampler
		boost::variate_generator<boost::mt19937&, boost::uniform_real<double> >  uniform_sampler_velocities(rng, uniform_vel); 

		Pos.push_back(uniform_sampler_position());
		Vel.push_back(uniform_sampler_velocities());
	}


	inertia=uniform_sampler_weights();
	memory=uniform_sampler_weights();
	cooperation=uniform_sampler_weights();
	perturbation=uniform_sampler_weights(); 

	this->dim=dim;

	this->ff = pFf;

	this->maxPos=maxPos;
	this->minPos=minPos;
	this->minVel=minVel;
	this->maxVel=maxVel;
	
	myBestPos=Pos;
	globalBestPos=Pos;

	this->commProbability = commProb; // default value
	
	this->UpdateFitness();
	myBestFitness=Fitness;
}




double particle::GetFitness()
{
	return Fitness;
}



void particle::UpdateFitness()
{
	this->Fitness=this->ff->calcFitness(Pos);	

}


void particle::Print()
{
	//unsigned int i;
	//for (i=0; i<Pos.size(); i++)	cout << "pos[" << i << "]=" << Pos[i] << endl;
	//this->UpdateFitness();
	cout << "=============================" << endl;	
	cout << " x1 : " << Pos[0] << " x2 : " << Pos[1] << " f: " << Fitness << endl;
	cout << " v1 : " << Vel[0] << " v2 : " << Vel[1] << endl;
	cout <<	" gB1: " << globalBestPos[0]<< " gB2: " << globalBestPos[1] << endl;
	cout << " in : " << inertia << " me : " << memory << " co : " << cooperation;
	//cout << " pe : " << perturbation << endl << endl;
	cout << endl;
}



vector <double > particle::GetBestPos()
{
	return myBestPos;
}

void particle::SetBestPos(vector<double> newGlobalBestPos)
{
	globalBestPos=newGlobalBestPos;
}


vector <double > particle::GetPos()
{
	return Pos;
}

vector <double > particle::GetVel()
{
	return Vel;
}


void particle::Mutate(boost::mt19937 &rng, double tau)
{
	//boost::lognormal_distribution<double> lognormal(1.0, 1.0);
	//boost::variate_generator<boost::mt19937&, boost::lognormal_distribution<double> >  rng_sampler(rng, lognormal); 
	//lognormal mutation with mean and sigma 1

	boost::normal_distribution<double> norm_dist(0.0, 1.0);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> >  rng_sampler(rng, norm_dist);
	// for gaussian mutation 

	double ranNum;

	ranNum=rng_sampler();
	//inertia=inertia * pow(exp(ranNum), tau); //lognormal type 
	inertia=inertia + ranNum*tau; // gaussian type
	if (inertia<0) inertia=0;
	if (inertia>1) inertia=1;

	ranNum=rng_sampler();
	//memory=memory * pow(exp(ranNum), tau);
	memory=memory + ranNum*tau;
	if (memory<0) memory=0;
	if (memory>1) memory=1;

	ranNum=rng_sampler();
	//cooperation=cooperation * pow(exp(ranNum), tau);
	cooperation=cooperation + ranNum * tau;
	if (cooperation<0) cooperation=0;
	if (cooperation>1) cooperation=1;


	ranNum=rng_sampler();
	//perturbation=perturbation * pow(exp(ranNum), tau);
	perturbation=perturbation + ranNum*tau;
	if (perturbation<0) perturbation=0;
	if (perturbation>1) perturbation=1;
	
}


void particle::Movement(boost::mt19937& rng)
{

	unsigned int i;
	double inertiaTerm, memoryTerm, cooperationTerm; // perturbationTerm;
	double tempVel;
	double tempPos;

	boost::uniform_real<double> uniform_dist(0,1);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<double> >  uniform_sampler(rng, uniform_dist);

	double ranNum;

	double disturbator;
	disturbator = uniform_sampler(); // for disturbing the global best of the particular particle
	
	
	// classic stochastic star - that only chooses whether to use globalBest ONCE for all dimensions is implemented like this:
	//double cooperationUsed;
	//cooperationUsed = uniform_sampler();


	for (i=0; i<dim; i++)
	{
		inertiaTerm = inertia * Vel[i]; // iner * v(t-1)
		memoryTerm = memory * ( myBestPos[i] - Pos[i] ); // mem * (myBest - pos)
		 
	// for "classic" stochastic star communication scheme: 
	// the random number is only sampled ONCE for the *whole* global best vector
	//if (cooperationUsed<0.2) 
    //	{
				//unsigned int j; 
				//double ranNum;
				//boost::uniform_real<double> uniform_dist(0,1);
				//boost::variate_generator<boost::mt19937&, boost::uniform_real<double> >  uniform_sampler(rng, uniform_dist);
				//ranNum=uniform_sampler();
				//for (j=0; j<dim; j++) // perturbation
				//{ 
				//	//ranNum=uniform_sampler(); // ..
				//	globalBestPos[j]=globalBestPos[j] * (1 + perturbation * ranNum);
				//}
		//		}


		if (uniform_sampler()<commProbability) // per-dimension stochastic star - default for EPSO

		{
			ranNum = disturbator;
			globalBestPos[i]=globalBestPos[i] * (1 + perturbation * ranNum);
			cooperationTerm = cooperation * (globalBestPos[i] - Pos[i]); // coop* (gBest - Pos)

		}
		else
		{
			cooperationTerm = 0;
		}
		tempVel = inertiaTerm + memoryTerm + cooperationTerm; // v(t)
		

		// ** PSO rules **
		//double fi1, fi2;
		//fi1 = uniform_sampler();
		//fi2 = uniform_sampler();
		//tempVel = Vel[i] + fi1 * (myBestPos[i] - Pos[i]) + fi2 * (globalBestPos[i] - Pos[i]);
		// **

		if (tempVel<minVel[i]) tempVel=minVel[i];
		if (tempVel>maxVel[i]) tempVel=maxVel[i];
		Vel[i]=tempVel; // update velocity

		tempPos=Pos[i]+Vel[i]; // x(t+1) = x(t) + v(t)

		if (tempPos<minPos[i]) 
		{
			tempPos=minPos[i];
			Vel[i]=-Vel[i]; // bounce
		}
		if (tempPos>maxPos[i])
		{
			tempPos=maxPos[i];
			Vel[i]=-Vel[i]; // bounce
		}
		Pos[i]=tempPos; // update position

		if (Vel[i]<minVel[i]) Vel[i]=minVel[i]; // in case of asymmetric velocity limits
		if (Vel[i]>maxVel[i]) Vel[i]=maxVel[i]; 
	}

	this->UpdateFitness();
	if (ff->fitnessCompare(this->Fitness, this->myBestFitness)) //if (this->Fitness > this->myBestFitness)
	{
		this->myBestFitness = this->Fitness;
		this->myBestPos = this->Pos;
	}


}



void particle::SetCommunicationProbability(double newCommProb)
{
	commProbability = newCommProb;
}





