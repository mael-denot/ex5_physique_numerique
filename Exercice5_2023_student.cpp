#include <iostream>       // basic input output streams
#include <fstream>        // input output file stream class
#include <cmath>          // librerie mathematique de base
#include <iomanip>        // input output manipulators
#include <valarray>       // valarray functions
#include "ConfigFile.h" // Il contient les methodes pour lire inputs et ecrire outputs 
#include <boost/random.hpp>
using namespace std; // ouvrir un namespace avec la librerie c++ de base



class Exercice7
{

private:
  // definition des constantes
  const double pi=3.1415926535897932384626433832795028841971e0;
  // definition des variables
  double tfin;          // Temps final
  unsigned int nsteps;  // Nombre de pas de temps
  double D;        // Coefficient de diffusion
  double v0;       // Moyenne de la Gaussienne initiale
  double gamma;    // Coefficient de friction
  double vc;       // Vitesse critique pour la friction
  int    N_part;   // Nombre de particules numériques
  int    N_bins;   // Nombre de bins de l'histogramme
  double vlb;      // v_min des bins de l'histogramme
  double vd_D;     // pour la distribution uniforme entre [vg_D et vd_D]
                   // ou double Dirac f = 1/2 (\delta(v-vg_D) + \delta(v-vd_D))
  double vg_D;     // pour la distribution uniforme entre [vg_D et vd_D] 
                   // ou double Dirac f = 1/2 (\delta(v-vg_D) + \delta(v-vd_D))
  string initial_distrib; // type de distribution initiale ('D' pour Dirac)
  double vhb;      // v_max pour les bins de l'histogramme
  double sigma0;   // écart-type de la Gaussienne initiale

  unsigned int sampling;  // Nombre de pas de temps entre chaque ecriture des diagnostics
  unsigned int last;       // Nombre de pas de temps depuis la derniere ecriture des diagnostics
  ofstream *outputFile;    // Pointeur vers le fichier de sortie

  void printOut(bool write)
  {
    valarray<double> moments;  
    double var;
    // Ecriture tous les [sampling] pas de temps, sauf si write est vrai
    if((!write && last>=sampling) || (write && last!=1))
    {
      bins    = binning_fun(v);
      moments = fun_moments(v) ;
      var     = moments[1] - moments[0]*moments[0];
      *outputFile << t << " " << N_part<< " "<<moments[0]<<" "<<var;
      for(int ib = 0; ib < N_bins; ++ib){
        *outputFile<<" "<< bins[ib];
      }
      *outputFile <<endl; // write output on file
      last = 1;
    }
    else
    {
      last++;
    }
  }

protected:
  double t;
  double dt;
  valarray<double> v ;
  valarray<double> bins ;

  double acceleration(double v_p){
    return -gamma*(v_p - vc);
  }

  valarray<double> binning_fun(const valarray<double> &v){
     valarray<double> bins_fun = valarray<double>(N_bins); 
     // binning of the particle to compute the distribution function
     // histogramme de la fonction de distribution
     // bins_fun[i] devrait être égal au nombre de particules dans le bin numéro i
    double box_size = (vhb - vlb)/N_bins; 
    int a(0);

     for (auto speed : v) {
      a = v/box_size;
      if (a>vhb) {++bins_fun[N_bins - 1];}
      else if (a<vlb)
      {
        ++bins_fun[0];
      }
      else {
        ++bins_fun[a];
      }
     }

     return bins_fun;
  }


  valarray<double> fun_moments(const valarray<double> &v){
     valarray<double> moments = valarray<double>(2); 
     // compute first and second order moment
     // moyenne de v et moyenne de v^2
    moments[0] = v.sum()/v.size();
    valarray<double> v_squared=v*v;
    moments[1] = v_squared.sum()/v_squared.size();
     return moments;
  }

  valarray<double> initialization(){
     valarray<double> vel = valarray<double>(N_part);
     boost::mt19937 rng;
     if (initial_distrib == "D"){
	cout << "Delta distribution"<<endl;
	// TODO: initialize the initial particle velocities according to a double Dirac distribution
	// f = 1/2 (\delta(v-vg_D) + \delta(v-vd_D))
     }
     else{
	cout << "Gaussian distribution"<<endl;
        boost::normal_distribution<> initial_distribution(v0,sigma0);
        boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > initial_velocities_generator(rng,initial_distribution);
	// Uncomment the following two lines (and comment the two lines above) to use a uniform sampling between vg_D and vd_D
        //boost::uniform_int<> initial_distribution(vg_D,vd_D);
	//boost::variate_generator<boost::mt19937&, boost::uniform_int<> > initial_velocities_generator(rng,initial_distribution);
        for(int ip = 0; ip < N_part; ++ip){
          vel[ip] = initial_velocities_generator();
	}	
    }
     return vel;
   }


public:

  Exercice7(int argc, char* argv[])
  {
    string inputPath("configuration.in"); // Fichier d'input par defaut
    if(argc>1) // Fichier d'input specifie par l'utilisateur ("./Exercice7 config_perso.in")
      inputPath = argv[1];

    ConfigFile configFile(inputPath); // Les parametres sont lus et stockes dans une "map" de strings.
    for(int i(2); i<argc; ++i) // Input complementaires ("./Exercice3 config_perso.in input_scan=[valeur]")
      configFile.process(argv[i]);

    // see input file for description
    tfin     = configFile.get<double>("tfin",tfin);
    nsteps   = configFile.get<unsigned int>("nsteps",nsteps); 
    D        = configFile.get<double>("D",D);
    gamma    = configFile.get<double>("gamma",gamma);
    v0       = configFile.get<double>("v0",v0);
    sigma0   = configFile.get<double>("sigma0",sigma0);
    N_part   = configFile.get<double>("N_part",N_part);
    N_bins   = configFile.get<double>("N_bins",N_bins);
    vhb      = configFile.get<double>("vhb",vhb);
    vlb      = configFile.get<double>("vlb",vlb);
    vd_D     = configFile.get<double>("vd_D",vd_D);
    vg_D     = configFile.get<double>("vg_D",vg_D); 
    vc       = configFile.get<double>("vc",vc);
    sampling = configFile.get<unsigned int>("sampling",sampling); 
    initial_distrib = configFile.get<string>("initial_distrib");

    dt = tfin / nsteps; 

    // Ouverture du fichier de sortie
    outputFile = new ofstream(configFile.get<string>("output","output.out").c_str()); 
    outputFile->precision(15); // Les nombres seront ecrits avec 15 decimales

  };

  ~Exercice7()
  {
    outputFile->close();
    delete outputFile;
  };

  void run()
  {
    bins =valarray<double>(N_bins);
    v =valarray<double>(N_part); 
    t = 0.; // initialiser le temps
    last = 0; // initialise le parametre d'ecriture
    boost::mt19937 rng; // initialise le générateur de nombres pseudo-aléatoire
    printOut(true); // ecrire premier pas de temps
    boost::normal_distribution<> displace_gauss(0.,1.); // définit une distribution normale
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > random_deplacement(rng,displace_gauss);

    //initialize particles velocity according to a given distribution function
    v = initialization();
    
    //TODO: time step loop



      while (t < tfin){
          // particle loop: evolve particles velocity
          v += dt * acceleration(v) + random_deplacement()*sqrt(2*D*dt);
          // use printOut to write the output
          printOut(false);
      }


    //suggestion: write here the algorithm, DO NOT create a function step() or there can be problems with 
    //the boost library for random number generation
    
    printOut(true); // ecrire dernier pas de temps
  };
};


int main(int argc, char* argv[])
{
  Exercice7 engine(argc,argv); // definit la classe pour la simulation
  engine.run(); // executer la simulation
  cout << "Fin de la simulation." << endl;
  return 0;
}
