#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <libscuff.h>
#include "SSSolver.h"
using namespace scuff;

int main(int argc, char *argv[])
{
  // read geometry
  RWGGeometry *G=new RWGGeometry("mysgf.scuffgeo");
  printf("G->Surfaces[i]->Label:\n");
  for(int nr=0; nr<G->NumSurfaces; nr++) { printf("\t%s\n", G->Surfaces[nr]->Label); }

  // read capacitances from command line
  if (argc<4) {
    printf("\nCapacitances Caa, Cab, Cbb must be passed as command line parameters !\n\n");
    return -1;
  }
  double Caa = atof(argv[1]);
  double Cab = atof(argv[2]);
  double Cbb = atof(argv[3]);
  std::cout << "Caa=" << Caa << "\tCab=" << Cab << "\tCbb=" << Cbb << "\n";

  // fix charges on the spheres
  double Qa = 1.0;
  double Qb = 0.5;

  // set up solver and allocate memory
  SSSolver *SSS   = new SSSolver("mysgf.scuffgeo");
  HMatrix *M      = SSS->AllocateBEMMatrix();
  HVector *Sigma  = SSS->AllocateRHSVector();

  // assemble and factorize the BEM matrix
  SSS->AssembleBEMMatrix(M);
  M->LUFactorize();

  // calculate potentials 
  double *Potentials = new double[G->NumSurfaces];
  Potentials[0] = ( Qa*Cbb-Qb*Cab )/(Caa*Cbb-Cab*Cab);
  Potentials[1] = ( Qb*Caa-Qa*Cab )/(Caa*Cbb-Cab*Cab);

  // solve the system
  SSS->AssembleRHSVector(Potentials, 0, 0, Sigma);
  M->LUSolve(Sigma);  

  // write charge density to file
  SSS->PlotChargeDensity(Sigma, "surface_charge_output.pp", 0);

  return 0;
}
