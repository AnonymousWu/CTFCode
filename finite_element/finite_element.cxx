/** \addtogroup examples 
  * @{ 
  * \defgroup finite
  * @{ 
  * \brief Finite element test/benchmark
  */
#include <ctf.hpp>
using namespace CTF;

/**
 * \brief computes the following kernel of the finite element method
 *        Where u, M^-1, K, X, p are given by:
 *        u = n^3 tensor, representing all the points in the mesh
 *        M^-1 = Diagonal n x n matrix (see book), Tensor producted with I 
 *        (->) Looks like this: M^-1_3D = I ⊗ I ⊗ M^-1_1D + I ⊗ M^-1_1D ⊗ I + M^-1_1D ⊗ I ⊗ I 
 *        K = n x n stiffness matrix, Tensor producted like (->)
 *        X = n x n transformation matrix (Ω -> [-1,1]), Tensor producted like (->)
 *        p = constant (In this, p = 1)
 *        So, this is A(u,v) = p*M^-1*K*X*u. Also, this is the case for d=3
 */
 
int finite(int      n,
           World & dw){
  int lens_u[] = {n,n,n};
  
  Matrix<> Minv(n,n); //diagonal!
  Matrix<> K(n,n);
  Matrix<> X(n,n); 
  Minv.fill_random(0.0,1.0);
  K.fill_random(0.0,1.0);
  X.fill_random(0.0,1.0);
  
  Tensor<> u(3, lens_u);  
  u.fill_random(0.0,1.0);
  
  Tensor<> a(3,lens_u);
  Tensor<> b(3,lens_u);
  
  double st_time = MPI_Wtime();
  
  a["ijk"]  = X["kl"]*u["ijl"];
  a["ijk"] += X["jl"]*u["ilk"];
  a["ijk"] += X["il"]*u["ljk"];
  
  b["ijk"]  = K["kl"]*a["ijl"];
  b["ijk"] += K["jl"]*a["ilk"];
  b["ijk"] += K["il"]*a["ljk"];
  
  u["ijk"]  = Minv["kl"]*b["ijl"];
  u["ijk"] += Minv["jl"]*b["ilk"];
  u["ijk"] += Minv["il"]*b["ljk"];
  
  //Multiplication for p would go here
  double exe_time = MPI_Wtime() - st_time;
  
  bool pass = u.norm2() >= 1.E-6; //same criterion as spectral_element.cxx
  
  if (dw.rank == 0){
    if (pass)
      printf("{ Finite element method } passed \n");
    else
      printf("{ Finite element method } failed \n");
    #ifndef TEST_SUITE
    printf("Finite element method on %d*%d*%d grid with %d processors took %lf seconds\n",n,n,n,dw.np,exe_time);
    #endif
  }
  return pass;
}

#ifndef TEST_SUITE
char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}

int main(int argc, char ** argv){
  int rank, np, n, pass;
  int const in_num = argc;
  char ** input_str = argv;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mp);
  
  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 16;
  } else n = 16;
  
  {
    World dw(argc, argv);
    
    if (rank == 0){
      printf("Running 3D finite element method with %d*%d*%d grid\n",n,n,n);
    }
    pass = spectral(n,dw);
    assert(pass);
  }
  
  MPI_Finalize();
  return 0;
}
/**
 * @}
 * @}
 */

#endif
