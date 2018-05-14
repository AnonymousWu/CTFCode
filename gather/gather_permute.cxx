/** \addtogroup examples 
  * @{ 
  * \defgroup gather
  * @{ 
  * \brief Gather test/benchmark for regular grid (2D)
  */
#include <ctf.hpp>
using namespace CTF;

/**
 * \brief Builds the gather/scatter test for a regular grid.
 *        However, unlike before, we are treating this as an
 *        unstructured mesh. Also, we are doing 3 different
 *        tensor contractions.
 */
int64_t getPosInArrWW(int64_t i, int n){
  return (i%(2*n))/n;
}

//This finds the element that is in the direction indicated
int64_t findBot(int64_t n, int64_t elems, int64_t i){
  return i/(n*n) - elems;
}

int64_t findLeft(int64_t n, int64_t elems, int64_t i){
  return i/(n*n) - 1;
}

int64_t findRight(int64_t n, int64_t elems, int64_t i){
  return i/(n*n) + 1;
}

int64_t findTop(int64_t n, int64_t elems, int64_t i){
  return i/(n*n) + elems;
}

int64_t findBL(int64_t n, int64_t elems, int64_t i){
  return findLeft(n,elems,i) - elems;
}

int64_t findBR(int64_t n, int64_t elems, int64_t i){
  return findRight(n,elems,i) - elems;
}

int64_t findTL(int64_t n, int64_t elems, int64_t i){
  return findLeft(n,elems,i) + elems;
}

int64_t findTR(int64_t n, int64_t elems, int64_t i){
  return findRight(n,elems,i) + elems;
}

int64_t getPosShort(int64_t x, int64_t y, int64_t z, int64_t w,
                    int64_t lenX, int64_t lenY, int64_t lenZ){
  return x + y*lenX + z*lenX*lenY + w*lenX*lenY*lenZ;
}

int64_t getPos(int64_t x, int64_t y, int64_t z, int64_t a, int64_t b, int64_t c,
               int64_t lenX, int64_t lenY, int64_t lenZ, int64_t lenA, int64_t lenB){
  return x + y*lenX + z*lenX*lenY + a*lenX*lenY*lenZ
           + b*lenX*lenY*lenZ*lenA + c*lenX*lenY*lenZ*lenA*lenB;
}

void fill_Gwwew(Tensor<double> & Gwwew, int64_t elems, int64_t n, World & dw){
  int64_t elems2 = elems*elems;
  int64_t len = elems2*elems2*4;

  int64_t my_col = len/dw.np;
  int64_t my_col_st = dw.rank*my_col;
  if (dw.rank < len%dw.np) my_col++;
  my_col_st += std::min((int)dw.rank, (int)len%dw.np);
  
  int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*4*my_col);
  double * vals = (double*)malloc(sizeof(double)*4*my_col);
  
  int64_t act_tot_nnz = 0;
  for (int64_t i = my_col_st; i < my_col_st + my_col; i++){
    
    int idx1 = i % elems2;              //current pos
    int idx2 = i/elems2 % elems2;       //where we want
    int idx3 = i/(elems2*elems2) % 2;   
    int idx4 = i/(elems2*elems2*2) % 2;
    
    printf("%d %d %d %d\n",idx1,idx2,idx3,idx4);
    bool eastGood = idx2-1==idx1 and idx3==0 and idx4==0 and idx2%elems!=0;
    bool westGood = idx2==idx1-1 and idx3==1 and idx4==1 and idx2%elems!=elems-1;
  
    if (eastGood || westGood){
      inds[act_tot_nnz] = getPosShort(idx1,idx2,idx3,idx4,
                                 elems2,elems2,2);
      vals[act_tot_nnz] = 1;
      act_tot_nnz++;
    }
  }
  Gwwew.write(act_tot_nnz,inds,vals);
  free(inds);
  free(vals);
}

void fill_Gwwns(Tensor<double> & Gwwns, int64_t elems, int64_t n, World & dw){
  int64_t elems2 = elems*elems;
  int64_t len = elems2*elems2*4;

  int64_t my_col = len/dw.np;
  int64_t my_col_st = dw.rank*my_col;
  if (dw.rank < len%dw.np) my_col++;
  my_col_st += std::min((int)dw.rank, (int)len%dw.np);
  
  int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*2*my_col);
  double * vals = (double*)malloc(sizeof(double)*2*my_col);
  
  int64_t act_tot_nnz = 0;
  for (int64_t i = my_col_st; i < my_col_st + my_col; i++){
  
    int idx1 = i % elems2;
    int idx2 = i/elems2 % elems2;
    int idx3 = i/(elems2*elems2) % 2;
    int idx4 = i/(elems2*elems2*2) % 2;
    
    bool northGood = idx2-n==idx1 and idx3==0 and idx4==0;
    bool southGood = idx2==idx1-n and idx3==1 and idx4==1;
    if (northGood || southGood){
      inds[act_tot_nnz] = getPosShort(idx1,idx2,idx3,idx4,
                                 elems2,elems2,2);
      vals[act_tot_nnz] = 1;
      act_tot_nnz++;
    }
  }
  Gwwns.write(act_tot_nnz,inds,vals);
  free(inds);
  free(vals);
}

void fill_Gcc(Tensor<double> & Gcc, int64_t elems, int64_t n, World & dw){
  int64_t elems2 = elems * elems;
  int64_t len = elems2*4;

  int64_t my_col = len/dw.np;
  int64_t my_col_st = dw.rank*my_col;
  if (dw.rank < len%dw.np) my_col++;
  my_col_st += std::min((int)dw.rank, (int)len%dw.np);
  
  int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*4*my_col);
  double * vals = (double*)malloc(sizeof(double)*4*my_col);
  
  int64_t act_tot_nnz = 0;
  for (int64_t i = my_col_st; i < my_col_st + my_col; i++){
    int64_t rowsUp = i / (elems*4*n*n);
    int64_t colsOver = (i%(elems*n*n))/(n*n)*n + i%n; //elems_over * n + pos_in_elem
    
    bool topLeft = i%(n*n) == n*n-n and rowsUp != n-1 and colsOver != 0;
    bool topRight = i%(n*n) == n*n-1 and rowsUp != n-1 and colsOver != (elems*2-1);
    bool botLeft = i%(n*n) == 0 and rowsUp != 0 and colsOver != 0;
    bool botRight = i%(n*n) == n-1 and rowsUp != 0 and colsOver != 0;
    
    if (topLeft){
      inds[act_tot_nnz] = getPosShort(rowsUp, findTL(n,elems,i),0,1,elems,elems,2);
      vals[act_tot_nnz] = 1;
      act_tot_nnz++;
    }
    else if (topRight){
      inds[act_tot_nnz] = getPosShort(rowsUp, findTR(n,elems,i),1,1,elems,elems,2);
      vals[act_tot_nnz] = 1;
      act_tot_nnz++;
    }
    else if (botLeft){
      inds[act_tot_nnz] = getPosShort(rowsUp, findBL(n,elems,i),0,0,elems,elems,2);
      vals[act_tot_nnz] = 1;
      act_tot_nnz++;
    }
    else if (botRight){
      inds[act_tot_nnz] = getPosShort(rowsUp, findBR(n,elems,i),1,0,elems,elems,2);
      vals[act_tot_nnz] = 1;
      act_tot_nnz++;
    }
  }
  Gcc.write(act_tot_nnz,inds,vals);
  free(inds);
  free(vals);
}

void fill_Pc(Tensor<double> & Pc, int64_t n){
  int64_t pos[] = {n-1,2*n-n};
  double vals[] = {1,1};
  Pc.write(2,pos,vals);
}

void fill_Pw(Tensor<double> & Pw, int64_t n){
  int64_t pos[] = {1,(n-1)*2};
  double vals[] = {1,1};
  Pw.write(2, pos, vals);
}

void fill_Ps(Tensor<double> & Pc, Tensor<double> & Pw, int64_t n, World & dw){
  fill_Pc(Pc, n);
  fill_Pw(Pw, n);
}

void fill_Gww(Tensor<double> & Gwwns, Tensor<double> & Gwwew, int64_t elems, int64_t n, World & dw){
  fill_Gwwns(Gwwns, elems, n, dw);
  fill_Gwwew(Gwwew, elems, n, dw);
}

int64_t gather(int      n,
               int  elems,
               World & dw){
  int lens_u[] = {elems*elems,n,n};
  int lens_G[] = {elems*elems,elems*elems,2,2};
  int lens_P[] = {2,n};
  Tensor<double> Pc(2,lens_P);
  Tensor<double> Pw(2,lens_P);
  Tensor<double> u(3,lens_u);
  Tensor<double> Gcc(4,lens_G);
  Tensor<double> Gwwns(4,lens_G);
  Tensor<double> Gwwew(4,lens_G);
  Tensor<double> utemp(3,lens_u);
  u.fill_random(1.0,1.0);
  
  double fg_time = MPI_Wtime();
  
  fill_Gcc(Gcc,elems,n,dw);
  fill_Gww(Gwwns,Gwwew,elems,n,dw);
  fill_Ps(Pc,Pw,n,dw);
  
  double fgexe_time = MPI_Wtime() - fg_time;
  printf("make time is: %lf\n", fgexe_time);
  Gwwew.print();
  double st_time = MPI_Wtime();
  
  utemp["ijk"]  = Pw["cj"]*Gwwns["iwac"]*Pw["ab"]*u["wbk"];
  utemp["ijk"] += Pw["ck"]*Gwwew["iwac"]*Pw["ab"]*u["wjb"]; // Same thing, but transpose
  
  //u["ijk"] += Pc["ad"]*Pc["cd"]*Gcc["iwce"]*((Pc["eb"]*u["wjb"])*Pc["ak"]); // For the corners
  u["ijk"] += utemp["ijk"];
  double exe_time = MPI_Wtime() - st_time;
  printf("exec time is: %lf\n", exe_time);
  u.print();
  
  return 1;
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
  int rank, np, n, pass, elems;
  int const in_num = argc;
  char ** input_str = argv;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  
  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 3;
  } else n = 3;
  
  if (getCmdOption(input_str, input_str+in_num, "-elems")){
    elems = atoi(getCmdOption(input_str, input_str+in_num, "-elems"));
    if (elems < 0) elems = 3;
  } else elems = 3;
  
  {
    World dw(argc, argv);
    if (rank == 0){
      printf("Running gather with %d*%d*%d grid\n",elems*elems,n,n);
    }
    pass = gather(n,elems,dw);
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
