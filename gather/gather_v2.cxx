/** \addtogroup examples 
  * @{ 
  * \defgroup gather
  * @{ 
  * \brief Gather test/benchmark for regular grid (2D)
  */
#include <ctf.hpp>
using namespace CTF;

/**
 * \brief Builds the gather matrix, and mv mutiplies it with 
 *        m1*m2 finite elements, gathering them into a large,
 *        global vector.
 */

int64_t getPos(int64_t x, int64_t y, int64_t z, int64_t w, int64_t lenX, int64_t lenY, int64_t lenZ){
  return x + y*lenX + z*lenX*lenY + w*lenX*lenY*lenZ;
}

int64_t getN(int64_t i, int64_t m1, int64_t m2, int64_t n, int64_t colsOver, int64_t rowsUp){
  return getPos(colsOver,rowsUp+1,i%n,0,m1,m2,n);
}

int64_t getE(int64_t i, int64_t m1, int64_t m2, int64_t n, int64_t colsOver, int64_t rowsUp){
  return getPos(colsOver+1,rowsUp,0,(i%(n*n))/n,m1,m2,n);
}

int64_t getS(int64_t i, int64_t m1, int64_t m2, int64_t n, int64_t colsOver, int64_t rowsUp){
  return getPos(colsOver,rowsUp-1,i%n,n-1,m1,m2,n);
}

int64_t getW(int64_t i, int64_t m1, int64_t m2, int64_t n, int64_t colsOver, int64_t rowsUp){
  return getPos(colsOver-1,rowsUp,n-1,(i%(n*n))/n,m1,m2,n);
}

int64_t getNW(int64_t i, int64_t m1, int64_t m2, int64_t n, int64_t colsOver, int64_t rowsUp){
  return getPos(colsOver-1,rowsUp+1,n-1,0,m1,m2,n);
}

int64_t getNE(int64_t i, int64_t m1, int64_t m2, int64_t n, int64_t colsOver, int64_t rowsUp){
  return getPos(colsOver+1,rowsUp+1,0,0,m1,m2,n);
}

int64_t getSW(int64_t i, int64_t m1, int64_t m2, int64_t n, int64_t colsOver, int64_t rowsUp){
  return getPos(colsOver-1,rowsUp-1,n-1,n-1,m1,m2,n);
}

int64_t getSE(int64_t i, int64_t m1, int64_t m2, int64_t n, int64_t colsOver, int64_t rowsUp){
  return getPos(colsOver+1,rowsUp-1,0,n-1,m1,m2,n);
}

void fill_gather(Tensor<double> * G, int m1, int m2, int n, World & dw){
  int64_t w = n*n*m1*m2;
  int64_t h = w;

  int64_t my_col = h/dw.np;
  int64_t my_col_st = dw.rank*my_col;
  if (dw.rank < h%dw.np) my_col++;
  my_col_st += std::min((int)dw.rank, (int)h%dw.np);
  
  int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*4*my_col);
  double * vals = (double*)malloc(sizeof(double)*4*my_col);
  
  int64_t act_tot_nnz = 0;
  int64_t iters = my_col_st;
  for (int64_t i = my_col_st; iters < my_col_st+my_col; i++,act_tot_nnz++,iters++){
    int64_t colsOver = (double)iters/(double)(m1*n*n*m2)*m2;
    int64_t rowsUp = ((int64_t)((double)iters/(double)(m1*n*n)*m1)) % m1;
    int im = iters % (n*n);
    bool NW = im == n*n - n;
    bool NE = im == n*n - 1;
    bool SW = im == 0;
    bool SE = im == n-1;
    bool N = im >= n*n - n;
    bool E = (iters % n) == n-1;
    bool S = im <= n-1;
    bool W = (iters % n) == 0;
    bool onN = iters%(n*n)>=(n*n-n) and rowsUp == m2-1;
    bool onE = iters%n == n-1 and colsOver == m1-1;
    bool onS = iters%(n*n) < n and rowsUp == 0;
    bool onW = iters%n == 0 and colsOver == 0;
    inds[i-my_col_st] = iters;
    vals[i-my_col_st] = 1;
    if (NW) {
      if (not onN and not onW) {
        inds[i-my_col_st+1] = getNW(iters,m1,m2,n,colsOver,rowsUp);
        inds[i-my_col_st+2] = getW(iters,m1,m2,n,colsOver,rowsUp);
        inds[i-my_col_st+3] = getN(iters,m1,m2,n,colsOver,rowsUp);
        vals[i-my_col_st+1] = 1;
        vals[i-my_col_st+2] = 1;
        vals[i-my_col_st+3] = 1;
        i+=3;act_tot_nnz+=3;
      }
      else if (onN and not onW){
        inds[i-my_col_st+1] = getW(iters,m1,m2,n,colsOver,rowsUp);
        vals[i-my_col_st+1] = 1;
        i+=1;act_tot_nnz+=1;
      }
      else if (not onN and onW){
        inds[i-my_col_st+1] = getN(iters,m1,m2,n,colsOver,rowsUp);
        vals[i-my_col_st+1] = 1;
        i+=1;act_tot_nnz+=1;
      }
    }
    else if (NE) {
      if (not onN and not onE) {
        inds[i-my_col_st+1] = getNE(iters,m1,m2,n,colsOver,rowsUp);
        inds[i-my_col_st+2] = getE(iters,m1,m2,n,colsOver,rowsUp);
        inds[i-my_col_st+3] = getN(iters,m1,m2,n,colsOver,rowsUp);
        vals[i-my_col_st+1] = 1;
        vals[i-my_col_st+2] = 1;
        vals[i-my_col_st+3] = 1;
        i+=3;act_tot_nnz+=3;
      }
      else if (onN and not onE){
        inds[i-my_col_st+1] = getE(iters,m1,m2,n,colsOver,rowsUp);
        vals[i-my_col_st+1] = 1;
        i+=1;act_tot_nnz+=1;
      }
      else if (not onN and onE){
        inds[i-my_col_st+1] = getN(iters,m1,m2,n,colsOver,rowsUp);
        vals[i-my_col_st+1] = 1;
        i+=1;act_tot_nnz+=1;
      }
    }
    else if (SE) {
      if (not onS and not onE) {
        inds[i-my_col_st+1] = getSW(iters,m1,m2,n,colsOver,rowsUp);
        inds[i-my_col_st+2] = getS(iters,m1,m2,n,colsOver,rowsUp);
        inds[i-my_col_st+3] = getW(iters,m1,m2,n,colsOver,rowsUp);
        vals[i-my_col_st+1] = 1;
        vals[i-my_col_st+2] = 1;
        vals[i-my_col_st+3] = 1;
        i+=3;act_tot_nnz+=3;
      }
      else if (onS and not onE){
        inds[i-my_col_st+1] = getE(iters,m1,m2,n,colsOver,rowsUp);
        vals[i-my_col_st+1] = 1;
        i+=1;act_tot_nnz+=1;
      }
      else if (not onS and onE){
        inds[i-my_col_st+1] = getS(iters,m1,m2,n,colsOver,rowsUp);
        vals[i-my_col_st+1] = 1;
        i+=1;act_tot_nnz+=1;
      }
    }
    else if (SW) {
      if (not onS and not onW) {
        inds[i-my_col_st+1] = getSE(iters,m1,m2,n,colsOver,rowsUp);
        inds[i-my_col_st+2] = getS(iters,m1,m2,n,colsOver,rowsUp);
        inds[i-my_col_st+3] = getE(iters,m1,m2,n,colsOver,rowsUp);
        vals[i-my_col_st+1] = 1;
        vals[i-my_col_st+2] = 1;
        vals[i-my_col_st+3] = 1;
        i+=3;act_tot_nnz+=3;
      }
      else if (onS and not onW){
        inds[i-my_col_st+1] = getW(iters,m1,m2,n,colsOver,rowsUp);
        vals[i-my_col_st+1] = 1;
        i+=1;act_tot_nnz+=1;
      }
      else if (not onS and onW){
        inds[i-my_col_st+1] = getS(iters,m1,m2,n,colsOver,rowsUp);
        vals[i-my_col_st+1] = 1;
        i+=1;act_tot_nnz+=1;
      }
    }
    else if (N and not onN) {
      inds[i-my_col_st+1] = getN(iters,m1,m2,n,colsOver,rowsUp);
      vals[i-my_col_st+1] = 1;
      i+=1;act_tot_nnz+=1;
    }
    else if (E and not onE) {
      inds[i-my_col_st+1] = getE(iters,m1,m2,n,colsOver,rowsUp);
      vals[i-my_col_st+1] = 1;
      i+=1;act_tot_nnz+=1;
    }
    else if (S and not onS) {
      inds[i-my_col_st+1] = getS(iters,m1,m2,n,colsOver,rowsUp);
      vals[i-my_col_st+1] = 1;
      i+=1;act_tot_nnz+=1;
    }
    else if (W and not onW) {
      inds[i-my_col_st+1] = getW(iters,m1,m2,n,colsOver,rowsUp);
      vals[i-my_col_st+1] = 1;
      i+=1;act_tot_nnz+=1;
    }
  }
  G->write(act_tot_nnz,inds,vals);
  free(inds);
  free(vals);
}

int gather(int      n,
           int     m1,
           int     m2,
           World & dw){
  int lens_u[] = {m1,m2,n,n};
  Tensor<double> u(4,lens_u);
  int lens_G[] = {m1,m2,n,n};
  Tensor<double> G(4,lens_G);
  u.fill_random(0.0,1.0);
  double fg_time = MPI_Wtime();
  fill_gather(&G,m1,m2,n,dw);
  double fgexe_time = MPI_Wtime() - fg_time;
  printf("make gather time is: %lf\n", fgexe_time);
  Tensor<double> v(4,lens_u);
  double st_time = MPI_Wtime();
  v["ijkl"] = G["iakb"]*G["cjdl"]*u["cadb"];
  double exe_time = MPI_Wtime() - st_time;
  printf("exec time is: %lf\n", exe_time);
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
  int rank, np, n, pass, m1, m2;
  int const in_num = argc;
  char ** input_str = argv;
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  
  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 3;
  } else n = 3;
  
  if (getCmdOption(input_str, input_str+in_num, "-m1")){
    m1 = atoi(getCmdOption(input_str, input_str+in_num, "-m1"));
    if (m1 < 0) m1 = 3;
  } else m1 = 3;
  
  if (getCmdOption(input_str, input_str+in_num, "-m2")){
    m2 = atoi(getCmdOption(input_str, input_str+in_num, "-m2"));
    if (m2 < 0) m2 = 3;
  } else m2 = 3;

  {
    World dw(argc, argv);
    if (rank == 0){
      printf("Running gather with %d*%d*%d*%d grid\n",m1,m2,n,n);
    }
    pass = gather(n,m1,m2,dw);
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
