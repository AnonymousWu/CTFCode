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
 *        unstructured mesh.
 *        Note: Only works on elem**2, where elem is an int
 *        Also, (elem**2*n*n)**2 cannot go above INT_MAX
 */

void fillWV(int * wv,int n){
  bool left = true;
  int row = 1;
  for (int i = 0; i < 4*(n-2); i++){
    if (i < n-2){
      wv[i] = i+1;
    }
    else if (i < 3*(n-2)){
      if (left){
        wv[i] = row*n;
      }
      else {
        wv[i] = row*n + n - 1;
        row++;
      }
      left = !left;
    }
    else {
      wv[i] = row * n + i%(n-2) + 1;
    }
  }
}

int64_t checkColWWStart(int64_t * my_col_st, int * wallVerts, int n){
  int64_t my_m = *my_col_st % n*n;
  int i;
  for (i = 0; i < 4*(n-2); i++){
    if (wallVerts[i] >= my_m){
      break;
    }
  }
  int add = wallVerts[i] - my_m;
  *my_col_st += add;
  return i;
}

int64_t checkColCCStart(int64_t * my_col_st, int * cornerVerts, int n){
  int64_t my_m = *my_col_st % n*n;
  int i;
  for (i = 0; i < 4; i++){
    if (cornerVerts[i] >= my_m){
      break;
    }
  }
  int add = cornerVerts[i] - my_m;
  *my_col_st += add;
  return i;
}

int64_t incGww(int64_t i, int64_t n, int * wallVerts, int pos){
  int newPos = pos + 1;
  if (newPos == 4*(n-2)) return i + 2;
  return i + wallVerts[newPos] - wallVerts[pos];
}

int64_t incGcc(int64_t i, int64_t n, int * cornerVerts, int pos){
  int newPos = pos + 1;
  if (newPos == 4) return i + 1;
  return i + cornerVerts[newPos] - cornerVerts[pos];
}

int64_t findBot(int64_t n, int64_t elems, int64_t i){
  return i - (elems-1)*n*n - n;
}

int64_t findLeft(int64_t n, int64_t elems, int64_t i){
  return i + n*n - n + 1;
}

int64_t findRight(int64_t n, int64_t elems, int64_t i){
  return i - n*n + n - 1;
}

int64_t findTop(int64_t n, int64_t elems, int64_t i){
  return i + (elems-1)*n*n + n;
}

int64_t findBL(int64_t n, int64_t elems, int64_t i){
  return findBot(n,elems,findLeft(n,elems,i));
}

int64_t findBR(int64_t n, int64_t elems, int64_t i){
  return findBot(n,elems,findRight(n,elems,i));
}

int64_t findTL(int64_t n, int64_t elems, int64_t i){
  return findTop(n,elems,findLeft(n,elems,i));
}

int64_t findTR(int64_t n, int64_t elems, int64_t i){
  return findTop(n,elems,findRight(n,elems,i));
}

int64_t getPos(int64_t x, int64_t y, int64_t lenX){
  return x + y*lenX;
}

void fill_Gww(Tensor<double> & Gww, int elems, int n, World & dw){
  int64_t elems2 = elems*elems;
  int64_t len = elems2*n*n;

  int64_t my_col = len/dw.np;
  int64_t my_col_st = dw.rank*my_col;
  if (dw.rank < len%dw.np) my_col++;
  my_col_st += std::min((int)dw.rank, (int)len%dw.np);
  
  int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*2*my_col);
  double * vals = (double*)malloc(sizeof(double)*2*my_col);
  
  int * wallVerts = (int*)malloc(sizeof(int)*4*(n-2));
  fillWV(wallVerts,n);
  
  int64_t pos_in_arr = checkColWWStart(&my_col_st, wallVerts,n);
  bool left = true;
  int64_t act_tot_nnz = 0;
  int64_t oldPos = pos_in_arr;
  for (int64_t i = my_col_st; i < my_col_st + my_col; i = incGww(i,n,wallVerts,oldPos)){
    oldPos = pos_in_arr;
    bool botValid = pos_in_arr < n-2 and i > elems*n*n;
    bool leftValid = (pos_in_arr < 3*(n-2) and pos_in_arr >= (n-2) and left) and i/(n*n) % elems != 0;
    bool rightValid = (pos_in_arr < 3*(n-2) and pos_in_arr >= (n-2) and not left) and i/(n*n) % elems != elems-1;
    bool topValid = pos_in_arr >= 3*(n-2) and i/(elems*n*n) != elems-1;
    if (botValid){
      inds[act_tot_nnz] = getPos(i,findBot(n,elems,i),elems*n*n);
      vals[act_tot_nnz] = 1;
      act_tot_nnz++;
    }
    else if (leftValid){
      inds[act_tot_nnz] = getPos(i,findLeft(n,elems,i),elems*n*n);
      vals[act_tot_nnz] = 1;
      act_tot_nnz++;
      left = not left;
    }
    else if (rightValid){
      inds[act_tot_nnz] = getPos(i,findRight(n,elems,i),elems*n*n);
      vals[act_tot_nnz] = 1;
      act_tot_nnz++;
      left = not left;
    }
    else if (topValid){
      inds[act_tot_nnz] = getPos(i,findTop(n,elems,i),elems*n*n);
      vals[act_tot_nnz] = 1;
      act_tot_nnz++;
    }
    pos_in_arr = (pos_in_arr + 1) % (4*(n-2));
  }
  Gww.write(act_tot_nnz,inds,vals);
  free(inds);
  free(vals);
  free(wallVerts);
}

void fill_Gcc(Tensor<double> & Gcc, int elems, int n, World & dw){
  int elems2 = elems * elems;
  int64_t len = elems2*n*n;

  int64_t my_col = len/dw.np;
  int64_t my_col_st = dw.rank*my_col;
  if (dw.rank < len%dw.np) my_col++;
  my_col_st += std::min((int)dw.rank, (int)len%dw.np);
  
  int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*4*my_col);
  double * vals = (double*)malloc(sizeof(double)*4*my_col);
  
  int cornerVerts[] = {0,n-1,n*n-n,n*n-1}; //makes this O(elements) rather than O(points)
  
  int64_t pos_in_arr = checkColCCStart(&my_col_st, cornerVerts,n);
  int64_t act_tot_nnz = 0;
  int64_t oldPos = pos_in_arr;
  for (int64_t i = my_col_st; i < my_col_st + my_col; i = incGcc(i,n,cornerVerts,oldPos)){
    oldPos = pos_in_arr;
    bool botValid = (pos_in_arr == 0 or pos_in_arr == 1) and i > elems*n*n;
    bool leftValid = (pos_in_arr == 0 or pos_in_arr == 2) and i/(n*n) % elems != 0;
    bool rightValid = (pos_in_arr == 1 or pos_in_arr == 3) and i/(n*n) % elems != elems-1;
    bool topValid = (pos_in_arr == 2 or pos_in_arr == 3) and i/(elems*n*n) != elems-1;
    if (botValid){
      inds[act_tot_nnz] = getPos(i,findBot(n,elems,i),elems*n*n);
      vals[act_tot_nnz] = 1;
      act_tot_nnz++;
    }
    if (leftValid){
      inds[act_tot_nnz] = getPos(i,findLeft(n,elems,i),elems*n*n);
      vals[act_tot_nnz] = 1;
      act_tot_nnz++;
    }
    if (rightValid){
      inds[act_tot_nnz] = getPos(i,findRight(n,elems,i),elems*n*n);
      vals[act_tot_nnz] = 1;
      act_tot_nnz++;
    }
    if (topValid){
      inds[act_tot_nnz] = getPos(i,findTop(n,elems,i),elems*n*n);
      vals[act_tot_nnz] = 1;
      act_tot_nnz++;
    }
    if (botValid and leftValid){
      inds[act_tot_nnz] = getPos(i,findBL(n,elems,i),elems*n*n);
      vals[act_tot_nnz] = 1;
      act_tot_nnz++;
    }
    else if (botValid and rightValid){
      inds[act_tot_nnz] = getPos(i,findBR(n,elems,i),elems*n*n);
      vals[act_tot_nnz] = 1;
      act_tot_nnz++;
    }
    else if (topValid and leftValid){
      inds[act_tot_nnz] = getPos(i,findTL(n,elems,i),elems*n*n);
      vals[act_tot_nnz] = 1;
      act_tot_nnz++;
    }
    else if (topValid and rightValid){
      inds[act_tot_nnz] = getPos(i,findTR(n,elems,i),elems*n*n);
      vals[act_tot_nnz] = 1;
      act_tot_nnz++;
    }
    pos_in_arr = (pos_in_arr + 1) % 4;
  }
  
  Gcc.write(act_tot_nnz,inds,vals);
  free(inds);
  free(vals);
}

//TODO: Try actually constructing the P matricies to measure speed/correct
int gather(int      n,
           int  elems,
           World & dw){
  int lens_u[] = {elems*n*n};
  int lens_Gcc[] = {elems*n*n,elems*n*n};
  int lens_Gww[] = {elems*n*n,elems*n*n};
  Tensor<double> u(1,lens_u);
  Tensor<double> Gcc(2,lens_Gcc);
  Tensor<double> Gww(2,lens_Gww);
  u.fill_random(0.0,1.0);
  
  double fg_time = MPI_Wtime();
  
  fill_Gcc(Gcc,elems,n,dw);
  
  double fgexe_time = MPI_Wtime() - fg_time;
  printf("make Gcc time is: %lf\n", fgexe_time);
  
  fg_time = MPI_Wtime();
  
  fill_Gww(Gww,elems,n,dw);
  
  fgexe_time = MPI_Wtime() - fg_time;
  printf("make Gww time is: %lf\n", fgexe_time);
  
  double st_time = MPI_Wtime();
  
  u["i"] += Gcc["ij"]*u["j"];
  
  double exe_time = MPI_Wtime() - st_time;
  printf("Gcc exec time is: %lf\n", exe_time);
  
  st_time = MPI_Wtime();
  
  u["i"] += Gww["ij"]*u["j"];
  
  exe_time = MPI_Wtime() - st_time;
  printf("Gww exec time is: %lf\n", exe_time);
  
  
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
