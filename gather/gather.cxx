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

int64_t getVertBoundCol(int i, int m1, int m2, int n){
  int rowsUp = (int)(double)i/(double)(m1*n*n*m2)*m2;
  int internalRepeat = (rowsUp-1)*(m1-1)*(2*n-1); //blocks*(left+bottom points - corner)
  int leftRepeat = (rowsUp-1)*n;
  int botRepeat = (m1-1)*n;
  int botPoints = (n-1)*(n-1)*(m1-1);
  return i - (internalRepeat+leftRepeat+botRepeat+botPoints)-n;
}

int64_t getVertCol(int i, int m1, int m2, int n){
  int rowsUp = (int)(double)i/(double)(m1*n*n*m2)*m2;
  int leftRepeat = rowsUp*n;
  int botRepeat = (m1-1)*n;
  int internalRepeat = (rowsUp-1)*(m1-1)*(2*n-1);
  return i - (leftRepeat+botRepeat+internalRepeat);
}

int64_t getBotBoundCol(int i, int m1, int m2, int n){
  int colsOver = (int)((double)i/(double)(m1*n*n)*m1) % m1;
  int internalHeight = (i%(n*n))/n;
  if (i < 2*n*n) return i - ((colsOver-1)*n + internalHeight*n + (n-internalHeight-1)*n + 2);
  return i - ((colsOver-1)*n + internalHeight*n + (n-internalHeight-1)*(n-1) + 2);
}

int64_t getBotCol(int i, int m1, int m2, int n){
  int colsOver = (int)((double)i/(double)(m1*n*n)*m1) % m1;
  int internalHeight = (i%(n*n))/n;
  return i - ((colsOver-1)*n + internalHeight + 1);
}

int64_t getInternal(int i, int m1, int m2, int n){
  int rowsUp = (int)(double)i/(double)(m1*n*n*m2)*m2;
  int colsOver = (int)((double)i/(double)(m1*n*n)*m1) % m1;
  int rowsFull = (rowsUp-1)*m1*n;
  int colsFull = rowsUp*(m1-1)*(n-1);
  int rowCols = (colsOver+1)*n;
  int colCols = colsOver*(n-1);
  int internalHeight = (i%(n*n))/n;
  return i - (rowsFull + colsFull + rowCols + colCols + internalHeight);
}

int64_t getInternalBottom(int i, int m1, int m2, int n){
  int64_t internal = getInternal(i+n,m1,m2,n);
  int colsOver = (int)((double)i/(double)(m1*n*n)*m1) % m1;
  int currRowOff = (colsOver-1)*(n-1)*(n-1) + n*(n-1);
  int nextRow = (m1-colsOver-1)*(n-1)*(n-1);
  if (i < 2*m1*n*n) nextRow = (m1-colsOver-1)*(n-1)*n;
  return internal - currRowOff - nextRow - (n-1);
}

int64_t getInternalLeft(int i, int m1, int m2, int n){
  return getInternal(i+1,m1,m2,n) - (n-1)*(n-1) + 1;
}

int64_t getInternalCorner(int i, int m1, int m2, int n){
  int64_t bottomPoint = getInternalBottom(i+1,m1,m2,n);
  int rowsUp = (int)(double)i/(double)(m1*n*n*m2)*m2;
  if (rowsUp == 1) return bottomPoint - (n-1)*(n-1);
  return bottomPoint - (n-2)*(n-1);
}

int64_t getPosInMtx(int64_t col, int64_t row, int64_t w){
  return row*w + col;
}

void fill_gather(Tensor<double> & G, int m1, int m2, int n, World & dw){
  int64_t w = n*n*m1*m2;
  int64_t h = w;
  
  int64_t my_col = h/dw.np;
  int64_t my_col_st = dw.rank*my_col;
  if (dw.rank < h%dw.np) my_col++;
  my_col_st += std::min((int)dw.rank, (int)h%dw.np);
  
  int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*my_col); //val = i*len(j) + j
  double * vals = (double*)malloc(sizeof(double)*my_col);
  
  for (int64_t i = my_col_st; i < my_col_st+my_col; i++){
    if (i < n*n){
      //bottom right
      inds[i-my_col_st] = i*w + i; //i = j
    }
    else if (i < m1*n*n){
      //on the bottom row
      if (i % n == 0){
        //left side of the square. Get idx of previous.
        inds[i-my_col_st] = getPosInMtx(getBotBoundCol(i,m1,m2,n),i,w);
      }
      else {
        //internal to the local square
        inds[i-my_col_st] = getPosInMtx(getBotCol(i,m1,m2,n),i,w);
      }
    }
    else if (i % (m1*n*n) < n*n){
      //on the left side
      if (i % (n*n) < n){
        //on the bottom
        inds[i-my_col_st] = getPosInMtx(getVertBoundCol(i,m1,m2,n),i,w);
      }
      else {
        inds[i-my_col_st] = getPosInMtx(getVertCol(i,m1,m2,n),i,w);
      }
    }
    else {
      //internal
      if (i % (n*n) < n and i % n == 0){
        //corner
        inds[i-my_col_st] = getPosInMtx(getInternalCorner(i,m1,m2,n),i,w);
      }
      else if (i % (n*n) < n){
        //on the bottom
        inds[i-my_col_st] = getPosInMtx(getInternalBottom(i,m1,m2,n),i,w);
      }
      else if (i % n == 0){
        //left of the square
        inds[i-my_col_st] = getPosInMtx(getInternalLeft(i,m1,m2,n),i,w);
      }
      else {
        inds[i-my_col_st] = getPosInMtx(getInternal(i,m1,m2,n),i,w);
      }
    }
    vals[i-my_col_st] = 1;
  }
  G.write(my_col,inds,vals);
  free(inds);
  free(vals);
}

int gather(int      n,
           int     m1,
           int     m2,
           World & dw){
  int lens_u[] = {n*n*m1*m2};
  Tensor<> u(1,lens_u);
  int lens_G[] = {n*n*m1*m2,n*n*m1*m2};
  Tensor<double> G(2,lens_G);
  u.fill_random(0.0,1.0);
  fill_gather(G,m1,m2,n,dw);
  Tensor<double> v(1,lens_u);
  double st_time = MPI_Wtime();
  v["i"] = G["ij"]*u["j"];
  double exe_time = MPI_Wtime() - st_time;
  printf("exec time is: %lf\n", exe_time);
  return -1;
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
