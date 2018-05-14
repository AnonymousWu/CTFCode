/** \addtogroup examples 
  * @{ 
  * \defgroup gather
  * @{ 
  * \brief Gather test/benchmark for irregular grid (2D)
  */
#include <ctf.hpp>
using namespace CTF;

/**
 * \brief This Gather/Scatter method will be based on tensor 
 *        contractions capable of having a truly unstructured grid.
 *        This means that it does not have to conform to any regular pattern,
 *        and that any 2-d grid of any shape can be exploited here.
 *        However, user has to impliment all the details of the mesh
 *        themselves, thus incurring extra setup time.
 */


/**
 * Gets the (x,y,z,w) position of an order 4 tensor
 * @param x,y,z,w indices in the tensor
 * @param lenX,lenY,lenZ size of x,y,z indices
 * @return position in the tensor
 */
int64_t getPos(int64_t x, int64_t y, int64_t z, int64_t w,
               int64_t lenX, int64_t lenY, int64_t lenZ){
  return x + y*lenX + z*lenX*lenY + w*lenX*lenY*lenZ;
}

/**
 * Same as getPos(), but can support order 6 tensor
 * @param x,y,z,a,b,c indices in the tensor
 * @param lenX,lenY,lenZ,lenA,lenB size of x,y,z,a,b indices
 * @return position in the tensor
 */
int64_t getPos6(int64_t x, int64_t y, int64_t z, int64_t a,
                int64_t b, int64_t c, int64_t lenX, int64_t lenY,
                int64_t lenZ, int64_t lenA, int64_t lenB){
  return x + y*lenX + z*lenX*lenY + a*lenX*lenY*lenZ + 
         b*lenX*lenY*lenZ*lenA + c*lenX*lenY*lenZ*lenA*lenB;
}

/**
 * Traverses every corner in an element and returns their neighboring DOFs
 * @param currElem current element in the grid to work on
 * @param edges global array of wall-wall connectivity
 * @param cornerVals whether to return the element number or the corner number (T/F)
 * @return corners 4xVector length array of each corner connectivity (Last element is always negative)
 */
int64_t** traverseCorner(int64_t currElem, int64_t** edges, bool cornerVals){
  int64_t** corners = (int64_t**)malloc(sizeof(int64_t*)*4);
  for (int i = 0; i < 4; i++){
    corners[i] = (int64_t*)malloc(sizeof(int64_t)*10); //will realloc later if need be
    int64_t elemsInArr = 10;
    int posToTraverse = i;
    int64_t tempElem = currElem;
    int64_t currLen = 0;
    while (edges[tempElem][posToTraverse] != currElem){
      if (edges[tempElem][posToTraverse] == -1){
        //don't add up anything
        corners[i][0] = -1;
        break;
      }
      if (currLen*.7 > elemsInArr){
        corners[i] = (int64_t*)realloc(corners[i],sizeof(int64_t)*elemsInArr*elemsInArr);
        elemsInArr *= elemsInArr;
      }
      int64_t nextElem = edges[tempElem][posToTraverse];
      int posOfPrev = 0;
      for (; posOfPrev < 4; posOfPrev++){
        if (edges[nextElem][posOfPrev] == tempElem){
          break;
        }
      }
      if (posOfPrev == 4){
        printf("Bad mesh. Exiting...\n");
        exit(10);
      }
      if (currLen != 0 and edges[currElem][(i+1)%4] != nextElem){
        if (cornerVals) corners[i][currLen-1] = nextElem;
        else corners[i][currLen-1] = (posToTraverse + 1) % 4;
      }
      if (edges[currElem][(i+1)%4] == nextElem){
        corners[i][currLen-1] = -2;
      }
      tempElem = nextElem;
      posToTraverse = (posOfPrev + 3) % 4;
      currLen++;
    }
  }
  return corners;
}

/**
 * Finds where the other element borders the current element
 * @param edges global array of wall-wall connectivity
 * @param otherElem other element that current element is connected to
 * @param elem current element this is working with
 * @return index of other elem's wall that equals the current elem
 */
int getOtherWall(int64_t**edges, int64_t otherElem, int64_t elem){
  for (int i = 0; i < 4; i++){
    if (edges[otherElem][i] == elem){
      return i;
    }
  }
  return -1;
}

void updatePos(int64_t*** elemIndsInfo, int64_t currElem, int cornerNo, 
               int64_t currPos, int64_t& nextIdx1, int64_t& nextIdx2, int n){
  if (elemIndsInfo[currElem][cornerNo][currPos] == 0){
    nextIdx1 = n-1;
    nextIdx2 = n-1;
  }
  else if (elemIndsInfo[currElem][cornerNo][currPos] == 1){
    nextIdx1 = n-1;
    nextIdx2 = 0;
  }
  else if (elemIndsInfo[currElem][cornerNo][currPos] == 2){
    nextIdx1 = 0;
    nextIdx2 = 0;
  }
  else if (elemIndsInfo[currElem][cornerNo][currPos] == 3){
    nextIdx1 = 0;
    nextIdx2 = n-1;
  }
}

/**
 * Checks whether the gather/scatter method worked
 * @param u modified grid after preforming the gather/scatter method
 * @param uorig unmodified grid
 * @param elems number of elements
 * @param n number of degrees of freedom
 * @param edges global array of wall-wall connectivity
 * @param dw world where ctf lives in
 * @return good whether the resulting method worked
 */
int checkCorrect(Tensor<double>& u, Tensor<double>& uorig,
                 int64_t elems, int n, int64_t** edges, World& dw){
  int64_t len = elems*n*n;
  int64_t my_col = len/dw.np;
  int64_t my_col_st = dw.rank*my_col;
  if (dw.rank < len%dw.np) my_col++;
  my_col_st += std::min((int)dw.rank, (int)len%dw.np);
  
  double* uinds;
  double* uoinds;
  int64_t unum, uonum;
  double* ucompi = (double*)malloc(sizeof(double*)*elems*n*n);
  u.read_all(&unum,&uinds);
  uorig.read_all(&uonum,&uoinds);
  
  int currPos = 0;
  int64_t nextIdx1 = 0;
  int64_t nextIdx2 = 0;
  
  int64_t*** elemCornerInfo = (int64_t***)malloc(sizeof(int64_t**)*elems);
  int64_t*** elemIndsInfo = (int64_t***)malloc(sizeof(int64_t**)*elems);
  for (int64_t i = 0; i < elems; i++){
    elemCornerInfo[i] = traverseCorner(i,edges,true); 
    elemIndsInfo[i] = traverseCorner(i,edges,false);
  }
  
  for (int64_t i = my_col_st; i < my_col_st + my_col; i++){
    ucompi[i] = uoinds[i];
    int64_t currIdx = i;
    int64_t currElem = currIdx % elems;
    int64_t currIdx1 = (currIdx/elems) % n;
    int64_t currIdx2 = currIdx/(elems*n);
    
    bool isEdge0 = currIdx2 == n-1;
    bool isEdge1 = currIdx1 == n-1;
    bool isEdge2 = currIdx2 == 0;
    bool isEdge3 = currIdx1 == 0;
    
    bool isCorner0 = isEdge0 and isEdge1;
    bool isCorner1 = isEdge1 and isEdge2;
    bool isCorner2 = isEdge2 and isEdge3;
    bool isCorner3 = isEdge3 and isEdge0;
    
    bool isEdge = isEdge0 or isEdge1 or isEdge2 or isEdge3;
    bool isCorner = isCorner0 or isCorner1 or isCorner2 or isCorner3;
    
    if (isEdge){
      if (isEdge0){
        int64_t otherElem = edges[currElem][0];
        if (otherElem != -1){
          int otherWall = getOtherWall(edges,otherElem,currElem);
          if (otherWall == 0){
            ucompi[i] += uoinds[getPos(otherElem,n-1-currIdx1,n-1,0,elems,n,n)];
          }
          else if (otherWall == 1){
            ucompi[i] += uoinds[getPos(otherElem,n-1,currIdx1,0,elems,n,n)];
          }
          else if (otherWall == 2){
            ucompi[i] += uoinds[getPos(otherElem,currIdx1,0,0,elems,n,n)];
          }
          else if (otherWall == 3){
            ucompi[i] += uoinds[getPos(otherElem,0,n-1-currIdx1,0,elems,n,n)];
          }
          else{
            printf("Bad Mesh. Exiting...\n");
            exit(15);
          }
        }
      }
      if (isEdge1){
        int64_t otherElem = edges[currElem][1];
        if (otherElem != -1){
          int otherWall = getOtherWall(edges,otherElem,currElem);
          if (otherWall == 0){
            ucompi[i] += uoinds[getPos(otherElem,currIdx2,n-1,0,elems,n,n)];
          }
          else if (otherWall == 1){
            ucompi[i] += uoinds[getPos(otherElem,n-1,n-1-currIdx2,0,elems,n,n)];
          }
          else if (otherWall == 2){
            ucompi[i] += uoinds[getPos(otherElem,n-1-currIdx2,0,0,elems,n,n)];
          }
          else if (otherWall == 3){
            ucompi[i] += uoinds[getPos(otherElem,0,currIdx2,0,elems,n,n)];
          }
          else{
            printf("Bad Mesh. Exiting...\n");
            exit(15);
          }
        }
      }
      if (isEdge2){
        int64_t otherElem = edges[currElem][2];
        if (otherElem != -1){
          int otherWall = getOtherWall(edges,otherElem,currElem);
          if (otherWall == 0){
            ucompi[i] += uoinds[getPos(otherElem,currIdx1,n-1,0,elems,n,n)];
          }
          else if (otherWall == 1){
            ucompi[i] += uoinds[getPos(otherElem,n-1,n-1-currIdx1,0,elems,n,n)];
          }
          else if (otherWall == 2){
            ucompi[i] += uoinds[getPos(otherElem,n-1-currIdx1,0,0,elems,n,n)];
          }
          else if (otherWall == 3){
            ucompi[i] += uoinds[getPos(otherElem,0,currIdx1,0,elems,n,n)];
          }
          else{
            printf("Bad Mesh. Exiting...\n");
            exit(15);
          }
        }
      }
      if (isEdge3){
        int64_t otherElem = edges[currElem][3];
        if (otherElem != -1){
          int otherWall = getOtherWall(edges,otherElem,currElem);
          if (otherWall == 0){
            ucompi[i] += uoinds[getPos(otherElem,n-1-currIdx2,n-1,0,elems,n,n)];
          }
          else if (otherWall == 1){
            ucompi[i] += uoinds[getPos(otherElem,n-1,currIdx2,0,elems,n,n)];
          }
          else if (otherWall == 2){
            ucompi[i] += uoinds[getPos(otherElem,currIdx2,0,0,elems,n,n)];
          }
          else if (otherWall == 3){
            ucompi[i] += uoinds[getPos(otherElem,0,n-1-currIdx2,0,elems,n,n)];
          }
          else{
            printf("Bad Mesh. Exiting...\n");
            exit(15);
          }
        }
      }
    }
    
    if (isCorner){
      if (isCorner0 and elemCornerInfo[currElem][0][0] != -1){
        currPos = 0;
        while (elemCornerInfo[currElem][0][currPos] >= 0){
          updatePos(elemIndsInfo,currElem,0,currPos,nextIdx1,nextIdx2,n);
          ucompi[i] += uoinds[getPos(elemCornerInfo[currElem][0][currPos],nextIdx1,
                                     nextIdx2,0,elems,n,n)];
          currPos++;
        }
      }
      if (isCorner1 and elemCornerInfo[currElem][1][0] != -1){
        currPos = 0;
        while (elemCornerInfo[currElem][1][currPos] >= 0){
          updatePos(elemIndsInfo,currElem,1,currPos,nextIdx1,nextIdx2,n);
          ucompi[i] += uoinds[getPos(elemCornerInfo[currElem][1][currPos],nextIdx1,
                                     nextIdx2,0,elems,n,n)];
          currPos++;
        }
      }
      if (isCorner2 and elemCornerInfo[currElem][2][0] != -1){
        currPos = 0;
        while (elemCornerInfo[currElem][2][currPos] >= 0){
          updatePos(elemIndsInfo,currElem,2,currPos,nextIdx1,nextIdx2,n);
          ucompi[i] += uoinds[getPos(elemCornerInfo[currElem][2][currPos],nextIdx1,
                                     nextIdx2,0,elems,n,n)];
          currPos++;
        }
      }
      if (isCorner3 and elemCornerInfo[currElem][3][0] != -1){
        currPos = 0;
        while (elemCornerInfo[currElem][3][currPos] >= 0){
          updatePos(elemIndsInfo,currElem,3,currPos,nextIdx1,nextIdx2,n);
          ucompi[i] += uoinds[getPos(elemCornerInfo[currElem][3][currPos],nextIdx1,
                                     nextIdx2,0,elems,n,n)];
          currPos++;
        }
      }
    }
  }
  bool good = true;
  for (int64_t i = my_col_st; i < my_col_st + my_col; i++){
    if (fabs(ucompi[i] - uinds[i]) >= .000001){
      int64_t currIdx = i;
      int64_t currElem = currIdx % elems;
      int64_t currIdx1 = (currIdx/elems) % n;
      int64_t currIdx2 = currIdx/(elems*n);
      printf("%lf %lf\n", ucompi[i], uinds[i]);
      printf("FAIL! The operation failed at index %ld, or coordinate (%ld,%ld,%ld).\n",
              i, currElem, currIdx1, currIdx2);
      good = 0;
      break;
    }
  }
  if (good) printf("Congratulations! This gather/scatter operation worked!\n");
  for (int64_t i = 0; i < elems; i++){
    for(int j = 0; j < 4; j++){
      free(elemCornerInfo[i][j]);
      free(elemIndsInfo[i][j]);
    }
    free(elemCornerInfo[i]);
    free(elemIndsInfo[i]);
  }
  free(elemCornerInfo);
  free(elemIndsInfo);
  free(uinds);
  free(uoinds);
  free(ucompi);
  return good;
}

/**
 * Fills the corner gather tensor
 * @param,@return Gcc the corner gather tensor
 * @param edges global array of wall-wall connectivity
 * @param elems number of elements in the mesh
 * @param n number of degrees of freedom in each element
 * @param dw world which ctf lives in
 */
void fill_Gcc(Tensor<double>& Gcc, int64_t** edges, int64_t elems, int n, World& dw){
  int64_t*** elemCornerInfo = (int64_t***)malloc(sizeof(int64_t**)*elems);
  int64_t*** elemIndsInfo = (int64_t***)malloc(sizeof(int64_t**)*elems);
  
  int64_t my_col = elems/dw.np;
  int64_t my_col_st = dw.rank*my_col;
  if (dw.rank < elems%dw.np) my_col++;
  my_col_st += std::min((int)dw.rank, (int)elems%dw.np);
  
  int64_t* inds = (int64_t*)malloc(sizeof(int64_t)*4*my_col);
  double* vals = (double*)malloc(sizeof(double)*4*my_col);
  
  int64_t act_tot_nnz = 0;
  //whole mesh, each elem
  for (int64_t i = my_col_st; i < my_col_st + my_col; i++){
    elemCornerInfo[i] = traverseCorner(i,edges,true); 
    elemIndsInfo[i] = traverseCorner(i,edges,false);
    //each elem, go thru each corner
    for (int64_t j = 0; j < 4; j++){
      int64_t k = 0;
      while (elemCornerInfo[i][j][k] >= 0){
        int64_t z = elemIndsInfo[i][j][k];
        inds[act_tot_nnz] = getPos(i,elemCornerInfo[i][j][k],j,z,elems,elems,4);
        vals[act_tot_nnz] = 1;
        act_tot_nnz++;
        k++;
      }
    }
  }
  Gcc.write(act_tot_nnz,inds,vals);
  free(inds);
  free(vals);
  for (int64_t i = my_col_st; i < my_col_st + my_col; i++){
    for(int j = 0; j < 4; j++){
      free(elemCornerInfo[i][j]);
      free(elemIndsInfo[i][j]);
    }
    free(elemCornerInfo[i]);
    free(elemIndsInfo[i]);
  }
  free(elemCornerInfo);
  free(elemIndsInfo);
}

/**
 * Fills the wall gather tensor
 * @param,@return Gww the corner gather tensor
 * @param edges global array of wall-wall connectivity
 * @param elems number of elements in the mesh
 * @param n number of degrees of freedom in each element
 * @param dw world which ctf lives in
 */
void fill_Gww(Tensor<double>& Gww, int64_t** edges, int64_t elems, int n, World& dw){
  int64_t len = elems*4;
  int64_t my_col = len/dw.np;
  int64_t my_col_st = dw.rank*my_col;
  if (dw.rank < len%dw.np) my_col++;
  my_col_st += std::min((int)dw.rank, (int)len%dw.np);
  int64_t* inds = (int64_t*)malloc(sizeof(int64_t)*my_col);
  double* vals = (double*)malloc(sizeof(double)*my_col);
  
  int64_t act_tot_nnz = 0;
  for (int64_t i = my_col_st; i < my_col_st + my_col; i++){
    
    int firstidx = i/4;
    int secondidx = i%4;
    int64_t newPos = 0;
    int myWall = 0;
    if (edges[firstidx][secondidx] != -1) myWall = getOtherWall(edges,edges[firstidx][secondidx],firstidx);
    if (secondidx == 0 and edges[firstidx][secondidx] != -1){
      if (myWall == 1){
        newPos = getPos6(firstidx,edges[firstidx][secondidx],0,1,0,0,elems,elems,2,2,2);
      }
      else if (myWall == 2){
        newPos = getPos6(firstidx,edges[firstidx][secondidx],1,1,0,0,elems,elems,2,2,2);
      }
      else {
        printf("Warning! This method will not work due to improper wall connectivity!\n");
      }
    }
    else if (secondidx == 1 and edges[firstidx][secondidx] != -1){
      if (myWall == 0){
        newPos = getPos6(firstidx,edges[firstidx][secondidx],0,0,0,1,elems,elems,2,2,2);
      }
      else if (myWall == 3){
        newPos = getPos6(firstidx,edges[firstidx][secondidx],1,0,0,1,elems,elems,2,2,2);
      }
      else {
        printf("Warning! This method will not work due to improper wall connectivity!\n");
      }
    }
    else if (secondidx == 2 and edges[firstidx][secondidx] != -1){
      if (myWall == 0){
        newPos = getPos6(firstidx,edges[firstidx][secondidx],0,0,1,1,elems,elems,2,2,2);
      }
      else if (myWall == 3){
        newPos = getPos6(firstidx,edges[firstidx][secondidx],1,0,1,1,elems,elems,2,2,2);
      }
      else {
        printf("Warning! This method will not work due to improper wall connectivity!\n");
      }
    }
    else if (secondidx == 3 and edges[firstidx][secondidx] != -1) {
      if (myWall == 1){
        newPos = getPos6(firstidx,edges[firstidx][secondidx],0,1,1,0,elems,elems,2,2,2);
      }
      else if (myWall == 2){
        newPos = getPos6(firstidx,edges[firstidx][secondidx],1,1,1,0,elems,elems,2,2,2);
      }
      else {
        printf("Warning! This method will not work due to improper wall connectivity!\n");
      }
    }
    if (edges[firstidx][secondidx] != -1){
      inds[act_tot_nnz] = newPos;
      vals[act_tot_nnz] = 1;
      act_tot_nnz++;
    }
  }
  Gww.write(act_tot_nnz,inds,vals);
  free(inds);
  free(vals);
}

/**
 * Fills the wall permutation tensor
 * @param,@return Pw wall permutation tensor
 * @param n number of degrees of freedom in each element
 * @param dw world which CTF lives in 
 */
void fill_Pw(Tensor<double>& Pw, int64_t n, World& dw){
  int64_t len = 4*n*n*n;
  int64_t my_col = (4*n*n*n)/dw.np;
  int64_t my_col_st = dw.rank*my_col;
  if (dw.rank < len%dw.np) my_col++;
  my_col_st += std::min((int)dw.rank, (int)len%dw.np);
  
  int64_t* inds = (int64_t*)malloc(sizeof(int64_t)*my_col);
  double* vals = (double*)malloc(sizeof(double)*my_col);
  
  int64_t act_tot_nnz = 0;
  for (int64_t i = my_col_st; i < my_col_st + my_col; i++){
    
    int64_t idx0 = i%2;
    int64_t idx1 = (i%4)/2;
    int64_t idx2 = (i%(4*n))/4;
    int64_t idx3 = (i%(4*n*n))/(4*n);
    int64_t idx4 = i/(4*n*n);
    
    bool good0 = idx4 == n-1 and idx3 == idx2 and idx0 == 0 and idx1 == 0;
    bool good1 = idx3 == n-1 and idx4 == idx2 and idx0 == 0 and idx1 == 1;
    bool good2 = idx4 == 0 and idx3 == idx2 and idx0 == 1 and idx1 == 1;
    bool good3 = idx3 == 0 and idx4 == idx2 and idx0 == 1 and idx1 == 0;  
    
    bool cornersgood = good0 or good1 or good2 or good3;
    if (cornersgood){
      inds[act_tot_nnz] = i;
      vals[act_tot_nnz] = 1;
      act_tot_nnz++;
    }
  }
  Pw.write(act_tot_nnz, inds, vals);
  free(inds);
  free(vals);
}

/**
 * Fills the corner permutation tensors
 * @param,@return Pc1, Pc2 corner permutation tensors
 * @param n number of degrees of freedom in each element
 * @param dw world in which CTF lives in
 */
void fill_Pc(Tensor<double>& Pc1, Tensor<double>& Pc2, int64_t n, World& dw){
  int64_t len = 4;
  int64_t my_col = 4/dw.np;
  int64_t my_col_st = dw.rank*my_col;
  if (dw.rank < len%dw.np) my_col++;
  my_col_st += std::min((int)dw.rank, (int)len%dw.np);
  int64_t pos1[] = {getPos(0,n-1,0,0,4,0,0),
                    getPos(3,n-1,0,0,4,0,0),
                    getPos(1,0  ,0,0,4,0,0),
                    getPos(2,0  ,0,0,4,0,0)};
                    
  int64_t pos2[] = {getPos(0,n-1,0,0,4,0,0),
                    getPos(1,n-1,0,0,4,0,0),
                    getPos(2,0  ,0,0,4,0,0),
                    getPos(3,0  ,0,0,4,0,0)};
  int64_t* write1 = (int64_t*)malloc(sizeof(int64_t)*4);
  int64_t* write2 = (int64_t*)malloc(sizeof(int64_t)*4);
  double* write3 = (double*)malloc(sizeof(double)*4);
  int act_tot_nnz = 0;
  for (int i = my_col_st; i < my_col_st + my_col; i++){
    write1[act_tot_nnz] = pos1[i];
    write2[act_tot_nnz] = pos2[i];
    write3[act_tot_nnz] = 1;
    act_tot_nnz++;    
  }
  Pc1.write(act_tot_nnz,write1,write3);
  Pc2.write(act_tot_nnz,write2,write3);
  free(write1);
  free(write2);
  free(write3);
}

/**
 * Wrapper function to fill the permutation tensors
 * @param,@return Pc1, Pc2, Pw permutation tensors
 * @param n number of degrees of freedom in each element
 * @param dw world which CTF lives in
 */
void fill_P(Tensor<double>& Pc1, Tensor<double>& Pc2, Tensor<double>& Pw, int64_t n, World& dw){
  fill_Pw(Pw,n,dw);
  fill_Pc(Pc1,Pc2,n,dw);
}

/**
 * Function that preforms the gather/scatter contractions
 * @param n number of degrees of freedom in each element
 * @param elems number of elements in the mesh
 * @param edges global array of wall-wall connectivity
 * @param dw world in which CTF lives in
 */
int gather(int n, int64_t elems, int64_t** edges, World& dw){
  int lens_Gww[] = {(int)elems,(int)elems,2,2,2,2};
  int lens_Gcc[] = {(int)elems,(int)elems,4,4};
  int lens_Pw[] = {2,2,n,n,n};
  int lens_Pc[] = {4,n};
  int lens_u[] = {(int)elems,n,n};
  Tensor<double> u(3,lens_u);
  Tensor<double> uorig(3,lens_u);
  Tensor<double> utemp(3,lens_u);
  Tensor<double> Pw(5,lens_Pw);
  Tensor<double> Pc1(2,lens_Pc);
  Tensor<double> Pc2(2,lens_Pc);
  Tensor<double> Gww(6,lens_Gww);
  Tensor<double> Gcc(4,lens_Gcc);
  u.fill_random(0,1);
  uorig["ijk"] = u["ijk"];
  
  /*************Make*************/
  double make_time = MPI_Wtime();
  
  fill_Gww(Gww,edges,elems,n,dw);
  fill_P(Pc1,Pc2,Pw,n,dw);
  fill_Gcc(Gcc,edges,elems,n,dw);
  
  double end_make = MPI_Wtime() - make_time;
  printf("Make time is: %lf\n", end_make);
  
  /*************Exec*************/
  double st_time = MPI_Wtime();
  
  utemp["ijk"]  = Pw["cdzjk"]*Gww["iwabcd"]*Pw["abzxy"]*u["wxy"];
  utemp["ijk"] += Pc1["ak"]*Pc2["aj"]*Gcc["iwab"]*Pc2["bq"]*Pc1["br"]*u["wqr"];
  u["ijk"] += utemp["ijk"];
  
  double exe_time = MPI_Wtime() - st_time;
  printf("exec time is: %lf\n",exe_time);
  
  /*************Check************/
  st_time = MPI_Wtime();
  
  int x = checkCorrect(u,uorig,elems,n,edges,dw);
  
  exe_time = MPI_Wtime() - st_time;
  printf("check time is: %lf\n",exe_time);
  
  return x;
}

/**
 * Fills the global array of wall-wall connectivity if user does not
 * @param,@return ret global array of wall-wall connectivity
 * @param elems number of elements in the mesh
 */
void fillRetAuto(int64_t** ret, int64_t elems){
  for (int64_t i = 0; i < elems*elems; i++){
    ret[i] = (int64_t*)malloc(sizeof(int64_t*)*4);
    int64_t top = (i >= elems*elems-elems) ? -1 : i + elems;
    int64_t right = (i%elems == elems-1) ? -1 : i+1;
    int64_t bot = (i < elems) ? -1 : i - elems;
    int64_t left = (i%elems == 0) ? -1 : i-1;
    ret[i][0] = top;
    ret[i][1] = right;
    ret[i][2] = bot;
    ret[i][3] = left;
  }
}

/**
 * Fills the global array of wall-wall connectivity
 * @param n number of degrees of freedom in the mesh
 * @param elems number of elements in the mesh
 * @param y whether the user will fill in the mesh themselves or not (T/F)
 * @return ret global array of wall-wall connectivity
 */
int64_t** fillEdges(int n, int64_t elems, int y){
  int64_t ** ret;
  if (y){
    ret = (int64_t**)malloc(sizeof(int64_t*)*elems);
    for (int64_t i = 0; i < elems; i++){
      ret[i] = (int64_t*)malloc(sizeof(int64_t)*4);
      printf("Elem number %ld\n", i);
      for (int j = 0; j < 4; j++){
        printf("Side number %d\n", j);
        int64_t side;
        std::cin >> side;
        ret[i][j] = side;
      }
    }
  }
  else{
    ret = (int64_t**)malloc(sizeof(int64_t*)*elems*elems);
    fillRetAuto(ret,elems);
  }
  return ret;
}

#ifndef TEST_SUITE
/**
 * Gets the option in the command line
 * @param begin,end positions in the command line argument
 * @param option what option to look for
 * @return value of the option in the command line
 */
char* getCmdOption(char** begin, char** end, const std::string& option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}

/**
 * Main function to start the gather/scatter operation
 * @param argc number of arguments
 * @param argv command line arguments
 * @return whether this was successful
 */
int main(int argc, char** argv){
  int rank, np, n, pass, y;
  int64_t elems;
  int const in_num = argc;
  char ** input_str = argv;
  
  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 3;
  } else n = 3;
  
  if (getCmdOption(input_str, input_str+in_num, "-elems")){
    elems = atol(getCmdOption(input_str, input_str+in_num, "-elems"));
    if (elems < 0) elems = 3;
  } else elems = 3;
  
  if (getCmdOption(input_str,input_str+in_num,"-y")){
    y = atoi(getCmdOption(input_str, input_str+in_num, "-y"));
  } else y = 0;
  
  int64_t** elemEdges = fillEdges(n,elems,y);
  
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  
  {
    World dw(argc, argv);
    if (rank == 0){
      if (y) printf("Running gather with %ld*%d*%d grid\n",elems,n,n);
      else {
        printf("Running gather with %ld*%d*%d grid\n",elems*elems,n,n);
      }
    }
    if (!y) elems *= elems;
    pass = gather(n,elems,elemEdges,dw);
    assert(pass);
  }
  for (int64_t i = 0; i < elems; i++){
    free(elemEdges[i]);
  }
  free(elemEdges);
  MPI_Finalize();
  return 0;
}
/**
 * @}
 * @}
 */

#endif
