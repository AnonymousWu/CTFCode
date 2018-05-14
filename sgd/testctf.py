import ctf,time,random
import numpy as np
import numpy.linalg as la
from ctf import random as crandom

class UnitTests:
  def testGetW(self):
    """Tests to see if getw works"""
    m=4
    n=3
    k=2
    W = crandom.random((m,k))
    omega = ctf.tensor((m,n))
    omega.fill_random(1,1)
    assert(W == getNewWCTF(omega[:,0],W,m,n,k))
    print("getWWorks!")
    
  def testCTFInverse(self):
    """Tests to see if CTF inverse function works"""
    m=4
    n=3
    k=2
    A = crandom.random((m,k))
    ctfInvA = CTFinverse(A)
    npA = la.pinv(A.to_nparray())
    for i in range(len(npA)):
      for j in range(len(npA[0])):
        assert(abs(ctfInvA[i][j] - npA[i][j]) < .00000000001)
    print("Inverse Works!")
  
  def testUpdateWCTF(self):
    """Tests to see if updating W works"""
    m=4
    n=3
    k=2
    H = getDenseMtx(n,k,0,1)
    W = getDenseMtx(m,k,0,1)
    pass
  
  def runAllTests(self):
    self.testGetW()
    self.testCTFInverse()
    self.testUpdateWCTF()

def CTFinverse(A):
  """Gets the pseudoinverse of matrix A"""
  U, sigma, VT = ctf.svd(A)
  sigmaInv = 1/sigma
  Ainv = ctf.zeros((A.to_nparray().shape[1],U.to_nparray().shape[0]))
  Ainv.i("ad") << VT.i("ba")*sigmaInv.i("b")*U.i("db")
  return Ainv


def getNewHCTF(omegai,H,m,n,k):
  '''gets the new W for each iteration'''
  newH = ctf.zeros((n,k))
  for j in range(n):
    if omegai[j] != 0:
      newH[j,:] = H[j,:]
  return newH
  
  
def updateW(A,H,regParam,omega,m,n,k):
  '''Gets the new W matrix by using the formula'''
  Wctf = ctf.zeros((m,k))
  identity = ctf.eye(k)
  for i in range(m):
    newH = getNewHCTF(omega[i],H,m,n,k)
    newMat = ctf.zeros((k,k))
    newMat.i("xy") << newH.i("bx")*newH.i("by")+regParam*identity.i("xy")
    inverseFirstPart = CTFinverse(newMat)
    temp = ctf.zeros((k))
    temp.i("j") << inverseFirstPart.i("ja")*H.i("ba")*A[i].i("b")
    Wctf[i] = temp
  return Wctf
  

def getNewWCTF(omegaj,W,m,n,k):
  '''gets the new W for each iteration'''
  newW = ctf.zeros((m,k))
  for i in range(m):
    if omegaj[i] != 0:
      newW[i,:] = W[i,:]
  return newW


def updateH(A,W,regParam,omega,m,n,k):
  '''Gets the new H matrix by using the formula, subbing H for W'''
  H = ctf.zeros((n,k))
  identity = ctf.eye(k)
  for i in range(n):
    newW = getNewWCTF(omega[:,i],W,m,n,k)
    newMat = ctf.zeros((k,k))
    newMat.i("xy") << newW.i("bx")*newW.i("by")
    newMat.i("xy") << regParam*identity.i("xy")
    inverseFirstPart = CTFinverse(newMat)
    temp = ctf.zeros((k))
    temp.i("j") << inverseFirstPart.i("ja")*W.i("ba")*A[:,i].i("b")
    H[i] = temp
  return H


def updateOmega(m,n,sparsity):
  '''Gets a random subset of rows for each H,W iteration'''
  Actf = ctf.tensor((m,n),sp=True)
  Actf.fill_sp_random(0,1,sparsity)
  omegactf = ((Actf > 0)*ctf.astensor(1.))
  return omegactf
  

def getIndexFromOmega(omega,m,n):
  '''Gets a random i,j contained in Ω[0...i-1,0...j-1]'''
  #TODO: Better sampling instead of MC.
  i = random.randint(0,m-1)
  j = random.randint(0,n-1)
  while omega[i][j] != 1:
    i = random.randint(0,m-1)
    j = random.randint(0,n-1)
  return (i,j)
  
  
def getDenseMtx(m,n,minBound,maxBound):
  '''Returns a dense matrix with (probably) no zeros'''
  X = ctf.tensor((m,n))
  X.fill_random(minBound,maxBound)
  return X


def getALSCtf(A,W,H,regParam,omega,m,n,k):
  """Same thing as above, but CTF"""
  prev = 10000000
  while prev - ((A - W@H.T())*omega).norm2()+(W.norm2()+H.norm2())*regParam > .001:
    prev = ((A - W@H.T())).norm2()+(W.norm2()+H.norm2())*regParam
    W = updateW(A,H,regParam,omega,m,n,k)
    H = updateH(A,W,regParam,omega,m,n,k)
    print(prev, ((A - W@H.T())).norm2()+(W.norm2()+H.norm2())*regParam)
  return W,H
    

def getSGDCtf(A,W,H,learningRate,regParam,width,convNo):
  """Updates the regularization parameter. Currently manual sgd, no CTF parallelization."""
  m = len(W.to_nparray())
  n = len(H.to_nparray())
  k = len(W.to_nparray()[0])
  lowerBound = 0
  stepRange  = 1
  percChange = .01
  result = (A - W@H.T()).norm2()+(W.norm2()+H.norm2())*regParam
  before = 0
  it = 0
  while abs(before - result) > convNo:
    it += 1
    print("Iteration:",it,"Before:",before,"Res:",result)
    i = random.randint(0,m-1)
    j = random.randint(0,n-1)
    R = A[i][j] - (W[i].T())@H[j]
    W[i] = W[i] - learningRate*(regParam*W[i] - R*H[j])
    H[j] = H[j] - learningRate*(regParam*H[j] - R*W[i])
    before = result
    result = (A - W@H.T()).norm2()+(W.norm2()+H.norm2())*regParam
  print("Finished with before:", before,"after:",result)


def getSGDSparse(A,W,H,omega,learningRate,regParam,width,convNo):
  """Does SGD. However, randomly selects one index from Ω, which is more sparse than above"""
  m = len(W.to_nparray())
  n = len(H.to_nparray())
  k = len(W.to_nparray()[0])
  lowerBound = 0
  stepRange  = 1
  percChange = .01
  result = (A - W@H.T()).norm2()+(W.norm2()+H.norm2())*regParam
  before = 0
  it = 0
  while abs(before - result) > convNo:
    it += 1
    print("Iteration (sparse):",it,"Before:",before,"Res:",result)
    i,j = getIndexFromOmega(omega,m,n)
    R = A[i][j] - (W[i].T())@H[j]
    W[i] = W[i] - learningRate*(regParam*W[i] - R*H[j])
    H[j] = H[j] - learningRate*(regParam*H[j] - R*W[i])
    before = result
    result = ((A - W@H.T())*omega).norm2()+(W.norm2()+H.norm2())*regParam
  print("Finished with before:", before,"after:",result)


def main():
  """Starts the program"""
  ut = UnitTests()
  ut.runAllTests()
  m = 40
  n = 30
  k = 8
  sparsity = .2
  learningRate = .1
  regParamALS = 2
  regParamSGD = .1
  width = 1 #how many random i,j we're going to get
  convNo = .00002 # When we'll stop converging
  
  X = getDenseMtx(m,k,0,1)
  Y = getDenseMtx(n,k,0,1)

  A = ctf.tensor((m,n))
  B = ctf.tensor((m,n),sp=True)
  B.fill_sp_random(1,1,1)
  A = (X@Y.T())
  
  X += crandom.random((m,k))*.001
  Y += crandom.random((n,k))*.001
  
  H = getDenseMtx(n,k,0,1)
  W = getDenseMtx(m,k,0,1)
  
  omega = updateOmega(m,n,sparsity)
  
  '''
  t = time.time()
  getALSCtf(A,X,Y,regParamALS,omega,m,n,k)
  print("Done with ALS! Time = ",np.round_(time.time()-t,4))
  '''
  
  t = time.time()
  #getSGDCtf(A,X,Y,learningRate,regParamSGD,width,convNo)
  dense_time = np.round_(time.time()-t,4)
  print(dense_time," seconds to convergence")
  
  X = getDenseMtx(m,k,0,1)
  Y = getDenseMtx(n,k,0,1)
  A = (X@Y.T())
  X = crandom.random((m,k))
  Y = crandom.random((n,k))
  A = A * omega
  print(A)
  t = time.time()
  getSGDSparse(A,X,Y,omega,learningRate,regParamSGD,width,convNo)
  sparse_time = np.round_(time.time()-t,4)
  print(sparse_time,"seconds to convergence for sparse, vs.", dense_time, "for dense")
  
  
main()
