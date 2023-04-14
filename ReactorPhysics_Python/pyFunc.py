import os
import numpy as np
import pandas as pd

# Read the CSV data file into a Pandas DataFrame object
data = pd.read_csv('B_010.CSV', delimiter=';', header=None)
df = data.replace(r'^\s*$', np.nan, regex=True).astype(float) # df contains empty cells, 

# Print the updated DataFrame
#print(df)

def extractNwords(n, iRow, m):
    k = 1
    iRowNew = iRow
    a = np.empty(n, dtype=float)
    for ii in range(int(n/6)):  # read lines with 6 words each
        for jj in range(6):
            a[k-1] = m[iRowNew-1][jj]
            k += 1
        iRowNew += 1
    
    if (n - int(n/6)*6) == 0:
        iRowNew -= 1
    
    for jj in range((n-int(n/6)*6)):  # read the last line with less than 6 words
        a[k-1] = m[iRowNew-1][jj]
        k += 1
    
    return a, iRowNew

def extract_mf6(mt, ntt, m):
    iRow = 0  # row number
    nTemp = 0  # number of temperatures
    ifrom = 0  # index of group 'from'
    ito = 0  # index of group 'to'
    sig = []  # list of sigma values
    
    while m[iRow, 6] != -1:  # up to the end
        if m[iRow, 7] == 6 and m[iRow, 8] == mt:  # find the row with mf=6 & mt
            if m[iRow, 9] == 1:  # this is the first line of mf=6 & mt: initialize
                nonz = 0  # number of nonzeros
                nLgn = m[iRow, 2]  # number of Legendre components
                nSig0 = m[iRow, 3]  # number of sigma-zeros

                iRow += 1
                nTemp += 1  # temperature index
            ng2 = m[iRow, 2]  # number of secondary positions
            ig2lo = m[iRow, 3]  # index to lowest nonzero group
            nw = m[iRow, 4]  # number of words to be read
            ig = m[iRow, 5]  # current group index

            iRow += 1     
            a, iRowNew = extractNwords(nw, iRow, m)  # extract nw words in vector a
            iRow = iRowNew

            if nTemp == ntt:
                k = nLgn*nSig0  # the first nLgn*nSig0 words are flux -- skip.
                for iTo in range(ig2lo, ig2lo+ng2-1):
                    nonz += 1
                    ifrom.append(ig)
                    ito.append(iTo)
                    for iSig0 in range(nSig0):
                        sig_iLgn_iSig0 = []
                        for iLgn in range(nLgn):
                            k += 1
                            sig_iLgn_iSig0.append(a[k])
                        sig.append(sig_iLgn_iSig0)
                        if nLgn == 1:
                            sig.append([0])
                            sig.append([0])
            
        iRow += 1
          
    if nTemp == 0:
        sig = 0
        
    return ifrom, ito, sig


mt = 102  # set the value of mt
ntt = 1  # set the value of ntt
m = df.to_numpy()  # convert DataFrame to numpy array
#m = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]])
ifrom, ito, sig = extract_mf6(mt, ntt, m)  # apply the function to the numpy array
print(sig)