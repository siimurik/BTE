import os
import numpy as np
import scipy
#=========================================================================
def extractNwords(n, iRow, m):
    k = 1
    iRowNew = iRow
    a = []
    for ii in range(int(n/6)):  # read lines with 6 words each
        for jj in range(6):
            a.append(m[iRowNew-1][jj])
            k += 1
        iRowNew += 1
    
    if (n - int(n/6)*6) == 0:
        iRowNew -= 1
    
    for jj in range((n-int(n/6)*6)):  # read the last line with less than 6 words
        a.append(m[iRowNew-1][jj])
        k += 1
    
    return a, iRowNew
#=========================================================================
def extract_mf3(mt, ntt, m):
    nRow = m.shape[0]
    nTemp = 0
    iRowFound = 0

    for iRow in range(nRow):
        if m[iRow, 7] == 3 and m[iRow, 8] == mt and m[iRow, 9] == 1:
            nTemp += 1
            if nTemp == ntt:
                iRowFound = iRow + 1
                break

    if iRowFound > 0:
        nSig0 = int(m[iRowFound-1, 3]) # cast nSig0 as integer
        nLgn = m[iRowFound-1, 2]
        iRow = iRowFound + 1
        sig = np.zeros((nSig0, m.shape[1]-1))
        while m[iRow, 7] == 3 and m[iRow, 8] == mt:
            ig = m[iRow-1, 5]
            a, iRowNew = extractNwords(nSig0*nLgn*2, iRow, m)
            sig[0:nSig0, ig-1] = a[nSig0*nLgn:(nSig0*nLgn)+nSig0]
            iRow = iRowNew + 2
        return sig
    else:
        return 0

"""
def extract_mf3(mt, ntt, m):
    nRow = m.shape[0]
    nTemp = 0
    iRowFound = 0

    for iRow in range(nRow):
        if m[iRow, 7] == 3 and m[iRow, 8] == mt and m[iRow, 9] == 1:
            nTemp += 1
            if nTemp == ntt:
                iRowFound = iRow + 1
                break

    if iRowFound > 0:
        nSig0 = m[iRowFound-1, 3]
        nLgn = m[iRowFound-1, 2]
        iRow = iRowFound + 1
        sig = np.zeros((nSig0, m.shape[1]-1))
        while m[iRow, 7] == 3 and m[iRow, 8] == mt:
            ig = m[iRow-1, 5]
            a, iRowNew = extractNwords(nSig0*nLgn*2, iRow, m)
            sig[0:nSig0, ig-1] = a[nSig0*nLgn:(nSig0*nLgn)+nSig0]
            iRow = iRowNew + 2
        return sig
    else:
        return 0
        """
#=========================================================================
def extract_mf6(mt, ntt, m):
    iRow = 0
    nTemp = 0
    ifrom = []
    ito = []
    sig = []

    while m[iRow, 6] != -1:
        if m[iRow, 7] == 6 and m[iRow, 8] == mt:
            if m[iRow, 9] == 1:
                nonz = 0
                nLgn = m[iRow, 2]
                nSig0 = m[iRow, 3]
                iRow += 1
                nTemp += 1
            ng2 = m[iRow, 2]
            ig2lo = m[iRow, 3]
            nw = m[iRow, 4]
            ig = m[iRow, 5]
            iRow += 1
            a, iRowNew = extractNwords(nw, iRow, m)
            iRow = iRowNew

            if nTemp == ntt:
                k = nLgn*nSig0
                for iTo in range(ig2lo-1, ig2lo+ng2-1):
                    nonz += 1
                    ifrom.append(ig)
                    ito.append(iTo)
                    sig.append([[] for _ in range(nSig0)])

                    for iSig0 in range(nSig0):
                        for iLgn in range(nLgn):
                            k += 1
                            sig[nonz-1][iSig0].append(a[k-1])
                        if nLgn == 1:
                            sig[nonz-1][1].append(0)
                            sig[nonz-1][2].append(0)
        iRow += 1

    if nTemp == 0:
        return 0
    else:
        return ifrom, ito, sig

#=========================================================================
def convertCSVtoCSV():
    filesCSV = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.CSV')]
    for fileCSV in filesCSV:
        nameOnly = os.path.splitext(fileCSV)[0]
        print(f'Import data for {nameOnly} and check if CSV files for all temperatures are already available. ')
        with open(fileCSV, 'r') as f:
            m = []
            for line in f:
                try:
                    values = list(map(float, line.strip().split(';')))
                    m.append(values)
                except ValueError:
                    pass  # skip over lines that contain non-numeric values or empty strings
        m = np.array(m)
        print(m)
        nRow = m.shape[0]

        # Find number of temperatures and values of temperatures using mf=1 and mt=451
        nTemp = 0  # number of temperatures
        temp = []  # vector of temperatures
        for iRow in range(nRow):
            if m[iRow][7] == 1 and m[iRow][8] == 451 and m[iRow][9] == 2:
                nTemp += 1  # number of temperatures
                temp.append(m[iRow][0])  # vector of temperatures

        for iTemp in range(nTemp):  # loop over temperatures
            if temp[iTemp] < 1000:
                isoName = f'micro_{nameOnly}__{round(temp[iTemp])}K'
            else:
                isoName = f'micro_{nameOnly}_{round(temp[iTemp])}K'

            if not os.path.isfile(f'{isoName}.CSV'):  # if the corresponding CSV file does not exist
                with open(f'{isoName}.CSV', 'w') as fd:
                    # Make a header for the file to be created with important parameters for
                    # which the macroscopic cross sections were generated
                    fd.write('%% ---------------------------------------------------------\n')
                    fd.write('%% Python-based Reactor Physics Simulation\n')
                    fd.write('%% ---------------------------------------------------------\n')
                    fd.write('%% Author: Siim Erik Pugal, 2023.\n')
                    fd.write('%%\n')
                    fd.write(f'%% Microscopic cross sections for {nameOnly} at {round(temp[iTemp])} K in 421 energy group structure\n')
                    fd.write('aw;ng;eg;sz;siga;sigb;sigc;sigd;nubar;chi;sigt;sigf;sigg;n2n\n')

                    # atomic weight (amu)
                    #fd.write(f's.aw = {m[1,2]*1.008664916:.6e};\n\n')
                    fd.write(f's.aw = {m[1][2]*1.008664916:.6e};\n\n')

                    ng = 421
                    # number of energy groups
                    fd.write(f'ng = {ng};\n\n')

                    nSig0 = int(m[1][4])
                    a = extractNwords(1+nSig0+(ng+1), 4, m)
                    #print(f'a = {a}')
                    #print(f'nSig0 = {nSig0}')

                    # energy group boundaries (eV)
                    fd.write('s.eg = [')
                    #fd.write(' '.join([f'{a[2+nSig0+i]:.5e}' for i in range(ng+1)]))
                    fd.write(' '.join([f'{a[2+nSig0+i]:.5e}' for i in range(ng+1) if len(a) > 2+nSig0+i]))
                    fd.write('];\n\n')

                    # sigma-zeros (b)
                    fd.write('s.sig0 = [')
                    #fd.write(' '.join([f'{a[2+i]:.5e}' for i in range(nSig0)]))
                    fd.write(' '.join([f'{a[2+i]:.5e}' for i in range(nSig0) if len(a) > 2+i]))
                    fd.write('];\n')
                    fd.write(f'nSig0 = {nSig0};\n\n')

                    # temperature (K)
                    fd.write(f's.temp = {temp[iTemp]:.2f};\n\n')

                    # (n,gamma)
                    # Convert CSV to .m: mf=3 mt=102 radiative capture
                    print(f'Convert {nameOnly}.CSV to {isoName}.CSV: mf=3 mt=102 radiative capture\n')
                    sigC = extract_mf3(102, iTemp, m) # Extract mf=3 mt=102 (radiative capture cross sections)
                    print(f"sigC = {sigC}")
                    nSig0C = sigC.shape[0]
                    print(f"nSig0C = {nSig0C}")
                    nSig0C = len(sigC)
                    fd.write(f'%% radiative capture cross section (b) for {nSig0C} sigma-zero(s)\n')
                    for iSig0 in range(nSig0C):
                        fd.write(f's.sigC({iSig0+1},:)=[')
                        fd.write(';'.join(f'{x:.6e}' for x in sigC[iSig0,:ng]))
                        fd.write('];\n')
                    if nSig0C == 1 and nSig0 > 1:
                        fd.write('sigC[1:nSig0,:] = sigC[0,:];\n')
                    fd.write('\n')

                    # (n,alfa)
                    print(f'Convert {nameOnly}.CSV to {isoName}.CSV: mf=3 mt=107 (n,alfa)\n')
                    sigL = extract_mf3(107, iTemp, m)  # Extract mf=3 mt=107 (production of an alfa particle)
                    if sigL is None:
                        sigL = np.zeros((nSig0, ng))
                        fd.write('%% (n,alfa) cross section (b)\n')
                        fd.write('s.sigL = np.zeros((nSig0, ng))\n\n')
                    else:
                        nSig0L = sigL.shape[0]
                        fd.write(f'%% (n,alfa) cross sections (b) for {nSig0L:2d} sigma-zero(s)\n')
                        for iSig0 in range(nSig0L):
                            fd.write(f's.sigL({iSig0+1:2d},:) = [{"; ".join([f"{x:13.6e}" for x in sigL[iSig0, :]])}];\n')
                        if nSig0L == 1 and nSig0 > 1:
                            fd.write(f's.sigL = np.tile(s.sigL, (nSig0, 1))\n')
                            sigL = np.tile(sigL, (nSig0, 1))
                        fd.write('\n')

                        # (n,2n)
                        ifrom2, ito2, sig2 = extract_mf6(16, iTemp, m)  # Extract mf=6 mt=16 ((n,2n) matrix)
                        fd.write('%% (n,2n) matrix for 1 Legendre components\n')
                        if ifrom2[0] == 0:
                            isn2n = 0
                            fd.write('s.sig2=zeros(ng,ng);\n\n')
                        else:
                            isn2n = 1
                            fd.write(f'%% Convert {nameOnly}.CSV to {isoName}.CSV: mf=6 mt=16 (n,2n) reaction\n')
                            fd.write('ifrom=[')
                            fd.write(''.join(f'{str(x).ljust(13)}' for x in ifrom2))
                            fd.write('];\n')
                            fd.write('ito  =[')
                            fd.write(''.join(f'{str(x).ljust(13)}' for x in ito2))
                            fd.write('];\n')
                            fd.write('s.sig2=sparse(ifrom,ito,')
                            fd.write(' '.join(f'{x:13.6e}' for x in sig2[1][0]))
                            fd.write(',ng,ng);\n')
                            fd.write('\n')


                    # (n,n')
                    igThresh = 95 # last group of thermal energy (e = 4 eV)

                    print('Convert %s.CSV to %s.CSV: mf=6 mt=2 elastic scattering' % (nameOnly, isoName))
                    ifromE, itoE, sigE = extract_mf6(2, iTemp, m) # Extract mf=6 mt=2 (elastic scattering matrix)
                    nLgn = sigE.shape[0] - 1
                    sigS = []
                    for jLgn in range(nLgn + 1):
                        for iSig0 in range(nSig0):
                            for ii in range(len(ifromE)):
                                if ifromE[ii] <= igThresh:
                                    sigE[jLgn + 1, iSig0, ii] = 0
                            sigS.append(scipy.sparse.csr_matrix((sigE[jLgn + 1, iSig0] + 1e-30, (ifromE, itoE)), shape=(ng, ng)))

                    for ii in range(51, 92):
                        ifromI, itoI, sigI = extract_mf6(ii, iTemp, m) # Extract mf=6 mt=51 ... 91 (inelastic scattering matrix)
                        if ifromI[0] > 0:
                            print('Convert %s.CSV to %s.CSV: mf=6 mt=%2i inelastic scattering' % (nameOnly, isoName, ii))
                            nLgn = sigI.shape[0] - 1
                            for jLgn in range(nLgn + 1):
                                for iSig0 in range(nSig0):
                                    sigS[jLgn + 1, iSig0] += scipy.sparse.csr_matrix((sigI[jLgn + 1, 0] + 1e-30, (ifromI, itoI)), shape=(ng, ng))

                    if isoName[0:11] == 'micro_H_001':
                        print('Convert {}.CSV to {}.CSV: mf=6 mt=222 thermal scattering for hydrogen binded in water'.format(nameOnly, isoName))
                        ifromI, itoI, sigI = extract_mf6(222, iTemp, m) # Extract mf=6 mt=222 thermal scattering for hydrogen binded in water
                    else:
                        print('Convert {}.CSV to {}.CSV: mf=6 mt=221 free gas thermal scattering'.format(nameOnly, isoName))
                        ifromI, itoI, sigI = extract_mf6(221, iTemp, m) # Extract mf=6 mt=221 free gas thermal scattering

                    nLgn = sigI.shape[0] - 1
                    for jLgn in range(nLgn + 1):
                        for iSig0 in range(nSig0):
                            sigS[1+jLgn,iSig0] += scipy.sparse.csr_matrix((sigI[1+jLgn, 0] + 1e-30, (ifromI, itoI)), shape=(ng, ng))

                    fd.write('%% scattering matrix for 3 Legendre components for {} sigma-zeros\n'.format(nSig0))
                    for jLgn in range(3):
                        notYetPrinted = True
                        for iSig0 in range(nSig0):
                            ifrom, ito, sigS_ = scipy.sparse.find(sigS[1+jLgn,iSig0]) # Find indices and values of nonzero elements
                            # ifrom and ito are printed only once
                            if notYetPrinted:
                                fd.write('ifrom=[' + ' '*26 + ''.join(['{:13d} '.format(x) for x in ifrom]) + '];\n')
                                fd.write('ito  =[' + ' '*26 + ''.join(['{:13d} '.format(x) for x in ito]) + '];\n')
                                notYetPrinted = False
                            fd.write('s.sigS{{1+{}, {}}}=sparse(ifrom,ito,['.format(jLgn, iSig0))
                            fd.write(''.join(['{:13.6e} '.format(x) for x in sigS_]))
                            fd.write('],ng,ng);\n')
                        fd.write('\n')

                    # (n,fis)
                    sigF = extract_mf3(18, iTemp, m) # Extract mf=3 mt=18 (fission cross sections)

                    if sigF.shape == (1,):
                        sigF = np.zeros((nSig0, ng))
                        fd.write('s.fissile = 0;\n\n')

                        fd.write('%% fission cross sections (b)\n')
                        fd.write('s.sigF = zeros(nSig0, ng);\n\n')

                        fd.write('%% nubar\n')
                        fd.write('s.nubar = zeros(nSig0, ng);\n\n')

                        fd.write('%% fission spectrum\n')
                        fd.write('s.chi = zeros(nSig0, ng);\n\n')
                    else:
                        print('Convert %s.CSV to %s.m: mf=3 mt=18 fission\n' % (nameOnly, isoName))
                        fd.write('s.fissile = 1;\n\n')

                        nSig0F = sigF.shape[0]
                        fd.write('%% fission cross sections (b) for %2i sigma-zero(s)\n' % nSig0F)
                        for iSig0 in range(nSig0F):
                            fd.write('s.sigF(%2i,:) = [' % (iSig0 + 1))
                            fd.write('%13.6e ' % tuple(sigF[iSig0,:]))
                            fd.write('];\n')
                        fd.write('\n')

                        nubar = extract_mf3(452, iTemp, m) # Extract mf=3 mt=452 (total nubar)
                        print('Convert %s.CSV to %s.m: mf=3 mt=452 total nubar\n' % (nameOnly, isoName))
                        nSig0nu = nubar.shape[0]
                        fd.write('%% total nubar for %2i sigma-zero(s)\n' % nSig0nu)
                        for iSig0 in range(nSig0nu):
                            fd.write('s.nubar(%2i,:) = [' % (iSig0 + 1))
                            fd.write('%13.6e ' % tuple(nubar[iSig0,:]))
                            fd.write('];\n')
                        fd.write('\n')

                        # chi
                        print(f'Convert {nameOnly}.CSV to {isoName}.m: mf=6 mt=18 fission spectrum')
                        iRow = 0
                        while m[iRow, 7] != 6 or m[iRow, 8] != 18:  # find fission spectrum
                            iRow += 1

                        iRow += 1
                        ig2lo = m[iRow, 3]  # index to lowest nonzero group
                        nw = m[iRow, 4]  # number of words to be read
                        iRow += 1

                        a = np.zeros(nw, dtype=np.float64)
                        for iii in range(nw):
                            a[iii] = m[iRow, iii]
                            
                        chi = np.zeros(ng, dtype=np.float64)
                        for iii in range(ig2lo):
                            chi[iii] = 0.0

                        for iii in range(nw):
                            chi[iii + ig2lo - 1] = a[iii]

                        fd.write('%% fission spectrum\n')
                        fd.write('s.chi=[')
                        fd.write(' '.join(f'{x/sum(chi):13.6e}' for x in chi))
                        fd.write('];\n\n')

                        # total
                        # Calculate total cross sections (note that mf=3 mt=1 does not include upscatters).
                        fd.write("%% total cross sections (b) for %2i sigma-zeros\n" % nSig0)
                        sigT = np.zeros((nSig0, ng))
                        for iSig0 in range(nSig0):
                            sigT[iSig0,:] = sigC[iSig0,:] + sigF[iSig0,:] + sigL[iSig0,:] + np.sum(sigS[1+0,iSig0],axis=0)
                            if isn2n:
                                sigT[iSig0,:] = sigT[iSig0,:] + np.sum(sig2[1+0,1],axis=0)
                            fd.write("s.sigT(%2i,:)=[" % iSig0)
                            fd.write("%13.6e " % tuple(sigT[iSig0,:]))
                            fd.write("];\n")
                        fd.write('\n')

if __name__ == '__main__':
    convertCSVtoCSV()
#=========================================================================


