clear
m = importdata('B_010_mat.CSV',';'); % load CSV file into matrix m

iTemp = 4;

%%sigC = extract_mf3(102, iTemp, m);
mt = 16;
ntt = iTemp;
nTemp = 4;

for iTemp = 1:nTemp
    [ifrom2, ito2, sig2] = extract_mf6(16, iTemp, m);
end
