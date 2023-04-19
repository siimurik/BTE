clear
m = importdata('B_010_mat.CSV',';'); % load CSV file into matrix m

iTemp = 4;

%%sigC = extract_mf3(102, iTemp, m);
mt = 16;
ntt = iTemp;
%nTemp = 4;

%for iTemp = 1:nTemp
%    [ifrom2, ito2, sig2] = extract_mf6(16, iTemp, m);
%end

temp = [293.60,	600, 900, 1200];

nRow = size(m,1); % number of rows
nTemp = 0; % number of temperatures
for iRow = 1:nRow
    if m(iRow,8) == 1 && m(iRow,9) == 451 && m(iRow,10) == 2
        nTemp = nTemp + 1; % number of temperatures
        temp(nTemp) = m(iRow,1); % vector of temperatures
    end
end

for iTemp = 1:nTemp
    [ifromE, itoE, sigE] = extract_mf6(2, iTemp, m);
end

sigE