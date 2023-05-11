% The function searches matrix m for cross sections sig from file mf=3 for
% reaction mt and temperature ntt and and returns sig(ng,nSig0), where ng
% is the number of energy groups and nSig0 is the the number of
% sigma-zeros.
%
%
function sig = extract_mf3(mt, ntt, m)

  nRow = size(m,1); % number of rows
  nTemp = 0; % number of temperatures
  iRowFound = 0;
  for iRow = 1:nRow
      if m(iRow,8) == 3 && m(iRow,9) == mt && m(iRow,10) == 1 % find the row with mf=3 and required mt
         nTemp = nTemp + 1; % number of temperatures
         if nTemp == ntt
            iRowFound = iRow+1;
            break;
         end
      end
  end

  if iRowFound > 0 % there is mf=3 and required mt for this isotope
     nSig0 = m(iRowFound-1,4); % number of sigma-zeros
     nLgn = m(iRowFound-1,3); % number of Legendre components
     iRow = iRowFound + 1;
     while m(iRow,8) == 3 && m(iRow,9) == mt
         ig = m(iRow-1,6);
         [a, iRowNew] = extractNwords(nSig0*nLgn*2, iRow, m); 
         sig(1:nSig0,ig) = a(nSig0*nLgn+(1:nSig0));
         iRow = iRowNew + 2;
     end
  else
     sig = 0;
  end
end