 function [ifrom,ito,sig] = extract_mf6(mt, ntt, m)
 
  iRow = 1; % row number
  nTemp = 0; % number of temperatures
  ifrom = 0; % index of group 'from'
  ito = 0; % index of group 'to'
  
  while m(iRow,7) ~= -1 % up to the end
      if m(iRow,8) == 6 && m(iRow,9) == mt % find the row with mf=6 & mt
         if m(iRow,10) == 1  % this is the first line of mf=6 & mt: initialize
            nonz = 0; % number of nonzeros
            nLgn = m(iRow,3); % number of Legendre components
            nSig0 = m(iRow,4); % number of sigma-zeros

            iRow = iRow + 1;
            nTemp = nTemp + 1; % temperature index
         end
         ng2 = m(iRow,3); % number of secondary positions
         ig2lo = m(iRow,4); % index to lowest nonzero group
         nw = m(iRow,5); % number of words to be read
         ig = m(iRow,6); % current group index

         iRow = iRow + 1;     
         [a,iRowNew] = extractNwords(nw,iRow,m); % extract nw words in vector a
         iRow = iRowNew;

         if nTemp == ntt
            k = nLgn*nSig0; % the first nLgn*nSig0 words are flux -- skip.
            for iTo = ig2lo : ig2lo+ng2-2
                nonz = nonz + 1;
                ifrom(nonz) = ig;
                ito(nonz) = iTo;
                for iSig0 = 1:nSig0
                    for iLgn = 1:nLgn
                        k = k + 1;
                        sig{iLgn,iSig0}(nonz) = a(k);
                    end
                    if nLgn == 1
                       sig{1+1,iSig0}(nonz) = 0;
                       sig{1+2,iSig0}(nonz) = 0;
                    end
                end
            end    
         end
      end
      iRow = iRow + 1;
  end
  if nTemp == 0
      sig = 0;
  end
end