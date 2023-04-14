% The function reads n words from row iRow of matrix m and returns them in
% vector a together with the new row number iRowNew, i.e. the row where the
% last word was read.
%
%
 function [a, iRowNew] = extractNwords(n, iRow, m)
  
  k = 1; 
  iRowNew = iRow;
  for ii = 1:fix(n/6) %read lines with 6 words each
      for jj = 1:6
          a(k) = m(iRowNew,jj);
          k = k + 1;
      end
      iRowNew = iRowNew + 1;
  end
  
  if (n - fix(n/6)*6) == 0
     iRowNew = iRowNew - 1;
  end
  
  for jj = 1:(n-fix(n/6)*6) %read the last line with less than 6 words
      a(k) = m(iRowNew,jj); 
      k = k + 1;
  end
 end