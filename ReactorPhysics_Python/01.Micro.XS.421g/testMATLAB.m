clear
m = importdata('NA023.CSV',';');
mm = importdata('U_235.CSV',';');
ng = 421;
mt = 16;
ntt = 3;

[ifrom,ito,sig] = extract_mf6(mt, ntt, mm);