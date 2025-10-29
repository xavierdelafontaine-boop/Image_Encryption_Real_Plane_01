function CPMcuan=CPMfun(NFnew,delta_row,delta_col,xo,yo,alfNew,ttte,kapa) %Creacion Mascara Caotica
alf=double(alfNew);
K=kapa;
transt=ttte; 
NNg=delta_col*delta_row+transt;
vec1=double(zeros(1,NNg));
vec2=double(zeros(1,NNg));
  xn=double(xo);
  yn=double(yo);
for n = 1:NNg,
    vec1(1,n)=xn*cos(alf)-(yn+K*sin(xn))*sin(alf);
    vec2(1,n)=xn*sin(alf)+(yn+K*sin(xn))*cos(alf);    
    xn=vec1(1,n);
    yn=vec2(1,n);  
end
pixn=zeros(delta_row,delta_col);
nk=1;
for xx=1:delta_row,
    for yy=1:delta_col,        
        pixn(xx,yy)=(mod(2*vec2(1,nk+transt),2*pi))./(2*pi);
        nk=nk+1;       
    end
end
CPM1=pixn;  % (cy-delta:cy,cx-delta:cx+delta);

for xx=1:delta_row,
    for yy=1:delta_col,        
        if CPM1(xx,yy) < 0.00392,
           CPM1(xx,yy) = 0.00392;
        end     
    end
end
CPMcuan=floor(CPM1.*NFnew);%./NFnew;% Fase discretizada
return