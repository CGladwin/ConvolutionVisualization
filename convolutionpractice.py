import torch
import torch.nn as nn
import pandas as pd
import numpy as np

mata = torch.tensor([[1,1],[1,1]])
matb = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
# print("matb",matb)
def convolvemanual(mata,matb,stride=[1,1],padding=0):
    hor = mata.shape[1]
    ver = mata.shape[0]
    
    padfunc = nn.ZeroPad2d(padding)
    matb=padfunc(matb)
    
    hor_end = matb.shape[0]
    ver_end = matb.shape[1]
    # print("horend",hor_end)
    output = torch.zeros(ver_end,hor_end)
    v_out=0
    h_out=0
    print("after padding, matb is:\n",matb)
    for i in range(0,ver_end-(ver-1),stride[0]):
        for j in range(0,hor_end-(hor-1),stride[1]):
            patch = matb[i:i+ver,j:j+hor]
            out_elem = (patch * mata)
            output[int(i/stride[0]),int(j/stride[1])] = out_elem.sum()
            if(i==0):
                h_out+=1
            print("patch is:\n",patch)
            print("\nresult of multiplying by kernel:\n",out_elem)
            print("\nsum of these values:\n",out_elem.sum().item(),"\n")        
        v_out+=1
    # ends = outcount**0.5
    return output[:v_out,:h_out]
convolvemanual(mata,matb)
