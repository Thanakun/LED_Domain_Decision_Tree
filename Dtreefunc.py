#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy import linalg
import numpy as np
import math

def entropy_class(class_counts):
    """
    Extected information 
    Info(D)=-sum(pi)log2(pi)
    """
    # if x==0 or y==0:
    #     return (0)
    # else:
    #     out=(-x/(x+y))*math.log((x/(x+y)),2)+(-y/(x+y))*math.log((y/(x+y)),2)
    #     return(out)
    total = sum(class_counts)
    entropy = 0
    for count in class_counts:
        p = count / total
        entropy += -p * math.log2(p)
    return entropy

def entropy(x,y):
    """
    Extected information 
    Info(D)=-sum(pi)log2(pi)
    """ 
    if x==0 or y==0:
        return (0)
    else:
        out=(-x/(x+y))*math.log((x/(x+y)),2)+(-y/(x+y))*math.log((y/(x+y)),2)
        return(out)



def inforD(m,n): 
    """
    m is array of class in each attb group by domain
    n is array of entropy
    """
    c=len(m)
    #print(" size m is ",c)
    out=0
    i=0
    for i in range (c):
       # print("m[i] is", m[i])
        #print("sum(m) is ", sum(m))
        out +=(m[i]/sum(m))*n[i]
        #print("out is ",out)
    return(out)
    

