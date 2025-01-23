import numpy as np 

b = (5, 6, 7, 8)
b = np.array(b)
a  = (1, 2, 3, 4, b)

a_copy = []
for x in a:
    if isinstance(x, int): 
        a_copy.append(x)
    else: 
        for y in x: 
            a_copy.append(y)
a_copy = [a_copy]
c = np.concatenate(a_copy)
print(c)