Python 3.8.0 (tags/v3.8.0:fa919fd, Oct 14 2019, 19:21:23) [MSC v.1916 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
============= RESTART: C:\Users\reddy\Desktop\ML Lab Assignment.py =============
>>> #Roll No: 19H61A3527
>>> #Date: 24/08/2021
>>> 
>>> #Variables and Arithmetic
>>> 
>>> a=3
>>> a
3
>>> a+2
5
>>> a=7.1
>>> a+2
9.1
>>> b=10
>>> a+b*2
27.1
>>> mike=17
>>> mike*b+1
171
>>> mike**b # ' ** ' in python denotes power
2015993900449
>>> print(mike*b+1)
171
>>> print(mike**b)
2015993900449
>>> b=0
>>> print(mike*b+1)
1
>>> print(mike**b)
1
>>> 
>>> #STRING VARIABLES
>>> greeting = 'Hello. My name is'
>>> Print(greeting)
Traceback (most recent call last):
  File "<pyshell#23>", line 1, in <module>
    Print(greeting)
NameError: name 'Print' is not defined
>>> print(greeting)
Hello. My name is
>>> name='Nakshatra'
>>> print(greeting name)
SyntaxError: invalid syntax
>>> print(greeting + name)
Hello. My name isNakshatra
>>> print(greeting + name + '!') #in string variables + is used for conacatination
Hello. My name isNakshatra!
>>> age = 10
>>> print('I am' + age + 'Years old.')
Traceback (most recent call last):
  File "<pyshell#30>", line 1, in <module>
    print('I am' + age + 'Years old.')
TypeError: can only concatenate str (not "int") to str
>>> 
>>> # python cannot concatenate integer with string
>>> #so in python a number is converted to string by using str()
>>> print('I am' + str(age) + 'Years old.')
I am10Years old.
>>> 
>>> #ASSIGNMENT
>>> print('My name is ' + name + ', I am now ' + age + ' years old, \ and in 6 years I\'ll be ' + ' years old.')#que
Traceback (most recent call last):
  File "<pyshell#37>", line 1, in <module>
    print('My name is ' + name + ', I am now ' + age + ' years old, \ and in 6 years I\'ll be ' + ' years old.')#que
TypeError: can only concatenate str (not "int") to str
>>> 
>>> #correct code
>>> print('My name is ' + name + ', I am now ' + str(age) + ' years old, and in 6 years I\'ll be ' + str(age+2) + ' years old.')
My name is Nakshatra, I am now 10 years old, and in 6 years I'll be 12 years old.
>>> 
>>> #The NUMPY MODULE
>>> 
>>> numlist = [1,2,3,4,4] #in python lists are denoted by []
>>> numlist
[1, 2, 3, 4, 4]
>>> abs(-5) #absolute value is distance of that particular number from 0
5
>>> sqrt(4)
Traceback (most recent call last):
  File "<pyshell#48>", line 1, in <module>
    sqrt(4)
NameError: name 'sqrt' is not defined
>>> 
>>> #Python doesnot have any SQUARE ROOT FUNCTION.
>>> 
>>> average(numlist)
Traceback (most recent call last):
  File "<pyshell#52>", line 1, in <module>
    average(numlist)
NameError: name 'average' is not defined
>>> mean(numlist)
Traceback (most recent call last):
  File "<pyshell#53>", line 1, in <module>
    mean(numlist)
NameError: name 'mean' is not defined
>>> 
>>> #Python doesnot have any function to compute the average in the base environment
>>> # IN PYTHON THERE ARE MODULES(also called as toolboxes) that allows us to process the with functions which are not allowed in the base environment
>>> 
>>> # importing a module
>>> import numpy as np
>>> mean(numlist)
Traceback (most recent call last):
  File "<pyshell#60>", line 1, in <module>
    mean(numlist)
NameError: name 'mean' is not defined
>>> #still it's showing an error because it should be declared with np.mean
>>> np.mean(numlist)
2.8
>>> np.sqrt(4)
2.0
>>> 
>>> # linspace function creates linearly spaced numbers between the 2 numbers
>>> np.linspace(1,10,7) # 1 - first number, 10 - second number, 7 denotes number of spaced numbers
array([ 1. ,  2.5,  4. ,  5.5,  7. ,  8.5, 10. ])
>>> # if you know what the function is called but you are not really sure what the inputs are or how the function works then you can click the cusrsor inside the function i.e. inside the paranthesis and then pressing SHIFT+TAB then that opens tiny window which is also known as HELP TEXT or DOC STRING which displays how to use that particular function
>>> np.linspace(1,10,12)
array([ 1.        ,  1.81818182,  2.63636364,  3.45454545,  4.27272727,
        5.09090909,  5.90909091,  6.72727273,  7.54545455,  8.36363636,
        9.18181818, 10.        ])
>>> funout = np.linspace(1,10,12)
>>> funout + 2
array([ 3.        ,  3.81818182,  4.63636364,  5.45454545,  6.27272727,
        7.09090909,  7.90909091,  8.72727273,  9.54545455, 10.36363636,
       11.18181818, 12.        ])
>>> 
>>> #ASSIGNMENT 2
>>> #part 1 - create a list of 15 numbers from 4 to 100
>>> #part 2 - round those numbers to the nearest integer, and store in the another variable
>>> #part 3 - print out #2
>>> #part 4 - print out the square root of each number in the list
>>> 
>>> #PART 1
>>> np.linspace(4,100,15)
array([  4.        ,  10.85714286,  17.71428571,  24.57142857,
        31.42857143,  38.28571429,  45.14285714,  52.        ,
        58.85714286,  65.71428571,  72.57142857,  79.42857143,
        86.28571429,  93.14285714, 100.        ])
>>> #PART 2
>>> arr = np.round(np.linspace(4,100,15))
>>> print(arr)
[  4.  11.  18.  25.  31.  38.  45.  52.  59.  66.  73.  79.  86.  93.
 100.]
>>> #PART 3
>>> arr1 = np.round(np.linspace(4,100,15),2)
>>> print(arr1)
[  4.    10.86  17.71  24.57  31.43  38.29  45.14  52.    58.86  65.71
  72.57  79.43  86.29  93.14 100.  ]
>>> #all the numbers in the list are rounded-off to 2 decimal points
>>> #PART 4
>>> arr=np.sqrt(arr)
>>> print('square root of list is ' + str(arr))
square root of list is [ 2.          3.31662479  4.24264069  5.          5.56776436  6.164414
  6.70820393  7.21110255  7.68114575  8.1240384   8.54400375  8.88819442
  9.2736185   9.64365076 10.        ]
>>> 
>>> # MATPLOTLIB MODULE
>>> import matplotlib.pyplot as plt #importing a matplotlib module
>>> import numpy as np
>>> plt.plot(1,3) #1 is x co-ordinate & 3 is y co-ordinate
[<matplotlib.lines.Line2D object at 0x0AAB7988>]
>>> plt.plot(1,3,'ko')
[<matplotlib.lines.Line2D object at 0x0AAB7BC8>]
>>> plt.show()
>>> #plt.show() is used display a graph
>>> #'ro' indicated red color & 'ko' indicates black color
>>> 
>>> x=np.arange(-9,10) #arange is used to plot multiple dots
>>> print(x)
[-9 -8 -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7  8  9]
>>> 
>>> y=x**2
>>> plt.plot(x,y,'r')
[<matplotlib.lines.Line2D object at 0x0153C718>]
>>> plt.show()
>>> plt.plot(x,y/2,'gs')
[<matplotlib.lines.Line2D object at 0x097B4C88>]
>>> plt.show()
>>> plt.show()
>>> plt.plot(x,y/2,'gs')
[<matplotlib.lines.Line2D object at 0x097F6190>]
>>> plt.show()
>>> 
>>> plt.plot([0,3],[-1,1])#creating a line
[<matplotlib.lines.Line2D object at 0x0982B490>]
>>> plt.show()
>>> plt.plot([0,3],[-1,1],label='first line') #if graph contains more than one line then to identify which line is which then we use label function
[<matplotlib.lines.Line2D object at 0x0A2957A8>]
>>> plt.plot([-2,0],[-4,1],label='second line')
[<matplotlib.lines.Line2D object at 0x0A295028>]
>>> plt.show()
>>> plt.plot([0,3],[-1,1],label='first line')
[<matplotlib.lines.Line2D object at 0x0978C268>]
>>> plt.plot([-2,0],[-4,1],label='second line')
[<matplotlib.lines.Line2D object at 0x0978C1C0>]
>>> plt.legend()
<matplotlib.legend.Legend object at 0x0978C5B0>
>>> plt.show()
>>> 
>>> #matrix
>>> M=np.random.randint(0,10,size=(4,5))#random generates any random numbers
>>> M
array([[1, 3, 2, 3, 3],
       [0, 1, 7, 5, 4],
       [1, 8, 2, 0, 2],
       [4, 9, 5, 5, 2]])
>>> M
array([[1, 3, 2, 3, 3],
       [0, 1, 7, 5, 4],
       [1, 8, 2, 0, 2],
       [4, 9, 5, 5, 2]])
>>> plt.imshow(M)
<matplotlib.image.AxesImage object at 0x0A2F35C8>
>>> plt.show()
>>> #plt.impshow() used to represent the matrix with colors
>>> plt.plot([0,4],[-0.5,3.5])
[<matplotlib.lines.Line2D object at 0x09698E98>]
>>> plt.imshow(M)
<matplotlib.image.AxesImage object at 0x096A3148>
>>> plt.show()
>>> #above code is assignment
>>> 
>>> #SCALAR AND VECTOR MULTIPLICATIONS
>>> #vector - an ordered list of numbers , order is important for a vector
>>> #Dimensionality - number of elements in a vector
>>> #therefor [1 2 3] is not equal to [2 3 1] since order is not same
>>> 
>>> #Geometric vectors(1D,2D,3D)
>>> #Scalar is a single number
>>> vec=[3,4,5,2] #since vector is a orderd list
>>> sca = 2#Scalar is a single number
>>> vec*sca
[3, 4, 5, 2, 3, 4, 5, 2]
>>> s=3
>>> vec*sca
[3, 4, 5, 2, 3, 4, 5, 2]
>>> vec=np.array([3,4,5,2])
>>> #np.array() is used to multiply a scalar to a vector
>>> sca = 3
>>> vec*sca
array([ 9, 12, 15,  6])
>>> 
>>> #the above vector vec is 4dimensonal, since we are not capable of 4dimensional we are creating a new vector of 2dimensional
>>> vec2d = np.array([1,2])
>>> s1=2
>>> s2=0.5
>>> s3=-1 #multiple scalars
>>> #indexing : vector name + paranthesis ie.,vec2d[1]...1 denotes the number which is in index numbered 1
>>> vec2d[0]
1
>>> plt.plot([0,vec2d])
TypeError: only size-1 arrays can be converted to Python scalars

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<pyshell#158>", line 1, in <module>
    plt.plot([0,vec2d])
  File "C:\Users\reddy\AppData\Local\Programs\Python\Python38-32\lib\site-packages\matplotlib\pyplot.py", line 2787, in plot
    return gca().plot(
  File "C:\Users\reddy\AppData\Local\Programs\Python\Python38-32\lib\site-packages\matplotlib\axes\_axes.py", line 1667, in plot
    self.add_line(line)
  File "C:\Users\reddy\AppData\Local\Programs\Python\Python38-32\lib\site-packages\matplotlib\axes\_base.py", line 1902, in add_line
    self._update_line_limits(line)
  File "C:\Users\reddy\AppData\Local\Programs\Python\Python38-32\lib\site-packages\matplotlib\axes\_base.py", line 1924, in _update_line_limits
    path = line.get_path()
  File "C:\Users\reddy\AppData\Local\Programs\Python\Python38-32\lib\site-packages\matplotlib\lines.py", line 1027, in get_path
    self.recache()
  File "C:\Users\reddy\AppData\Local\Programs\Python\Python38-32\lib\site-packages\matplotlib\lines.py", line 675, in recache
    y = _to_unmasked_float_array(yconv).ravel()
  File "C:\Users\reddy\AppData\Local\Programs\Python\Python38-32\lib\site-packages\matplotlib\cbook\__init__.py", line 1390, in _to_unmasked_float_array
    return np.asarray(x, float)
  File "C:\Users\reddy\AppData\Local\Programs\Python\Python38-32\lib\site-packages\numpy\core\_asarray.py", line 85, in asarray
    return array(a, dtype, copy=False, order=order)
ValueError: setting an array element with a sequence.
>>> plt.plot([0,vec2d[0]],[0,vec2d[1]],'bs-')
[<matplotlib.lines.Line2D object at 0x096CFCD0>]
>>> plt.show
<function show at 0x085CF778>
>>> plt.show()
>>> 
>>> plt.plot([0,vec2d[0]],[0,vec2d[1]],'bs-',label='v')
[<matplotlib.lines.Line2D object at 0x0A8A9C58>]
>>> plt.plot([0,s1*vec2d[0]],[0,s1*vec2d[1]],'ro-',label='v*s1')
[<matplotlib.lines.Line2D object at 0x0A8A9E38>]
>>> plt.plot([0,s2*vec2d[0]],[0,s2*vec2d[1]],'ro-',label='v*s2')
[<matplotlib.lines.Line2D object at 0x0A8A9DF0>]
>>> plt.plot([0,s3*vec2d[0]],[0,s3*vec2d[1]],'ro-',label='v*s3')
[<matplotlib.lines.Line2D object at 0x0A8A9FE8>]
>>> plt.axis('square')#it can also be a rectangle also (what ever may be the shape)
(-1.15, 5.449999999999999, -2.3, 4.3)
>>> plt.xlim([-4,4])
(-4, 4)
>>> plt.ylim([-4,4])
(-4, 4)
>>> plt.grid()
>>> plt.legend()
<matplotlib.legend.Legend object at 0x0A8A9BC8>
>>> plt.show()
>>> plt.plot([0,vec2d[0]],[0,vec2d[1]],'bs-',label='v')
[<matplotlib.lines.Line2D object at 0x096980E8>]
>>> plt.plot([0,s1*vec2d[0]],[0,s1*vec2d[1]],'ro-',label='v*s1')
[<matplotlib.lines.Line2D object at 0x09698E98>]
>>> plt.plot([0,s2*vec2d[0]],[0,s2*vec2d[1]],'ro-',label='v*s2')
[<matplotlib.lines.Line2D object at 0x09698E20>]
>>> plt.plot([0,s2*vec2d[0]],[0,s2*vec2d[1]],'kp-',label='v*s2')
[<matplotlib.lines.Line2D object at 0x09698B08>]
>>> plt.plot([0,s3*vec2d[0]],[0,s3*vec2d[1]],'g*-',label='v*s3')
[<matplotlib.lines.Line2D object at 0x09698CB8>]
>>> plt.axis('rectangle')#it can also be a rectangle also (what ever may be the shape)
Traceback (most recent call last):
  File "<pyshell#178>", line 1, in <module>
    plt.axis('rectangle')#it can also be a rectangle also (what ever may be the shape)
  File "C:\Users\reddy\AppData\Local\Programs\Python\Python38-32\lib\site-packages\matplotlib\pyplot.py", line 2412, in axis
    return gca().axis(*args, **kwargs)
  File "C:\Users\reddy\AppData\Local\Programs\Python\Python38-32\lib\site-packages\matplotlib\axes\_base.py", line 1699, in axis
    raise ValueError('Unrecognized string %s to axis; '
ValueError: Unrecognized string rectangle to axis; try on or off
>>> plt.axis('square')
(-1.15, 5.449999999999999, -2.3, 4.3)
>>> plt.xlim([-4,4])
(-4, 4)
>>> plt.ylim([-4,4])
(-4, 4)
>>> plt.grid()
>>> plt.legend()
<matplotlib.legend.Legend object at 0x096CFBE0>
>>> plt.show()
>>> 
>>> #VECTOR DOT PRODUCT
>>> #dot product id denoted by (vec1^T)vec2
>>> #2 vectors should have same dimensionality
>>> #in order to do dot product
>>> #dot product of 2 vectors is always a scalar(i.e.,an integer)
>>> v1=np.array([4,5,6,2])
>>> v2=np.array([3,0,5])
>>> np.dot(v1,v2) #it cannot perform a dot product since v1 and v2 are not having same dimensionality, so it throws an error
Traceback (most recent call last):
  File "<pyshell#193>", line 1, in <module>
    np.dot(v1,v2) #it cannot perform a dot product since v1 and v2 are not having same dimensionality, so it throws an error
  File "<__array_function__ internals>", line 5, in dot
ValueError: shapes (4,) and (3,) not aligned: 4 (dim 0) != 3 (dim 0)
>>> 
>>> #therefore we need change any one of the vector's dimensions equal to other vectors dimension
>>> v1=np.array([5,6,2])
>>> v2=np.array([3,0,5])
>>> np.dot(v1,v2)
25
>>> v3=np.array([-4,3,1])
>>> np.dot(v2,v3)
-7
>>> v4=np.array([0,0,0])
>>> np.dot(v2,v4)
0
>>> #since v4 is a zero vector, so any number multiplied by zero will be 0 itself
>>> v3=np.array([-4,3,1])
>>> v5=np.array([2,3,-1])
>>> np.dot(v3,v5)
0
>>> #ORTHOGONALITY - TWO VECTORS HAVING DOT PRODUCT AS ZERO
>>> 
>>> #ASSIGNMENT
>>> #1) come up with 3 2D vectors, two of them are orthogonal, but neither is orthogonal to the third
>>> #2) plot all the three vectors -- make sure the axes are square and equal
>>> #CODE FOR 1
>>> v1=np.array([2,3])
>>> v2=np.array([-9,6])
>>> v3=np.array([1,2])
>>> np.dot(v1,v2)
0
>>> np.dot(v2,v3)
3
>>> np.dot(v1,v3)
8
>>> #CODE FOR 2
>>> #Plotting
>>> plt.plot([0,v1[0]],[0,v1[1]],'bs-',label='v1')
[<matplotlib.lines.Line2D object at 0x0977F508>]
>>> plt.plot([0,v2[0]],[0,v2[1]],'ro-',label='v2')
[<matplotlib.lines.Line2D object at 0x0978CA00>]
>>> plt.plot([0,v3[0]],[0,v3[1]],'ko-',label='v3')
[<matplotlib.lines.Line2D object at 0x0978C250>]
>>> plt.axis('square')
(-9.55, 2.5500000000000007, -0.30000000000000004, 11.8)
>>> plt.xlim([-10,10])
(-10, 10)
>>> plt.ylim([-10,10])
(-10, 10)
>>> plt.grid()
>>> plt.legend()
<matplotlib.legend.Legend object at 0x0977F118>
>>> plt.show()
>>> 
>>> #MATRICES
>>> #matrices are of 2 types square matrix and rectangle matrix
>>> #SQUARE MATRIX - No.of rows = No.of columns
>>> #RECTANGLE MATRIX - No.of rows and No.of columns are not equal
>>> #DIAGONAL MATRIX - in a matrix all the elements except diagonal elements are zero
>>> #OFF-DIAGONAL MATRIX - matrices other than diagonal matrices are known as off-diagonal matrices (opposite to diagonal matrices)
>>> #SYMMETRIC MATRIX - Is a square matix where all the elements above the diagonal are mirrored with all the elements below the diagonal
>>> #IDENTITY MATRIX - A diagonal matrix with all the diagonal elements as 1
>>> np.i(3) #3 indicates order of matrix
Traceback (most recent call last):
  File "<pyshell#239>", line 1, in <module>
    np.i(3) #3 indicates order of matrix
  File "C:\Users\reddy\AppData\Local\Programs\Python\Python38-32\lib\site-packages\numpy\__init__.py", line 219, in __getattr__
    raise AttributeError("module {!r} has no attribute "
AttributeError: module 'numpy' has no attribute 'i'
>>> np.eye(3)
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])
>>> np.zeros((3,4)) #zero matrix
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
>>> np.full((5,2),7) #matrix with oreder (5,2) with all the elements 7
array([[7, 7],
       [7, 7],
       [7, 7],
       [7, 7],
       [7, 7]])
>>> M= np.array([1,2,3],[4,5,6],[7,8,9]) #craeting our own matrix
Traceback (most recent call last):
  File "<pyshell#243>", line 1, in <module>
    M= np.array([1,2,3],[4,5,6],[7,8,9]) #craeting our own matrix
ValueError: only 2 non-keyword arguments accepted
>>> M= np.array([[1,2,3],[4,5,6],[7,8,9]]) #craeting our own matrix
>>> print(M)
[[1 2 3]
 [4 5 6]
 [7 8 9]]
>>> 
>>> #ASSIGNMENT
>>> #1)create 3 matrices : 2*2,2*2,3*2
>>> M= np.array([[1,2],[4,5]]) #creating 1st matrix
>>> M1= np.array([[7,3],[0,5]]) #creating 2nd matrix
>>> M1= np.array([[3,9,1],[4,0]]) #creating 3rd matrix
>>> print(M)
[[1 2]
 [4 5]]
>>> print(M1)
[list([3, 9, 1]) list([4, 0])]
>>> 
>>> #TRANSPOSING VECTORS AND MATRICES
>>> #TRANSPOSING - in a matrix converting rows into columns and vice versa
>>> v1= np.array([2,3,-1])
>>> print(v1)
[ 2  3 -1]
>>> v1= np.array([2,3,-1],ndmin=2)#ndmin is used to transpose a matrix
>>> print(v1)
[[ 2  3 -1]]
>>> print(' ')
 
>>> print(v1.T)
[[ 2]
 [ 3]
 [-1]]
>>> M=np.random.random(3,3))
SyntaxError: unmatched ')'
>>> M=np.random.random(3,3)
Traceback (most recent call last):
  File "<pyshell#264>", line 1, in <module>
    M=np.random.random(3,3)
  File "mtrand.pyx", line 423, in numpy.random.mtrand.RandomState.random
TypeError: random() takes at most 1 positional argument (2 given)
>>> M=np.random.randn(3,3)
>>> print(M)
[[ 1.34116959 -1.98382462  1.27415126]
 [ 0.97263253  0.67392317 -0.33189604]
 [ 0.42294561  1.26887063  0.25488191]]
>>> #random 3*3 matrix
>>> M=np.round(random.randn(3,3))
Traceback (most recent call last):
  File "<pyshell#268>", line 1, in <module>
    M=np.round(random.randn(3,3))
NameError: name 'random' is not defined
>>> M=np.round(np.random.randn(3,3))
>>> print(M)
[[ 0. -0. -2.]
 [ 0.  0.  1.]
 [ 0. -1.  0.]]
>>> print(' ')
 
>>> print(M.T)
[[ 0.  0.  0.]
 [-0.  0. -1.]
 [-2.  1.  0.]]
>>> 
>>> #ASSIGNMENT
>>> #1) What happens when we transpose twice? M matrix, mT transpose, MTT
>>> #2) confirm that the traspose operation works on non-square matrices
>>> print(M.T.T)
[[ 0. -0. -2.]
 [ 0.  0.  1.]
 [ 0. -1.  0.]]
>>> print(M.T.T)#twice transposing
[[ 0. -0. -2.]
 [ 0.  0.  1.]
 [ 0. -1.  0.]]
>>> #2)
>>> M=np.round(np.random.randn(3,4))
>>> print(M)
[[ 0.  2. -2.  1.]
 [-1.  1. -1.  3.]
 [-0.  2. -1. -1.]]
>>> print(' ')
 
>>> print(M.T)
[[ 0. -1. -0.]
 [ 2.  1.  2.]
 [-2. -1. -1.]
 [ 1.  3. -1.]]
>>> #YES TRANSPOSE WORKS ON NON-SQUARE MATRICES ALSO
>>> 
>>> #MATRIX MULTIPLICATION
>>> #Condition for multiplying 2 matrices is number of colums in a a first matrix should be equal to number of rows in a second matrix
>>> M1=np.random.randn(4,5)
>>> M2=np.random.randn(4,5)#these 2 matrices cannote be multipied since the condition is not satisfied(ie.,number of colums in a a first matrix should be equal to number of rows in a second matrix)
>>> print(np.matmul(M1,M2))
Traceback (most recent call last):
  File "<pyshell#290>", line 1, in <module>
    print(np.matmul(M1,M2))
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 4 is different from 5)
>>> print(np.matmul(M1,M2.T))
[[ 1.68892779  1.30209125 -0.63795048  0.20712899]
 [-0.17219962  3.58948827 -1.71772126 -5.18085376]
 [ 1.20518797 -0.61747049  0.28234111  1.00970157]
 [-2.27582311  2.72373212 -0.04001809  0.25295339]]
>>> print(' ')
 
>>> print(M1@M2.T)#np.matmul(M1,M2.T) AND M1@M2.T both are same
[[ 1.68892779  1.30209125 -0.63795048  0.20712899]
 [-0.17219962  3.58948827 -1.71772126 -5.18085376]
 [ 1.20518797 -0.61747049  0.28234111  1.00970157]
 [-2.27582311  2.72373212 -0.04001809  0.25295339]]
>>> print(np.matmul(M1,M2.T) - M1@M2.T)
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
>>> 
>>> #ASSIGNMENT
>>> #create a 3*3 matrix of integers
>>> #multiply that matrix with identity matrix, zero matrix and that matrix's transpose
>>> #CODE
>>> M= np.array([1,2,3],[4,5,6],[7,8,9])
Traceback (most recent call last):
  File "<pyshell#300>", line 1, in <module>
    M= np.array([1,2,3],[4,5,6],[7,8,9])
ValueError: only 2 non-keyword arguments accepted
>>> M= np.array([[1,2,3],[4,5,6],[7,8,9]])
>>> print(M)
[[1 2 3]
 [4 5 6]
 [7 8 9]]
>>> print(np.matmul(M,np.eye(3)))#multiply that matrix with identity matrix
[[1. 2. 3.]
 [4. 5. 6.]
 [7. 8. 9.]]
>>> print(np.matmul(M,np.zeros(3)))#multiply that matrix with zero matrix
[0. 0. 0.]
>>> print(np.matmul(M,M.T))
[[ 14  32  50]
 [ 32  77 122]
 [ 50 122 194]]
>>> 
>>> #MATRIX INVERSION
>>> M=np.random.randn(4,4)#4*4 random matrix
>>> print(M)
[[-0.59334184  0.05148649  0.41309298 -0.58951774]
 [ 0.5203167   1.41920841 -1.00826016 -0.6100727 ]
 [ 0.54864314 -1.09441584  1.04918206 -0.80237303]
 [-1.00824227  1.19182809  0.74803416  0.71462242]]
>>> M=np.random.randn(4,4)
>>> Minv= np.linalg.inv(A)
Traceback (most recent call last):
  File "<pyshell#311>", line 1, in <module>
    Minv= np.linalg.inv(A)
NameError: name 'A' is not defined
>>> Minv= np.linalg.inv(M)
>>> MinvM = M@Minv
>>> print(M)
[[ 0.89485582 -0.65359362 -0.68095639  0.9176753 ]
 [ 0.4243675  -0.6864128  -0.68466406 -0.08145267]
 [-1.37962846 -0.51243375  0.37991467 -1.32110956]
 [ 3.05196212  0.25039776  1.3797482  -0.49377724]]
>>> print(' ')
 
>>> print(Minv)
[[-0.23140818  0.49541355 -0.26807213  0.20544148]
 [-1.46808579  0.65543778 -1.01543709 -0.11970899]
 [ 1.19115825 -1.66196302  0.83325649  0.25850667]
 [ 1.15364581 -1.24952484  0.15649765 -0.09376929]]
>>> print(' ')
 
>>> print(M)
[[ 0.89485582 -0.65359362 -0.68095639  0.9176753 ]
 [ 0.4243675  -0.6864128  -0.68466406 -0.08145267]
 [-1.37962846 -0.51243375  0.37991467 -1.32110956]
 [ 3.05196212  0.25039776  1.3797482  -0.49377724]]
>>> print(MinvM)
[[ 1.00000000e+00 -2.22044605e-16  2.77555756e-17  5.55111512e-17]
 [ 2.63677968e-16  1.00000000e+00  2.08166817e-16  2.60208521e-18]
 [ 8.88178420e-16 -6.66133815e-16  1.00000000e+00 -2.77555756e-17]
 [ 0.00000000e+00  1.11022302e-16  5.55111512e-17  1.00000000e+00]]
>>> 
>>> fig,ax = plt.subplots(1,3,figsize=(6,5))
>>> ax[0].imshow(A)#for matrices represented with colors
Traceback (most recent call last):
  File "<pyshell#322>", line 1, in <module>
    ax[0].imshow(A)#for matrices represented with colors
NameError: name 'A' is not defined
>>> ax[0].imshow(M)#for matrices represented with colors
<matplotlib.image.AxesImage object at 0x0980F550>
>>> ax[1].imshow(Minv)
<matplotlib.image.AxesImage object at 0x01559970>
>>> ax[2].imshow(MinvM)
<matplotlib.image.AxesImage object at 0x01559D90>
>>> plt.show()
>>> ax[0].imshow(M)
<matplotlib.image.AxesImage object at 0x0A267610>
>>> ax[0].set.title('M')
Traceback (most recent call last):
  File "<pyshell#328>", line 1, in <module>
    ax[0].set.title('M')
AttributeError: 'function' object has no attribute 'title'
>>> ax[0].set_title('M')
Text(0.5, 1, 'M')
>>> ax[1].imshow(Minv)
<matplotlib.image.AxesImage object at 0x0982B430>
>>> ax[1].set_title('M$^{-1}$') #LATEX- is aspecial way of python to interpret non standard letters
Text(0.5, 1, 'M$^{-1}$')
>>> ax[2].imshow(MinvM)
<matplotlib.image.AxesImage object at 0x09807AD8>
>>> ax[2].set_title('M$^{-1}$M')
Text(0.5, 1, 'M$^{-1}$M')
>>> plt.show()
>>> 
>>> 
