# preamble
import numpy as np
import matplotlib.pyplot as plt
plt.interactive(True)

# Chebyshev polynomials 
def ChebU( n, x ):
    if n<0:  return 0
    if n==0: return 1
    chebs = np.zeros((n+1,))
    chebs[0] = 1
    chebs[1] = 2*x
    for i in range(2,n+1):
        chebs[i] = 2*x*chebs[i-1] - chebs[i-2]
    return chebs

# capacitance coefficients
def cap_coeffs( c, a, b, Nmax=10 ):
    coshU = (c**2 - a**2 - b**2)/(2*a*b)
    chebs = ChebU( Nmax, coshU )
    Caa = 1./b
    Caa += np.sum( 1./( a*chebs[:-1]+b*chebs[1:] ) )
    Caa *= a*b
    Cbb = 1./a
    Cbb += np.sum( 1./( b*chebs[:-1]+a*chebs[1:] ) )
    Cbb *= a*b
    Cab = -a*b/c*np.sum(1./chebs)
    return [Caa,Cab,Cbb]

# electrostatic energy
def W(Qa,Qb,Caa,Cab,Cbb): 
    return .5 * ( Qa**2*Cbb - 2*Qa*Qb*Cab + Qb**2*Caa ) / ( Caa*Cbb - Cab**2 )

# exemplary evaluation:
# define center-to-center distances
x  = np.linspace(3.0,4.5,60)
# evaluate the capacitance coeffs
cs  = np.array( [cap_coeffs(xx,1.,2.,Nmax=300) for xx in x] )
# find the corresponding energy for Q_a=1 and Q_b=Q_a/2
ws = np.array( [W(1,.5,c[0],c[1],c[2]) for c in cs] )
# and normalize to the self-energy of spheres A and B
ws /= .5+.5**2/2./2.

# plot energy
plt.plot( (x-3)/3, ws )
plt.xlabel('s/(a+b)')
plt.ylabel('electrostatic energy')
