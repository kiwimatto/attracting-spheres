# preamble
import numpy as np
from scipy.constants import epsilon_0

###  WARNING: if thetaqp is zero anywhere off the diagonal, 
###  the associated Legendre function of 1st kind will
###  diverge (when cos(thetaqp)==1.) 
###  Choose your geometries carefully.

USEMPMATH = False
if USEMPMATH:
    import mpmath
    mpmath.mp.dps = 20
    factorial = mpmath.fac
else:
    import scipy.special
    factorial = lambda(x): np.prod(np.arange(1,x+1))


def sphere_coords( xs, ys, zs ):
    distanceqp = np.zeros((xs.size,xs.size))
    thetaqp    = np.zeros((xs.size,xs.size))
    phiqp      = np.zeros((xs.size,xs.size))
    for i in range(xs.size):
        for j in range(ys.size):
            distanceqp[i,j] = np.sqrt( (xs[j]-xs[i])**2 + (ys[j]-ys[i])**2 + (zs[j]-zs[i])**2 )
            thetaqp[i,j] = np.arctan2( np.sqrt((xs[j]-xs[i])**2 +(ys[j]-ys[i])**2), zs[j]-zs[i] )
            if i!=j:
                if np.abs(np.cos(thetaqp[i,j])-1)<1e-10:
                    print '--- WARNING!!! ---  cos(theta) is (nearly) one,'
                    print '--- The Legendre function has a pole here!'
            phiqp[i,j] = np.arctan2( ys[j]-ys[i], xs[j]-xs[i] )
    return distanceqp,thetaqp,phiqp


# geometry definition, give number of spheres, coordinates, radii
def geometry(S, xs, ys, zs, radii):
    geom = {}
    geom['S'] = S
    distanceqp,thetaqp,phiqp = sphere_coords( np.array(xs), np.array(ys), np.array(zs) )
    geom['distanceqp'] = distanceqp
    geom['thetaqp'] = thetaqp
    geom['phiqp'] = phiqp
    geom['radii'] = radii
    return geom

# what follows below comes from this paper:
# QUARTERLY OF APPLIED MATHEMATICS, VOLUME LXXII, NUMBER 4, DECEMBER 2014, PAGES 613--623
# "TRANSLATIONAL ADDITION THEOREMS FOR SPHERICAL LAPLACIAN FUNCTIONS AND THEIR APPLICATION
#  TO BOUNDARY-VALUE PROBLEMS" by IOAN R. CIRIC and KUMARA S. C. M. KOTUWAGE 

def eqn42_solve( geom, Nmax ):
    # eqn 4.2 
    NMAX = Nmax         # where to truncate the infinite system of eqns
    S = geom['S']
    distanceqp = geom['distanceqp']
    thetaqp = geom['thetaqp']
    phiqp = geom['phiqp']
    radii = geom['radii']

    # coefficient matrix
    if USEMPMATH:
        CM = mpmath.zeros( S*NMAX*(2*NMAX+1) )
    else:
        CM = np.zeros( (S*NMAX*(2*NMAX+1), S*NMAX*(2*NMAX+1)) )

    for sprimei,sprime in enumerate(range(S)):
        for nprimei,nprime in enumerate(range(NMAX)):
            for mprimei,mprime in enumerate(range(-nprime,nprime+1)):

                # row index
                ri = sprimei*NMAX*(2*NMAX+1) + nprimei*(2*NMAX+1) + mprimei
                # row prefactors
                prefac = (-1)**(nprime+mprime) * radii[sprimei]**(nprime+1) \
                         * 1./factorial(nprime+mprime)

                for si,s in enumerate(range(S)):
                    if sprimei != si:
                        for ni,n in enumerate(range(NMAX)):
                            for mi,m in enumerate(range(-n,n+1)):

                                # column index
                                ci = si*NMAX*(2*NMAX+1) + ni*(2*NMAX+1) + mi

                                f1 = distanceqp[sprimei,si]**(-(n+1))
                                f2 = (radii[sprimei]/distanceqp[sprimei,si])**nprime
                                f3 = factorial(n-m+nprime+mprime)/factorial(n-m)
                                if USEMPMATH:
                                    f4 = mpmath.legenp( n+nprime, m-mprime, \
                                                        np.cos(thetaqp[sprimei,si]) )
                                else:
                                    f4 = scipy.special.lpmv( m-mprime, n+nprime, \
                                                             np.cos(thetaqp[sprimei,si]) )
                                f5 = np.exp( 1j*(m-mprime)*phiqp[sprimei,si] )

                                CM[ri,ci] = prefac*f1*f2*f3*f4*f5


    if USEMPMATH:
        CM += mpmath.diag(np.ones(S*NMAX*(2*NMAX+1)))
        Qs = mpmath.zeros(S)
    else:
        CM += np.diag(np.ones(S*NMAX*(2*NMAX+1)))
        Qs = np.zeros((S,S))              


    for si in range(S):
        if USEMPMATH:
            rhs = mpmath.zeros( S*NMAX*(2*NMAX+1), 1 )
        else:          
            rhs = np.zeros((CM.shape[0],))
        rhs[si*NMAX*(2*NMAX+1):(si*NMAX+1)*(2*NMAX+1)] = radii[si]
        if USEMPMATH:     
            sol = mpmath.lu_solve( CM, rhs )
        else:
            sol = np.linalg.solve( CM, rhs )
        #print sol[::(NMAX*(2*NMAX+1))]
        Qs[si,:] = sol[::(NMAX*(2*NMAX+1))]
    
    return Qs


def W( C, Q, geom ):
    """ Calculate energy when given capacitance _matrix_ and charge _vector_ """
    # cap matrix inverse
    Cinv = [np.linalg.inv(c) for c in C]
    # vector of potentials
    V = [np.dot(cinv,Q) for cinv in Cinv]
    # energy
    W  = [.5 * np.dot( Q, v ) for v in V]
    # divide by sum of the self-energies of the spheres
    W /= np.sum( np.array( [.5 * Q[i]**2/geom['radii'][i] for i in range(Q.size)] ) )
    return W


#### EXAMPLES, to be un-commented

## two spheres
#geom = geometry( S=2, xs=[0,0], ys=[0,0], zs=[0,4.], radii=[1.,2.] )
#print eqn42_solve( geom, Nmax=10 )

# ## three spheres
# geom = geometry( S=3, xs=[0,10.,5.], ys=[0,0,0], zs=[0,0,np.cos(np.pi/6)*10], radii=[3.,5.,4.] )
# r = eqn42_solve( geom, Nmax=10 )
# print r
# print "For comparison with the results in the C and K paper:"
# print r * 4*np.pi*epsilon_0

# ## LOOP FOR COMPARISON WITH LEKNER RESULTS
# QQ = []
# for d in np.linspace(3,4.5,60):
#     geom = geometry( S=2, xs=[0,0], ys=[0,0], zs=[0,d], radii=[1.,2.] )
#     QQ.append( eqn42_solve( geom, Nmax=16 ) )

