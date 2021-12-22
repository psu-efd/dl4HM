import numpy as np
from scipy.special import comb
from matplotlib import pyplot as plt
from scipy.stats.qmc import LatinHypercube

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def generate_polygons(nPolygons=1, nSegments=6):
    #generate a symmetric polygon using Bezier curve

    engine_x = LatinHypercube(d=(nSegments))  #exlude the first and last points
    engine_y = LatinHypercube(d=(nSegments)) #exlude the first and last points

    #xPoints = np.linspace(-0.5,0.5,nPoints)
    xPoints = np.zeros((nPolygons, nSegments+2))
    yPoints = np.zeros((nPolygons, nSegments+2))

    xPoints[:,1:(nSegments+1)] = engine_x.random(n=nPolygons)
    yPoints[:,1:(nSegments+1)] = engine_y.random(n=nPolygons)

    #scale and shift the x coordinates of internal points
    for iPolygon in range(nPolygons):
        for iPoint in range(1,nSegments+1):
            xstart = 1.0/(nSegments)*(iPoint-1)
            xend = 1.0/(nSegments)*iPoint

            xPoints[iPolygon,iPoint] = (xend-xstart)*xPoints[iPolygon,iPoint] + xstart

    #fix the start and end point x and y coordinates
    xPoints[:,0] = 0.0
    xPoints[:,-1] = 1.0

    yPoints[:,0] = 0.0
    yPoints[:,-1] = 0.0

    points = np.column_stack((xPoints[0,:],yPoints[0,:]))

    print(points)

    #sort the points along x
    #points = np.sort(points, axis=0)

    #xpoints = [p[0] for p in points]
    #ypoints = [p[1] for p in points]

    xvals, yvals = bezier_curve(points, nTimes=100)

    #mirror the curve and add to upper part to make a closed polygon
    #xvals_reverse = np.flip(xvals)
    #xvals = np.concatenate(xvals, xvals_reverse[1:-1])

    return xvals, yvals, xPoints, yPoints


if __name__ == "__main__":

    #number of polygons to generate
    nPolygons = 10

    #number of segements
    nSegments = 6

    #generate polygons
    xvals, yvals, xPoints, yPoints = generate_polygons(nPolygons, nSegments)

    plt.plot(xvals, yvals)

    plt.plot(xPoints[0,:], yPoints[0,:], "ro")
    #for nr in range(len(points)):
    #    plt.text(points[nr][0], points[nr][1], nr)

    #plot the
    for i in range(nSegments):
        r = np.random.random()
        b = np.random.random()
        g = np.random.random()
        color = (r, g, b)
        plt.axvspan(1.0/(nSegments)*i, 1.0/(nSegments)*(i+1), facecolor=color, alpha=0.5)

    plt.show()