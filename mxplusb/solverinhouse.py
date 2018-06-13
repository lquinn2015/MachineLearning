from numpy import *

def run():
	points = genfromtxt("data.csv", delimiter=",")
	initB = 0
	initM = 0
	numberOfIterations = 1000
	[b, m] = gradient_decent_runner(points, initB, initM, numberOfIterations)
	print("y=" + str(m) + "x+"+str(b) )
	print("Error: = " + str(compute_error_for_line_given_points(points, b, m)) )

def compute_error_for_line_given_points(points, b, m):
	accumulatedError = 0
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		accumulatedError += (y - (m*x+b)) ** 2
	return accumulatedError/float(len(points))

def gradient_decent_runner(points, bI, mI, numberOfIterations):
	b = bI
	m = mI
	for i in range(0, numberOfIterations):
		b, m = step_gradient(b, m, array(points))
	return [b, m]
	
def step_gradient(bCurr, mCurr, points):
	bPartial = 0
	mPartial = 0
	N = float(len(points))
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		mPartial += -(2/N) * x * (y-(mCurr*x+bCurr))
		bPartial += -(2/N) * (y - (mCurr*x+bCurr))
	bNew = bCurr - bPartial * .0001
	mNew = mCurr - mPartial * .0001
	return [bNew, mNew]


if __name__ == '__main__':
	run()

