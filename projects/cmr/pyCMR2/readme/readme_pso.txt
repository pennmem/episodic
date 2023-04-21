Particle Swarm Instructions

1.  Build CMR2 Cython package 

See readme_cmr2.txt, but in short, run at the command line:

>> python setup_cmr2.py build_ext —in_place

#########################################################################################

2.  Run noise_maker.py

This file takes two arguments at the command line.  The first is swarm size, the number of particles that you are using in pyswarm.  The second is equal to the parameter of max iterations that you set within pso_par_cmr2_eval.py.  

For example, if swarmsize=20 and max_iter=100, run:

>> python noise_maker.py 20 100

*** For a quick test run, run swarmsize=5, max_iter=5

*** Make sure to enter the lower and upper bounds that you are using for your parameters not only in the pso_par_cmr2_eval.py file, but also in the noise_maker.py file, because these set the bounds for the random noise draws.

This file creates two files per iteration of particle swarm.  

Each file contains a matrix of noise values that will be added to the parameter vectors as they are evolved during the fit process.  

We create these noise values ahead of time so that all instances of pso_par_cmr2_eval.py, as they run in parallel with one another, will still be using the same noise values and will be “on the same page” with each other.

#########################################################################################

3.  Set up pso_par_cmr2_eval.py

Set swarmsize = desired number of particles to test.  

	Each particle represents a vector of possible parameter values, 
	which are at first randomly assigned.  

	The default setting for the pyswarm code base we are working from is
	swarmsize=100.

Set maxiter = desired number of iterations, or evolutions of the particles.  

	On each iteration, the code will find the best set of parameters in this 
	iteration, then have all the other sets of parameters move slightly 
	toward that parameter.  

	Eventually, the parameter sets will all converge toward 
	one particular set of values.

	Preliminary analyses suggest that good fits begin to emerge, 
	for 100 particles, at about 30 iterations.  

So for instance, you might set:

	xopt, fopt = pso(obj_func, lb, ub, swarmsize=100, maxiter=30, debug=False)  

Set the path to your semantic similarity file.

	This occurs in the main function.  You can set alternate paths depending
	on whether you wish to use a test version on your local machine, 
	vs. data on Rhino.

	If you want to use the Rhino paths, set on_rhino = True.  Else, set it to False.

	Currently this variable is called LSA_path, but you can use 
	any type of similarity matrix that you like, including w2v, 
	as is now lab standard.

	The similarity matrix must be contained within a .txt file.

	This is because a 1638 x 1638 matrix of floats is large and introduces
	high overhead if read in from a MATLAB file via scipy.io.loadmat

Set the path to your data file.

	This file must be a MATLAB file (currently; other formats pending) 
	that is in the standard structure used by the CML 
	for free recall task behavioral data.

#########################################################################################

4.  Launch N versions of pso_par_cmr2_eval.py — see section 6 for details.

N should not exceed the number of particles in your swarm.  

	This is because each instance of pso_par_cmr2_eval.py can only work on 
	one particle (one parameter vector) at a time.  

	So if swarmsize=20 and you launch 30 instances of 
	pso_par_cmr2.py, you will have 20 jobs working, 
	whereas the 10 extra ones will just sit there doing nothing most of the time.

#########################################################################################

5.  Output

Your most important output will consist of a text file containing the final set of best-fitting parameters.

	You can change the file name in pso_par_cmr2_eval.py 
	in the second-to-last line in the code,

		np.savetxt('filename.txt', xopt, delimiter=',', fmt='%f')

	The current filename is set to: "xopt_k02.txt"

The code will also output a series of temporary files.  

	These essentially save the “state of the system” such that 
	if you decide you need more iterations than you originally gave, 
	you can restart the particle swarm and pick up right where you left off,
	without wasting time redoing the 30 iterations you just ran.

	Keep in mind that to do this, you will need to first generate more
	noise files (e.g., if you made 20 noise files for 20 iterations, 
	and now you wish you had done 40 iterations, re-run noise_maker.py
	with the value 40 iterations.

	If you are satisfied with your current number of iterations, 
	then you can delete these files.

	File stems include:

	xfile - saves particle values on that iteration
	vfile - saves particle velocities on that iteration
	pfile - saves best particle values on that iteration 
		(slightly unclear to me the function of this variable, 
		so more info pending)

	rmses_iter - saves all particle fit metrics from that iteration

	tempfile - a temporary file that contains just a single fit value 
		and the index of which parameter set it is from.  

		These files are how each instance of particle swarm stays 
		on the same page with other instances of particle swarm
		such that no one works on assessing the same parameter vec.

		These files will be cleaned out of your directory at each
		“cleanup” point of 10 iterations, as well as at the end 
		of the code. 

		All information in these files is contained in the rmse_iter
		files, so do not worry about saving the information contained
		in the tempfiles.


#########################################################################################

6.  Code for Easy launch of N jobs 

For easy launch of N versions of pso_par_cmr2_eval.py, you can use a script called pgo and its helper file, runpy.sh 

In the pyCMR2 svn repository, these are located in the directory “pgo_files”

To use:

	Make a copy of pgo and runpy.sh in your binaries (bin) folder 
	on your user directory on Rhino.  

	Update the path to your python distribution, as detailed below, 
	then run from the command line:

	>> pgo pso_par_cmr2_eval.py 50

		^ if you would like to submit 50 jobs.

File documentation (thanks to Christoph and others for both files):

pgo is a bash script to submit your jobs.  

	Its main function is a for loop that will submit 
	the desired number of jobs that you provide it.  

	Here is also where you control the hard limit on how much memory
	you want each job to use.  

	For example, if you want to limit each job to 2G of RAM, 
	then within the pgo file, at the very bottom where you see 
	the for loop, set:

	-l h_vmem=2G

	^ this is the current setting

runpyfile.sh is a bash script that enables you to control how python is set up 
	for the jobs that you submit via pgo.

	Before using pgo, change the variable PY_COMMAND in runpyfile.sh to the path of 
	the python distribution that you would like to use.  

	For example, my path is set to:

	PY_COMMAND="/home1/rivkat.cohen/anaconda3/bin/python”

	because that is where I have installed python on my Rhino account.

#########################################################################################

Additional notes:

For any future users who may be new to working with terminal commands, do not enter the >> at the beginning of each command line prompt instruction.  

These are just to distinguish the command line text from the rest of the instructions.

Good luck!  Contact Rivka Cohen (rivkat.cohen@gmail.com) if you have questions!




