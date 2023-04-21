************************************************************************

Here is a list of what I use each file for.  Let me know if something is missing!

1.  CMR2_pack_cyth_LTP228.pyx

This has the code for an immediate free recall version of CMR2.  To use it, build it by running the following from the command line:

>>python setup_cmr2_LTP228.py build_ext --inplace

2.  setup_cmr2_LTP228.py

This is the setup file for the CMR2 build described above.

3.  lagCRP2.py

This is a basic version of lag-CRP code.  

It is a little inefficient, in that it is not vectorized.  However, this allows us to get the standard errors (across lists) for the lag-CRP bins, which we need for norming the errors during the particle swarm.

4.  pres_nos_LTP228.txt, rec_nos_LTP228.txt

These are .txt files with the presented items and recalled items, respectively, for LTP228.

5.  make_txt_from_mat.py

This is the code used to turn an ltpFR2 matlab file into .txt files so that you can run them with CMR2.

6.  w2v.txt 

This is an inter-item similarity matrix.  Thanks Shai!

7.  get_fit_graph.py

This will take all the rmse_iter files in your directory (or the directory you point it to) and give you a graph showing you how the min and mean error is decreasing (you hope!) over time.

It will also give you which iteration (of the ones so far) has the best-fitting value, and what that value is.

8.  get_params_from_rmse.py

This code takes an argument from the command line for which iteration you would like the best-fitting parameters from.  

For instance, if you are mid-way through a particle swarm, and you want to check on how it is doing, first run:

>> python get_fit_graph.py

Then take the iteration that has the best-fitting value (e.g., 13) and run:

>> python get_params_from_rmse.py 13

This will also create a file for you called best_params_13, which you can tell graph_CMR2.py to take as its values.

9.  graph_CMR2.py

Takes in a set of parameter values and runs CMR2, then produces graphs of SPC, PFR, lag-CRP, and the eval clustering effect so that you can have a look at the visual fits of these parameters.

There is a bit of a graveyard in here of commented-out parameter sets that I was testing.  Sorry!