here, using mcx tool to generating Monte Carlo simulation.
Each simulation can be seemed as one point in the spectrum.

in the folder "input_template/share_files/model_input_related" contains the possibility distribution of LED source.
the mcx simulator is compiled from Eric syu and Kao Ben edited version : https://github.com/syuys/ijv_2/tree/master/mcx/src 

1. make_simarry.py : using the range based on the different literatures and get "mus_range" and "mua_range", you can setting each layer OPs split point to get total dataset. (e.g. skin_mus samples 7 points in range, fat_mus samples 7 points in range, muscle_mus samples 5 points in range, vessel_mus samples 5 points in range, so total we have 1225 combinations in mus_set. **note that ijv and cca mus is same)

2. 


TODO :

S6_plot_sim.py
plot_flux_3d.py
plot_flux.py
plot_voxel_fix.py
