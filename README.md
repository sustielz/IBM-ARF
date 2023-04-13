# IBM-ARF
An immersed boundary method to model an acoustically levitated droplet by calculating the acoustic radiation force at its surface. 


run example.py to simulate a spherical droplet in a standing plane wave. run _collect.py and then _draw.py to plot results. Adjust parameters by editing default_params.txt. The directory src contains python code for the immersed boundary method, and the main directory subclasses ib2 objects which calculates acoustic fields+forces and adds the ARF as a Lagrangian surface force.
