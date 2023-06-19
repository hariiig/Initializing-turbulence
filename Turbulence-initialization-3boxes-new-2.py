# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 11:55:52 2023

@author: dell
"""

import h5py
import numpy as np
import random

def create_simulation_parameters(box_number):
    """
    This function creates the simulation parameters based on the box number.
    """
    # Common parameters
    rho_init = 1 / (4 * np.pi)
    B0 = 0.01
    inner_radius = 1
    outer_radius = 4
    B_rms = 0.5 * B0
    u_rms = 2 * B0
    compressibility = 0.3

    # Parameters for each box
    if box_number == 1:
        Lx, Ly, Lz, nxc, nyc, nzc, qom_0, qom_1, beta_0, beta_1 = 64.0 / 3, 64.0/3, 0.001, 2048 / 3, 2048, 1, -100, 1, 0.5, 0.5
    elif box_number == 2:
        Lx, Ly, Lz, nxc, nyc, nzc, qom_0, qom_1, beta_0, beta_1 = 64.0 / 3 * 2, 64.0/3 *2, 0.001, 2048 / 3 * 2, 2048/3 *2, 1, -120, 1.5, 0.6, 0.4
    elif box_number == 3:
        Lx, Ly, Lz, nxc, nyc, nzc, qom_0, qom_1, beta_0, beta_1 = 64.0 / 3 / 2, 64.0/3 /2, 0.001, 2048 / 3 / 2, 2048/3/2, 1, -150, 2, 0.7, 0.3
    else:
        raise ValueError(f"Invalid box_number: {box_number}")

    return Lx, Ly, Lz, nxc, nyc, nzc, qom_0, qom_1, beta_0, beta_1, rho_init, B0, inner_radius, outer_radius, B_rms, u_rms, compressibility

def create_axes(Lx, Ly, Lz, nxc, nyc, nzc):
    """
    This function creates the x, y, and z axes for each box.
    """
    x = np.linspace(0, Lx, int(nxc)+1)
    y = np.linspace(0, Ly, int(nyc)+1)
    z = np.linspace(0, Lz, int(nzc)+1)

    return x, y, z

def create_modes(inner_radius, outer_radius, Lx):
    """
    This function creates the ring of modes in the Fourier space.
    """
    compressibility = 0.3
    modes = []
    for m in np.arange(-outer_radius, outer_radius+1):
        for n in np.arange(-outer_radius, outer_radius+1):
            if np.sqrt(m**2+n**2) >= inner_radius and np.sqrt(m**2+n**2) <= outer_radius:
                modes.append([m,n])
    modes = np.array(modes)

    k_factor = (2*np.pi/Lx)*np.sqrt(np.sum([n[0]**2 for n in modes])) # Used to link the potentials amplitudes to the fields rms.
    k_factor_u=(2*np.pi/Lx)*np.sqrt(np.sum([(n[0]*(1-compressibility)+n[1]*compressibility)**2 for n in modes]))

   
    return modes, k_factor, k_factor_u

def create_grid_axes(nxc, nyc):
    """
    This function creates the grid axes.
    """
    i, j = np.meshgrid(np.arange(int(nxc)+1), np.arange(int(nyc)+1)) # Grid axes.

    return i, j

def create_h5_file(file_path, simulation_name, box_number):
    """
    This function creates the h5 file.
    """
    file_name = file_path+"boxnumber-"+str(box_number)+"-Fields_000000.h5"
    f = h5py.File(file_name, "w")

    return f

def run_simulation(box_number):
    """
    This function runs the simulation for a given box number.
    """
    # Create the simulation parameters
    Lx, Ly, Lz, nxc, nyc, nzc, qom_0, qom_1, beta_0, beta_1, rho_init, B0, inner_radius, outer_radius, B_rms, u_rms, compressibility = create_simulation_parameters(box_number)

    # Create the x, y, and z axes for each box
    x, y, z = create_axes(Lx, Ly, Lz, nxc, nyc, nzc)
    #x=np.linspace(0,Lx,nxc+1)
    #y=np.linspace(0,Ly,nyc+1)
    #z=np.linspace(0,Lz,nzc+1)
    # Create the ring of modes in the Fourier space
    modes, k_factor, k_factor_u = create_modes(inner_radius, outer_radius, Lx)

    # Create the grid axes
    i, j = create_grid_axes(nxc, nyc)
    #print (i,j)
    # Path of the folder where the h5 file will be created
    file_path = "C:\\Users\\dell\\OneDrive\\Documents\\Post-Doc_Works\\Post-Doc-Work-Belgium\\Codes\\Test-run\\hdf-files\\"

    # Simulation name
    simulation_name = "turbEarth"

    # Create the h5 file
    f = create_h5_file(file_path, simulation_name, box_number)

     # Magnetic field.
    Ax=np.zeros((int(nzc+1),int(nyc+1),int(nxc+1)))
    Ay=np.zeros((int(nzc+1),int(nyc+1),int(nxc+1)))
    Az=np.zeros((int(nzc+1),int(nyc+1),int(nxc+1)))
    
    #print("Ax="+str(Ax))
    phases=np.array([random.random()*2*np.pi for n in range(np.shape(modes)[0])]) # Random phases.
    amplitudes=np.array([[random.random(),random.random(),random.random()] for n in range(np.shape(modes)[0])]) # Random amplitudes.
    
    for n in range(np.shape(modes)[0]):
    	for iz in range(nzc+1):
    		Ax[iz,:,:]=Ax[iz,:,:]+compressibility*(B_rms*2/k_factor)*amplitudes[n,0]*np.sin(phases[n]+(2*np.pi*modes[n,0]/nxc)*i+(2*np.pi*modes[n,1]/nyc)*j)
    		Ay[iz,:,:]=Ay[iz,:,:]+compressibility*(B_rms*2/k_factor)*amplitudes[n,1]*np.sin(phases[n]+(2*np.pi*modes[n,0]/nxc)*i+(2*np.pi*modes[n,1]/nyc)*j)
    		Az[iz,:,:]=Az[iz,:,:]+(B_rms*np.sqrt(8)/k_factor)*amplitudes[n,2]*np.sin(phases[n]+(2*np.pi*modes[n,0]/nxc)*i+(2*np.pi*modes[n,1]/nyc)*j)
    Bx=np.gradient(Az,y,axis=1,edge_order=2)
    By=-np.gradient(Az,x,axis=2,edge_order=2)
    Bz=np.gradient(Ay,x,axis=2,edge_order=2)-np.gradient(Ax,y,axis=1,edge_order=2)+B0 # The guide field is added.
   
    Bx[:,:,-1]=Bx[:,:,0] # Periodic boundaries are imposed.
    Bx[:,-1,:]=Bx[:,0,:]
    By[:,:,-1]=By[:,:,0]
    By[:,-1,:]=By[:,0,:]
    Bz[:,:,-1]=Bz[:,:,0]
    Bz[:,-1,:]=Bz[:,0,:]
   
    f.create_dataset("/Step#0/Block/Bx/0",data=Bx)
    f.create_dataset("/Step#0/Block/By/0",data=By)
    f.create_dataset("/Step#0/Block/Bz/0",data=Bz)
   
    Jx=np.gradient(Bz,y,axis=1,edge_order=2)/(4*np.pi)
    Jy=-np.gradient(Bz,x,axis=2,edge_order=2)/(4*np.pi)
    Jz=(np.gradient(By,x,axis=2,edge_order=2)-np.gradient(Bx,y,axis=1,edge_order=2))/(4*np.pi)
   
    # Electron and ion charge densities (quasi-neutrality and homogeneity are assumed).
    rho_0=np.full((int(nzc)+1,int(nyc)+1,int(nxc)+1),-rho_init)
    rho_1=np.full((int(nzc)+1,int(nyc)+1,int(nxc)+1),rho_init)
   
    f.create_dataset("/Step#0/Block/rho_0/0",data=rho_0)
    f.create_dataset("/Step#0/Block/rho_1/0",data=rho_1)
   
    # Electron and ion currents.
    Fx=np.zeros((int(nzc)+1,int(nyc)+1,int(nxc)+1))
    Fy=np.zeros((int(nzc)+1,int(nyc)+1,int(nxc)+1))
    Fz=np.zeros((int(nzc)+1,int(nyc)+1,int(nxc)+1))
    phi=np.zeros((int(nzc)+1,int(nyc)+1,int(nxc)+1))
   
    phases=np.array([random.random()*2*np.pi for n in range(np.shape(modes)[0])]) # Random phases.
    amplitudes=np.array([[random.random(),random.random(),random.random(),random.random()] for n in range(np.shape(modes)[0])]) # Random amplitudes.
   
    for n in range(np.shape(modes)[0]):
    	for iz in range(nzc+1):
    		Fx[iz,:,:]=Fx[iz,:,:]+compressibility*(u_rms*2/k_factor)*amplitudes[n,0]*np.sin(phases[n]+(2*np.pi*modes[n,0]/nxc)*i+(2*np.pi*modes[n,1]/nyc)*j)
    		Fy[iz,:,:]=Fy[iz,:,:]+compressibility*(u_rms*2/k_factor)*amplitudes[n,1]*np.sin(phases[n]+(2*np.pi*modes[n,0]/nxc)*i+(2*np.pi*modes[n,1]/nyc)*j)
    		Fz[iz,:,:]=Fz[iz,:,:]+(u_rms*np.sqrt(8)/k_factor_u)*(1-compressibility)*amplitudes[n,2]*np.sin(phases[n]+(2*np.pi*modes[n,0]/nxc)*i+(2*np.pi*modes[n,1]/nyc)*j)
    		phi[iz,:,:]=phi[iz,:,:]+(u_rms*np.sqrt(8)/k_factor_u)*compressibility*amplitudes[n,3]*np.sin(phases[n]+(2*np.pi*modes[n,0]/nxc)*i+(2*np.pi*modes[n,1]/nyc)*j)
   
    ux=np.gradient(Fz,y,axis=1,edge_order=2)+np.gradient(phi,x,axis=2,edge_order=2)
    uy=-np.gradient(Fz,x,axis=2,edge_order=2)+np.gradient(phi,y,axis=1,edge_order=2)
    uz=np.gradient(Fy,x,axis=2,edge_order=2)-np.gradient(Fx,y,axis=1,edge_order=2)
   
    Jx_0=(Jx-qom_1*((rho_0/qom_0)+(rho_1/qom_1))*ux)/(1-qom_1/qom_0)
    Jy_0=(Jy-qom_1*((rho_0/qom_0)+(rho_1/qom_1))*uy)/(1-qom_1/qom_0)
    Jz_0=(Jz-qom_1*((rho_0/qom_0)+(rho_1/qom_1))*uz)/(1-qom_1/qom_0)
   
    Jx_1=Jx-Jx_0
    Jy_1=Jy-Jy_0
    Jz_1=Jz-Jz_0
   
    Jx_0[:,:,-1]=Jx_0[:,:,0] # Periodic boundaries imposed.
    Jx_0[:,-1,:]=Jx_0[:,0,:]
    Jy_0[:,:,-1]=Jy_0[:,:,0] 
    Jy_0[:,-1,:]=Jy_0[:,0,:]
    Jz_0[:,:,-1]=Jz_0[:,:,0] 
    Jz_0[:,-1,:]=Jz_0[:,0,:]
   
    Jx_1[:,:,-1]=Jx_1[:,:,0] # Periodic boundaries imposed.
    Jx_1[:,-1,:]=Jx_1[:,0,:]
    Jy_1[:,:,-1]=Jy_1[:,:,0] 
    Jy_1[:,-1,:]=Jy_1[:,0,:]
    Jz_1[:,:,-1]=Jz_1[:,:,0] 
    Jz_1[:,-1,:]=Jz_1[:,0,:]
   
    f.create_dataset("/Step#0/Block/Jx_0/0",data=Jx_0)
    f.create_dataset("/Step#0/Block/Jy_0/0",data=Jy_0)
    f.create_dataset("/Step#0/Block/Jz_0/0",data=Jz_0)
   
    f.create_dataset("/Step#0/Block/Jx_1/0",data=Jx_1)
    f.create_dataset("/Step#0/Block/Jy_1/0",data=Jy_1)
    f.create_dataset("/Step#0/Block/Jz_1/0",data=Jz_1)
   
    # Electron and ion pressure tensors (isotropy and homogeneity are assumed).
    Pxx_0=np.full((int(nzc)+1,int(nyc)+1,int(nxc)+1),(qom_0*beta_0*B0**2)/(8*np.pi))+(Jx_0**2)/rho_0
    Pyy_0=np.full((int(nzc)+1,int(nyc)+1,int(nxc)+1),(qom_0*beta_0*B0**2)/(8*np.pi))+(Jy_0**2)/rho_0
    Pzz_0=np.full((int(nzc)+1,int(nyc)+1,int(nxc)+1),(qom_0*beta_0*B0**2)/(8*np.pi))+(Jz_0**2)/rho_0
   
    Pxx_1=np.full((int(nzc)+1,int(nyc)+1,int(nxc)+1),(qom_1*beta_1*B0**2)/(8*np.pi))+(Jx_1**2)/rho_1
    Pyy_1=np.full((int(nzc)+1,int(nyc)+1,int(nxc)+1),(qom_1*beta_1*B0**2)/(8*np.pi))+(Jy_1**2)/rho_1
    Pzz_1=np.full((int(nzc)+1,int(nyc)+1,int(nxc)+1),(qom_1*beta_1*B0**2)/(8*np.pi))+(Jz_1**2)/rho_1
   
    Pxx_0[:,:,-1]=Pxx_0[:,:,0] # Periodic boundaries imposed.
    Pxx_0[:,-1,:]=Pxx_0[:,0,:]
    Pyy_0[:,:,-1]=Pyy_0[:,:,0] 
    Pyy_0[:,-1,:]=Pyy_0[:,0,:]
    Pzz_0[:,:,-1]=Pzz_0[:,:,0] 
    Pzz_0[:,-1,:]=Pzz_0[:,0,:]
   
    Pxx_1[:,:,-1]=Pxx_1[:,:,0] # Periodic boundaries imposed.
    Pxx_1[:,-1,:]=Pxx_1[:,0,:]
    Pyy_1[:,:,-1]=Pyy_1[:,:,0] 
    Pyy_1[:,-1,:]=Pyy_1[:,0,:]
    Pzz_1[:,:,-1]=Pzz_1[:,:,0] 
    Pzz_1[:,-1,:]=Pzz_1[:,0,:]
   
    f.create_dataset("/Step#0/Block/Pxx_0/0",data=Pxx_0)
    f.create_dataset("/Step#0/Block/Pyy_0/0",data=Pyy_0)
    f.create_dataset("/Step#0/Block/Pzz_0/0",data=Pzz_0)
   
    f.create_dataset("/Step#0/Block/Pxx_1/0",data=Pxx_1)
    f.create_dataset("/Step#0/Block/Pyy_1/0",data=Pyy_1)
    f.create_dataset("/Step#0/Block/Pzz_1/0",data=Pzz_1)
   
    # Electric field.
    Ex=By*uz-Bz*uy+(Jy*Bz-Jz*By)/rho_init # Ohm's law with the Hall's term.
    Ey=Bz*ux-Bx*uz+(Jz*Bx-Jx*Bz)/rho_init
    Ez=Bx*uy-By*ux+(Jx*By-Jy*Bx)/rho_init
   
    Ex[:,:,-1]=Ex[:,:,0] # Periodic boundaries are imposed.
    Ex[:,-1,:]=Ex[:,0,:]
    Ey[:,:,-1]=Ey[:,:,0]
    Ey[:,-1,:]=Ey[:,0,:]
    Ez[:,:,-1]=Ez[:,:,0]
    Ez[:,-1,:]=Ez[:,0,:]
   
    f.create_dataset("/Step#0/Block/Ex/0",data=Ex)
    f.create_dataset("/Step#0/Block/Ey/0",data=Ey)
    f.create_dataset("/Step#0/Block/Ez/0",data=Ez)
   
    # rho_avg, B_ext and E_ext (all zero).
    f.create_dataset("/Step#0/Block/rho_avg/0",data=np.zeros((int(nzc)+1,int(nyc)+1,int(nxc)+1)))
    f.create_dataset("/Step#0/Block/Bx_ext/0",data=np.zeros((int(nzc)+1,int(nyc)+1,int(nxc)+1)))
    f.create_dataset("/Step#0/Block/By_ext/0",data=np.zeros((int(nzc)+1,int(nyc)+1,int(nxc)+1)))
    f.create_dataset("/Step#0/Block/Bz_ext/0",data=np.zeros((int(nzc)+1,int(nyc)+1,int(nxc)+1)))
    f.create_dataset("/Step#0/Block/Ex_ext/0",data=np.zeros((int(nzc)+1,int(nyc)+1,int(nxc)+1)))
    f.create_dataset("/Step#0/Block/Ey_ext/0",data=np.zeros((int(nzc)+1,int(nyc)+1,int(nxc)+1)))
    f.create_dataset("/Step#0/Block/Ez_ext/0",data=np.zeros((int(nzc)+1,int(nyc)+1,int(nxc)+1)))
   
    # Lambda, needed with the new-master version.
    f.create_dataset("/Step#0/Block/Lambda/0",data=np.zeros((int(nzc)+1,int(nyc)+1,int(nxc)+1)))
   
    # The number of species is added as an attribute to the group "Step#0".
    f["/Step#0"].attrs.create("nspec",2,dtype=np.int32)
   
    # The h5 file is closed.
    f.close()
     

# Run the simulation for each box
box_numbers = [1, 2,3]  
rho_init = 1 / (4 * np.pi)
B0 = 0.01
inner_radius = 1
outer_radius = 4
B_rms = 0.5 * B0
u_rms = 2 * B0
compressibility = 0.3
for box_number in box_numbers:
     print(box_number)   
     run_simulation(box_number)