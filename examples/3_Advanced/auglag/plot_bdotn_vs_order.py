#!/usr/bin/env python

r"""
This script gives suggestions of how to improve the plotting of the <B_N>/<B> values for various coil orders and radius multipliers from optimization runs.
It creates a semilogy plot with standard deviation error bars to show the relationship between coil order and magnetic field quality. It is beginning work on post-processing all of the runs, identifying
the order and radius multipliers, and then extracting the  <B_N>/<B> value. The data structure I want looks like, as an example: 
data = [
        {'filename': 'qh_1_0.05_ncoils4_curvature1_msc0.1_force10_flux1e-15_length160_cc0.8_cs1', 'order': 1, 'radius_mult': 0.05, 'bdotn': 1e-2},
        ]

For now, this is configured for qh reactor scale but can be easily adapted. 
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from simsopt.geo import SurfaceRZFourier
from simsopt import load
    



def compute_bdotn_from_json(json_file, order, radius_mult):
    """
    Compute <B·n>/<B> value from JSON file
    
    Parameters:
    -----------
    json_file : Path to the biot_savart_*.json file 
    order : Coil order
    radius_mult : Radius multiplier
        
    Returns:
    --------
    float or None
       <B·n>/<B> 
    """
  
    
    # Load the BiotSavart object from JSON
    Bfield = load(json_file)
    nfp = 1 
    stellsym = True

    nphi = 256  
    ntheta = 256  
    
    vmec_input_file = None
    json_dir = os.path.dirname(json_file)
    
    # Look for VMEC input
    possible_vmec = [
        "input.LandremanPaul2021_QH_reactorScale_lowres"
    ]
    
    for vmec_name in possible_vmec:
        test_path = os.path.join(json_dir, vmec_name) # This is the path to the VMEC input file
        if os.path.exists(test_path): # If the VMEC input file exists, then set it in this variable
            vmec_input_file = test_path
            break
    
    if vmec_input_file:
        # Create surface from VMEC input
        quadpoints_phi = np.linspace(0, 1, nphi, endpoint=True)
        quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
        
        try:
            surface = SurfaceRZFourier.from_vmec_input(vmec_input_file, 
                                                       quadpoints_phi=quadpoints_phi, 
                                                       quadpoints_theta=quadpoints_theta)
        except Exception as e:
            print(f"Failed to create surface from VMEC input: {e}")
            vmec_input_file = None

    # If no VMEC file, create manual surface
    if not vmec_input_file:
        # Create the surface
        quadpoints_phi = np.linspace(0, 1, nphi, endpoint=True)
        quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
        
        surface = SurfaceRZFourier(nfp=nfp, stellsym=stellsym, mpol=order, ntor=order,
                                   quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta)
        
        # Set basic shape -- this part needs to be fixed, in the case when a VMEC input is not found
        surface.rc[0, 0] = radius_mult
        surface.zs[0, 0] = 0.0


    
    # Get the surface coordinates
    eval_points = surface.gamma()
    # Flatten the 3D array as in run_post_processing.py
    eval_points_flat = eval_points.reshape((-1, 3))
    Bfield.set_points(eval_points_flat)
    
    # Get the B-field data from BiotSavart
    B_field_raw = Bfield.B()
    
    # Reshape to 2D grid
    B_field= B_field_raw.reshape((nphi, ntheta, 3))
    surface_normals = surface.unitnormal().reshape((nphi, ntheta, 3))
    
    # Compute B·n component  
    B_dot_n = np.sum(B_field * surface_normals, axis=2)
    
    # Calculate <B·n>/<B> 
    BdotN = np.mean(np.abs(B_dot_n))
    avg_B_magnitude = np.mean(Bfield.AbsB())
    BdotN_over_B = BdotN / avg_B_magnitude
    
    return float(BdotN_over_B)


def process_optimization_data(output_directory="./output_paper", file_pattern="*.8_cs1"):
    """
    Load and process optimization data 
    
    Parameters:
    -----------
    output_directory : Directory containing the optimization result directories
    file_pattern : Pattern to match and grab relevant directories
        
    Returns:
    --------
    tuple: (raw_data, processed_data)
    """
    raw_data = []
    
    # Find all matching directories
    search_pattern = os.path.join(output_directory, file_pattern)
    matching_dirs = glob.glob(search_pattern)
    
    
    for dir_path in matching_dirs:
        # Extract directory name
        dir_name = os.path.basename(dir_path)
        
        # Extract order and radius multiplier from filename
        order_match = re.search(r'qh_(\d+)_', dir_name) 
        radius_match = re.search(r'qh_\d+_([\d.]+)_', dir_name)
        
        if not (order_match and radius_match):
            print(f"Could not find order and radius multiplier in {dir_name}") #Problem with extracting data
            continue
            
        order = int(order_match.group(1))
        radius_mult = float(radius_match.group(1))
        
        # Look for JSON files in the directory
        json_files = glob.glob(os.path.join(dir_path, "biot_savart_*.json"))
        
        # Skip this directory if no JSON files found
        if not json_files:
            print(f"No biot_savart_*.json files found in {dir_name}")
            continue
        
        # Use the first JSON file found
        json_file = json_files[0]
        
        # Compute the <B·n>/<B> value from JSON file
        bdotn_value = compute_bdotn_from_json(json_file, order, radius_mult)
        
        if bdotn_value is not None:
            file_data = {
                'filename': dir_name,
                'order': order,
                'radius_mult': radius_mult,
                'bdotn': bdotn_value
            }
            raw_data.append(file_data)
        else:
            print("Could not obtain <B_N>/<B> value from JSON file")

    
    # Group data by order first, then by radius multiplier
    order_radius_stats = {}
    for run in raw_data:
        order = run['order']
        radius_mult = run['radius_mult']
        
        if order not in order_radius_stats:
            order_radius_stats[order] = {}
        
        if radius_mult not in order_radius_stats[order]:
            order_radius_stats[order][radius_mult] = []
        
        order_radius_stats[order][radius_mult].append(run['bdotn'])
    
    # Calculate statistics for each order across radius multipliers
    orders = sorted(order_radius_stats.keys())
    means, stds, counts, radius_counts = [], [], [], []

    for order in orders:
        print(f"\nOrder {order}:")
        
        # Calculate statistics for each radius multiplier
        radius_stats = []
        for radius_mult in sorted(order_radius_stats[order].keys()):
            values = order_radius_stats[order][radius_mult]
            mean_val = np.mean(values)
            std_val = np.std(values)
            radius_stats.append((mean_val, std_val, len(values)))
        
        # Average across radius multipliers for this order
        radius_means = [stat[0] for stat in radius_stats]
        means.append(np.mean(radius_means))
        stds.append(np.std(radius_means))
        counts.append(sum(stat[2] for stat in radius_stats))
        radius_counts.append(len(radius_stats))
    
    # Create processed dictionary
    processed_data = {
        'orders': orders,
        'means': means,
        'stds': stds,
        'counts': counts,
        'radius_counts': radius_counts,
        'order_radius_stats': order_radius_stats
    }
    
    return raw_data, processed_data


def create_bdotn_plot(processed_data, output_file="average_bdotn_vs_order.png"):
    """
    Create a semilogy plot of average <B_N>/<B> vs coil order with mean and standard deviation
    across different radius multipliers.
    """

    
    orders = processed_data['orders']
    means = processed_data['means']
    stds = processed_data['stds']
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Semilogy plot with error bars
    plt.errorbar(orders, means, yerr=stds, fmt='bo-', linewidth=2, markersize=8,
                 markerfacecolor='blue', markeredgecolor='black', markeredgewidth=1,
                 capsize=5, capthick=2, elinewidth=2, label='Mean ± Std Dev')
    
    # Add threshold line
    threshold = 1e-3
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold:.0e})')
    plt.yscale('log')
    plt.xlabel('Coil Order', fontsize=14)
    plt.ylabel('<B_N>/<B>', fontsize=14)
    plt.title('Magnetic Field Quality vs Coil Order averaged over Radius Multipliers', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlim(min(orders) - 1, max(orders) + 1)
    
    # Save and show
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    plt.show()
    
    # Find best and worst orders
    best_idx, worst_idx = np.argmin(means), np.argmax(means)
    print(f"\nBest order: {orders[best_idx]} ( <B_N>/<B> = {means[best_idx]:.2e})")
    print(f"Worst order: {orders[worst_idx]} ( <B_N>/<B> = {means[worst_idx]:.2e})")
    
    return orders, means, stds


def main():
    """
    Main function
    """

    output_dir = "./output_paper"
    file_pattern = "*.8_cs1"
    output_file = "average_bdotn_vs_order.png"
    
    # Load and plot
    raw_data, processed_data = process_optimization_data(output_dir, file_pattern)
    create_bdotn_plot(processed_data, output_file)
    

if __name__ == "__main__":
    main()


