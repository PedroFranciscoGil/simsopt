#!/usr/bin/env python

r"""
This script gives suggestions of how to improve the plot of the average B·n/B values for various coil orders and radius multipliers from optimization runs.
It creates a semilogy plot with error bars to show the relationship between coil order and magnetic field quality.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# I'm assuming runs_data is saved optimization runs with standardized file names


def create_bdotn_plot(runs_data, output_file="average_bdotn_vs_order.png"):
    """
    Create a semilogy plot of average  <B_N>/<B> vs coil order with mean and standard deviation
    across different radius multipliers.
    """
    if not runs_data:
        print("No data to work with!")
        return
    
    # Filter out runs where order couldn't be determined
    valid_runs = [run for run in runs_data if run['order'] is not None]

    # Filter out runs where radius multiplier couldn't be determined
    valid_runs = [run for run in valid_runs if run['radius_mult'] is not None]
    
    if not valid_runs:
        print("No runs with radius multiplier or order information found!")
        return
    
    # Group data by order first, then by radius multiplier
    order_radius_stats = {} # This holds all of the average B dot n /B data, sorted by order and radius multipliers
    for run in valid_runs:
        order = run['order']
        radius_mult = run['radius_mult']
        
        if order not in order_radius_stats:
            order_radius_stats[order] = {}
        
        if radius_mult not in order_radius_stats[order]:
            order_radius_stats[order][radius_mult] = []
        
        order_radius_stats[order][radius_mult].append(run['bdotn']) # Adding to lists
    
    # Calculate statistics for each order across radius multipliers
    orders = sorted(order_radius_stats.keys()) # Extracting orders from the order_radius_stats and sorting them in numerical order 
    means = [] 
    stds = []
    counts = []
    radius_counts = []

    # Loop over orders
    for order in orders:
        radius_means = []
        radius_stds = []
        radius_sample_counts = []
        
        print(f"\nOrder {order}:")
        
        # Calculate mean and std for each radius multiplier within this specific order
        for radius_mult in sorted(order_radius_stats[order].keys()):
            values = order_radius_stats[order][radius_mult]
            radius_mean = np.mean(values)
            radius_std = np.std(values)
            radius_count = len(values) # Keeping track how many different radius multipliers were used for this order
            radius_means.append(radius_mean)
            radius_stds.append(radius_std)
            radius_sample_counts.append(radius_count) # Number of samples for this specific multiplier loop, should be 1 for now
            
            print(f"  R={radius_mult}: Mean={radius_mean:.2e}, Std={radius_std:.2e}, Count={radius_count}")
        
        # Now average across radius multipliers for this order
        total_mean = np.mean(radius_means) # Mean
        total_std = np.std(radius_means)  # Standard deviation
        total_samples = sum(radius_sample_counts)
        num_radius_values = len(radius_means)
        
        means.append(total_mean)
        stds.append(total_std)
        counts.append(total_samples) # total number of samples for this order
        radius_counts.append(num_radius_values)
        
     
 
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Semilogy plot with error bars
    plt.errorbar(orders, means, yerr=stds, fmt='bo-', linewidth=2, markersize=8,
                 markerfacecolor='blue', markeredgecolor='black', markeredgewidth=1,
                 capsize=5, capthick=2, elinewidth=2, label='Mean plus or minus Std Dev')
    
    # Add threshold line
    threshold = 1e-3  # Adjust this to threshold
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
            label=f'Threshold ({threshold:.0e})')

    # Make y-axis logarithmic 
    plt.yscale('log')

    # Label the plot
    plt.xlabel('Coil Order', fontsize=14)
    plt.ylabel('<B_N>/<B>', fontsize=14)
    plt.title('Magnetic Field Quality vs Coil Order averaged over Radius Multipliers', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
 
    
    # Set axis limits with padding
    plt.xlim(min(orders) - 1, max(orders) + 1)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    
    # Show the plot
    plt.show()
    
    # Find best and worst orders (by mean)
    best_order= np.argmin(means) # Smallest average <B_N>/<B>
    worst_order= np.argmax(means) # Largest average <B_N>/<B> 
    
    print(f"\nBest order so far: {orders[best_order]} (Mean <B_N>/<B> = {means[best_order]:.2e})")
    print(f"Worst order so far: {orders[worst_order]} (Mean <B_N>/<B> = {means[worst_order]:.2e})")
    
    return orders, means, stds, counts, radius_counts

