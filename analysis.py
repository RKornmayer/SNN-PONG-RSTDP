import h5py
import numpy as np
import matplotlib.pyplot as plt

def analyze_simulation(h5_path="brain_telemetry.h5"):
    try:
        f = h5py.File(h5_path, 'r')
    except FileNotFoundError:
        print(f"Error: telemetry file '{h5_path}' not found.")
        return

    t = np.array(f['time'])
    if len(t) == 0:
        print("The telemetry file is empty.")
        f.close()
        return

    # Helper to fetch spike times for a specific neuron
    def get_spikes(area_name, neuron_idx):
        spike_grp = f.get(f"Spikes_{area_name}")
        if spike_grp is not None:
            i_arr = np.array(spike_grp['i'])
            t_arr = np.array(spike_grp['t'])
            return t_arr[i_arr == neuron_idx]
        return []

    # --- Plot 1: Cellular traces & conductances ---
    fig1, axes1 = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # 1. Sensory layer (excitatory)
    if 'Area_0_Sensory_Exc/v' in f:
        v_sens = np.array(f['Area_0_Sensory_Exc/v'])
        v_thresh_sens = np.array(f['Area_0_Sensory_Exc/v_thresh'])
        rand_idx = np.random.randint(0, v_sens.shape[1])
        
        spikes = get_spikes("Area0_Sensory_Exc", rand_idx)
        for st in spikes:
            axes1[0].axvline(st, color='gray', alpha=0.4, linestyle='-', linewidth=1)
            
        axes1[0].plot(t, v_sens[:, rand_idx], label='v', color='#1f77b4', zorder=2)
        axes1[0].plot(t, v_thresh_sens[:, rand_idx], label='v_thresh', color='#d62728', zorder=3)
        axes1[0].set_title(f"Sensory Layer (Excitatory) - Neuron {rand_idx}")
        axes1[0].set_ylabel("Voltage (V)")
        axes1[0].legend(loc="lower right")
        axes1[0].grid(True, linestyle='--', alpha=0.6)

    # 2. Motor layer (excitatory)
    rand_idx_mot = 0
    if 'Area_1_Motor_Exc/v' in f:
        v_mot = np.array(f['Area_1_Motor_Exc/v'])
        v_thresh_mot = np.array(f['Area_1_Motor_Exc/v_thresh'])
        g_exc_mot = np.array(f['Area_1_Motor_Exc/g_exc'])
        g_inh_mot = np.array(f['Area_1_Motor_Exc/g_inh'])
        
        rand_idx_mot = np.random.randint(0, v_mot.shape[1])
        spikes = get_spikes("Area1_Motor_Exc", rand_idx_mot)
        for st in spikes:
            axes1[1].axvline(st, color='gray', alpha=0.4, linestyle='-', linewidth=1)
        
        axes1[1].plot(t, v_mot[:, rand_idx_mot], label='v', color='#1f77b4', zorder=2)
        axes1[1].plot(t, v_thresh_mot[:, rand_idx_mot], label='v_thresh', color='#d62728', zorder=3)
        axes1[1].set_title(f"Motor Layer (Excitatory) - Neuron {rand_idx_mot}")
        axes1[1].set_ylabel("Voltage (V)")
        axes1[1].legend(loc="lower right")
        axes1[1].grid(True, linestyle='--', alpha=0.6)

        # 3. Motor layer conductances
        axes1[2].plot(t, g_exc_mot[:, rand_idx_mot] * 1e9, label='g_exc (excitation)', color='#2ca02c')
        axes1[2].plot(t, g_inh_mot[:, rand_idx_mot] * 1e9, label='g_inh (inhibition)', color='#9467bd')
        axes1[2].set_title(f"Motor Neuron {rand_idx_mot} Conductances")
        axes1[2].set_ylabel("Conductance (nS)")
        axes1[2].legend(loc="upper left")
        axes1[2].grid(True, linestyle='--', alpha=0.6)

    # 4. Motor layer (inhibitory)
    if 'Area_1_Motor_Inh/v' in f:
        v_inh = np.array(f['Area_1_Motor_Inh/v'])
        rand_idx_inh = np.random.randint(0, v_inh.shape[1])
        
        spikes = get_spikes("Area1_Motor_Inh", rand_idx_inh)
        for st in spikes:
            axes1[3].axvline(st, color='gray', alpha=0.4, linestyle='-', linewidth=1)
            
        axes1[3].plot(t, v_inh[:, rand_idx_inh], label='v (inhibitory)', color='#ff7f0e', zorder=2)
        axes1[3].axhline(-0.045, color='red', linestyle='--', label='threshold (-45mV)', zorder=3)
        axes1[3].set_title(f"Motor Layer (Inhibitory) - Neuron {rand_idx_inh}")
        axes1[3].set_ylabel("Voltage (V)")
        axes1[3].legend(loc="lower right")
        axes1[3].grid(True, linestyle='--', alpha=0.6)

    axes1[-1].set_xlabel("Time (s)")
    fig1.tight_layout()

    # --- Plot 2: Synaptic weight heatmaps ---
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    if 'motor_synapses/w' in f:
        w_rstdp = np.array(f['motor_synapses/w'])
        im1 = axes2[0].imshow(w_rstdp.T, aspect='auto', cmap='viridis', origin='lower', extent=[t[0], t[-1], 0, w_rstdp.shape[1]])
        fig2.colorbar(im1, ax=axes2[0], label='Weight (nS)')
        axes2[0].set_title("Accelerator: R-STDP (Sensor -> Motor)")
        axes2[0].set_ylabel("Synapse Index")

    if 'istdp_synapses/w' in f:
        w_istdp = np.array(f['istdp_synapses/w'])
        im2 = axes2[1].imshow(w_istdp.T, aspect='auto', cmap='plasma', origin='lower', extent=[t[0], t[-1], 0, w_istdp.shape[1]])
        fig2.colorbar(im2, ax=axes2[1], label='Weight (nS)')
        axes2[1].set_title("Brake: iSTDP (Inhibitory -> Motor)")
        axes2[1].set_ylabel("Synapse Index")

    axes2[-1].set_xlabel("Time (s)")
    fig2.tight_layout()

    # --- Plot 3: Macroscopic trend (with per-level regression) ---
    dist = np.array(f['distance'])
    dopamine = np.array(f['dopamine/D'])[:, 0]

    # Load the level array if present
    has_levels = False
    if 'level' in f:
        levels = np.array(f['level'])
        has_levels = True
    
    window = min(50, len(dist) // 10) 
    if window > 0:
        dist_smooth = np.convolve(dist, np.ones(window)/window, mode='valid')
        t_smooth = t[window-1:]
    else:
        dist_smooth, t_smooth = dist, t

    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig3.suptitle("System Performance: Tracking Error vs. Reward")

    # --- Subplot 1: High-resolution data ---
    color_dist = '#d62728'
    ax3a.set_ylabel('Tracking Error (Distance)', color=color_dist)
    ax3a.plot(t_smooth, dist_smooth, color=color_dist, label='Distance')
    ax3a.tick_params(axis='y', labelcolor=color_dist)

    ax3a_twin = ax3a.twinx()
    color_dop = '#ff7f0e'
    ax3a_twin.set_ylabel('Dopamine (Reward)', color=color_dop)
    ax3a_twin.plot(t, dopamine, color=color_dop, label='Dopamine', alpha=0.7)
    ax3a_twin.tick_params(axis='y', labelcolor=color_dop)
    ax3a.set_title("High-Resolution Data")

    # --- Subplot 2: 5-second bins ---
    chunk_size = 5.0
    max_time = t[-1] if len(t) > 0 else 0
    num_chunks = int(np.ceil(max_time / chunk_size))
    
    chunk_times = []
    chunk_err_mean = []
    chunk_err_std = []
    chunk_rew_mean = []
    
    # Track the dominant level within each 5-second bin
    chunk_levels = []

    for i in range(num_chunks):
        t_start = i * chunk_size
        t_end = (i + 1) * chunk_size
        mask = (t >= t_start) & (t < t_end)
        
        if np.any(mask):
            chunk_times.append(t_start + chunk_size / 2.0)
            chunk_err_mean.append(np.mean(dist[mask]))
            chunk_err_std.append(np.std(dist[mask]))
            chunk_rew_mean.append(np.mean(dopamine[mask]))
            
            if has_levels:
                # Most common level within this bin
                dominant_level = int(np.median(levels[mask]))
                chunk_levels.append(dominant_level)

    chunk_times = np.array(chunk_times)
    chunk_err_mean = np.array(chunk_err_mean)
    chunk_err_std = np.array(chunk_err_std)
    chunk_rew_mean = np.array(chunk_rew_mean)
    
    if has_levels:
        chunk_levels = np.array(chunk_levels)

    color_err_mean = '#b71c1c'  
    color_err_std = '#ef9a9a'   
    
    ax3b.plot(chunk_times, chunk_err_mean, color=color_err_mean, marker='o', linestyle='-', label="Mean Error")
    
    lower_bound = np.clip(chunk_err_mean - chunk_err_std, 0, None)
    ax3b.fill_between(chunk_times, lower_bound, chunk_err_mean + chunk_err_std, 
                      color=color_err_std, alpha=0.5, label="Std Error")
    
    # --- Per-level linear regression trend lines ---
    if has_levels and len(chunk_times) > 2:
        unique_levels = np.unique(chunk_levels)

        for lvl in unique_levels:
            # Bins belonging to this level
            lvl_mask = (chunk_levels == lvl)
            x_lvl = chunk_times[lvl_mask]
            y_lvl = chunk_err_mean[lvl_mask]

            # Need at least two points to fit a line
            if len(x_lvl) > 1:
                # First-degree regression (y = mx + b)
                m, b = np.polyfit(x_lvl, y_lvl, 1)

                # Evaluate the fitted line
                fit_y = m * x_lvl + b

                # Draw the trend line (black, dashed)
                ax3b.plot(x_lvl, fit_y, color='black', linestyle='--', linewidth=2, zorder=5)

                # Annotate the trend line with its start/end values
                mid_idx = len(x_lvl) // 2
                text_y = max(fit_y) + 0.1

                start_val = fit_y[0]
                end_val = fit_y[-1]

                # Format with two decimal places, e.g. "0.40 -> 0.20"
                ax3b.text(x_lvl[mid_idx], text_y, f"Lvl {lvl} Trend: {start_val:.2f} -> {end_val:.2f}",
                          ha='center', va='bottom', fontsize=9,
                          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Plot the mean dopamine trend
    ax3b_twin = ax3b.twinx()
    color_dop_mean = '#f57c00' 
    ax3b_twin.plot(chunk_times, chunk_rew_mean, color=color_dop_mean, marker='s', linestyle='-', label="Mean Dopamine")
    ax3b_twin.set_ylabel("Mean Dopamine", color=color_dop_mean)
    ax3b_twin.tick_params(axis='y', labelcolor=color_dop_mean)

    ax3b.set_zorder(ax3b_twin.get_zorder() + 1)
    ax3b.patch.set_visible(False)

    ax3b.set_xlabel('Time (s)')
    ax3b.set_ylabel("Mean Tracking Error", color=color_err_mean)
    ax3b.tick_params(axis='y', labelcolor=color_err_mean)
    ax3b.set_title("Macroscopic Trend (5-Second Bins) with Per-Level Regression")

    fig3.tight_layout()

    # --- Plot 4: Spike raster (motor layer) ---
    spike_grp = f.get("Spikes_Area1_Motor_Exc")
    if spike_grp is not None:
        i_arr = np.array(spike_grp['i'])
        t_arr = np.array(spike_grp['t'])

        fig4, ax_raster = plt.subplots(figsize=(14, 6))
        # Small marker size keeps the raster crisp
        ax_raster.scatter(t_arr, i_arr, s=2, color='black', alpha=0.6, marker='.')
        ax_raster.set_title("Macroscopic Structure: Spike Raster Plot (Motor Layer)")
        ax_raster.set_xlabel("Time (s)")
        ax_raster.set_ylabel("Neuron Index (0 to 99)")

        # Light grid to hint at chunk boundaries
        ax_raster.set_ylim(-2, 102)
        ax_raster.set_yticks([0, 25, 50, 75, 100])
        ax_raster.grid(True, linestyle='--', alpha=0.3)
        fig4.tight_layout()

    plt.show(block=True)
    f.close()