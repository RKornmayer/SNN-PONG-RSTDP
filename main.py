from brian2 import *
from brain import Brain_rstdp
from game import PongGame
from analysis import analyze_simulation
import h5py

topology = [100, 125]  # 100 input-encoding neurons, 100 output-encoding + 25 inhibitory neurons
brain_sim = Brain_rstdp(topology)

game_instance = PongGame(width=640, height=480, headless=False)

brain_sim.create_game_loop(game_instance, visual_mode=True)

print("Starting SNN Pong...")

try:
    brain_sim.net.run(6000*second) 
except KeyboardInterrupt:
    print("\nSimulation manually terminated.")

# 1. Flush the regular telemetry buffers to disk
brain_sim.telemetry.flush_to_disk()

# 2. Append spike data manually
print("Saving spike data...")
with h5py.File("brain_telemetry.h5", "a") as f:
    for mon in brain_sim.spike_monitors:
        grp_name = f"Spikes_{mon.source.name}"
        # Remove the group from a previous run, if present
        if grp_name in f:
            del f[grp_name]
        
        grp = f.create_group(grp_name)
        grp.create_dataset("i", data=np.array(mon.i))
        grp.create_dataset("t", data=np.array(mon.t / second))

print("Generating plots...")
analyze_simulation("brain_telemetry.h5")