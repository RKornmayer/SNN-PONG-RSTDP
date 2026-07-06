import h5py
import numpy as np
from brian2 import second

class BrainTelemetry:
    def __init__(self, filename, config, chunk_size=200):
        """
        High-performance HDF5 logger with LZF compression.
        """
        self.filename = filename
        self.config = config
        self.chunk_size = chunk_size
        
        self.buffers = {'time': []}
        self.step_counter = 0

        self._initialize_hdf5_and_buffers()

    def _initialize_hdf5_and_buffers(self):
        with h5py.File(self.filename, 'w') as f:
            # Create 1D time axis
            f.create_dataset('time', shape=(0,), maxshape=(None,), dtype='float32', chunks=True)
            
            # Create basic datasets with LZF compression
            f.create_dataset('distance', shape=(0,), maxshape=(None,), dtype='float32', chunks=True, compression='lzf')
            f.create_dataset('reward', shape=(0,), maxshape=(None,), dtype='float32', chunks=True, compression='lzf')
            f.create_dataset('v_current', shape=(0,), maxshape=(None,), dtype='float32', chunks=True, compression='lzf')
            
            # --- Level dataset ---
            f.create_dataset('level', shape=(0,), maxshape=(None,), dtype='int32', chunks=True, compression='lzf')
            
            self.buffers['distance'] = []
            self.buffers['reward'] = []
            self.buffers['v_current'] = []
            self.buffers['level'] = []
            
            for group_name, setup in self.config.items():
                brian_obj = setup['object']
                if brian_obj is None: continue
                
                N = int(len(brian_obj))
                
                for var_name in setup['vars']:
                    dataset_name = f"{group_name}/{var_name}"
                    self.buffers[dataset_name] = []
                    
                    # Create datasets with chunking and fast LZF compression
                    f.create_dataset(
                        dataset_name, 
                        shape=(0, N), 
                        maxshape=(None, N), 
                        dtype='float32', 
                        chunks=(self.chunk_size, N), # Time blocks x Neurons/Synapses
                        compression='lzf' 
                    )

    # Records one telemetry sample; `level` tags it with the current curriculum stage
    def gather_data(self, t, current_dist=0.0, reward=0.0, v_current=0.0, level=1):
        self.buffers['time'].append(t / second)
        self.buffers['distance'].append(current_dist)
        self.buffers['reward'].append(reward)
        self.buffers['v_current'].append(v_current)
        self.buffers['level'].append(level)
        
        for group_name, setup in self.config.items():
            brian_obj = setup['object']
            if brian_obj is None: continue
            
            for var_name in setup['vars']:
                dataset_name = f"{group_name}/{var_name}"
                
                # Extract values from Brian2 and strip units
                val_array = np.array(getattr(brian_obj, var_name)) 
                self.buffers[dataset_name].append(val_array)
        
        self.step_counter += 1
        if self.step_counter >= self.chunk_size:
            self.flush_to_disk()

    def flush_to_disk(self):
        if len(self.buffers['time']) == 0: return

        with h5py.File(self.filename, 'a') as f:
            for key, data_list in self.buffers.items():
                ds = f[key]
                curr_len = ds.shape[0]
                add_len = len(data_list)
                
                # Resize dataset and append current buffer data
                ds.resize((curr_len + add_len, *ds.shape[1:]))
                ds[curr_len:] = data_list
                
                self.buffers[key].clear()
                
        self.step_counter = 0