from brian2 import *
import numpy as np
import pygame
from telemitry import BrainTelemetry
import sys

class Brain_rstdp:
    def __init__(self, topology):
        defaultclock.dt = 0.5*ms

        self.topology = topology
        
        # --- Parameters ---

        # Neural physics
        self.Cm = 200*pF
        self.g_leak = 10*nS
        self.tau_m = self.Cm / self.g_leak
        # Reversal potentials
        self.E_leak = -70*mV
        self.E_exc = 0*mV
        self.E_inh = -80*mV
        # Synaptic time constants (conductance decay)
        self.tau_exc = 5*ms
        self.tau_inh = 10*ms
        self.tau_D = 25*ms
        self.tau_plus = 20*ms
        self.tau_minus = 20*ms
        # Thresholds and refractory periods
        self.v_reset = -65*mV
        self.v_thresh_base = -50*mV
        self.refractory_exc = 5*ms
        self.refractory_inh = 2*ms
        # Noise amplitude
        self.sigma_noise = 7*mV
        # Adaptive thresholds (homeostasis)
        self.tau_thresh = 3*second
        self.thresh_inc = 0.05*mV
        # Learning rates and weight bounds
        self.w_target = 6.0*nS
        self.w_max = 8*nS
        self.w_min = 1.0*nS
        self.learning_rate = 0.2
        self.decay_factor = 0.0005
        # Eligibility trace
        self.A_plus = 0.1
        self.A_minus = -0.01
        self.tau_c = 50*ms
        self.c_max = 5.0

        # --- Equations ---

        self.eqs_neuron = '''
        dv/dt = (g_leak * (E_leak - v) + g_exc * (E_exc - v) + g_inh * (E_inh - v)) / Cm + sigma_noise * xi * tau_m**-0.5 : volt (unless refractory)
        
        dg_exc/dt = -g_exc / tau_exc : siemens
        dg_inh/dt = -g_inh / tau_inh : siemens
        
        dv_thresh/dt = (v_thresh_base - v_thresh) / tau_thresh : volt
        
        v_thresh_base : volt
        sigma_noise : volt
        '''
        self.eqs_dopamine = '''
        dD/dt = -D / tau_D : 1
        davg_hit_rate/dt = -avg_hit_rate / (30*second) : 1
        gain = 1.0 / (1.0 + 10.0 * avg_hit_rate) : 1
        '''
        self.eqs_rstdp = '''
        w : siemens
        d_syn : 1 (linked)
        dopa_idx : integer (constant)
        tau_c : second (shared)
        dc/dt = -c / tau_c : 1 (clock-driven)
        dapre/dt = -apre / tau_plus : 1 (event-driven)
        dapost/dt = -apost / tau_minus : 1 (event-driven)
        '''
        self.eqs_istdp = '''
        dapre_inh/dt = -apre_inh / tau_istdp : 1 (event-driven)
        dapost_inh/dt = -apost_inh / tau_istdp : 1 (event-driven)
        alpha_istdp : 1 
        dw/dt = -w / tau_forget : siemens (clock-driven) 
        '''

        # --- Network construction ---

        self.neuron_groups = []
        self.exc_groups = []
        self.inh_groups = []
        self.synapse_groups = []
        self.monitors = []

        self._create_neurons()
        self._create_input_and_reward_groups()
        self._create_synapses()
        self._create_monitors()
        self._update_weights()

        self.net = Network(self.neuron_groups, self.synapse_groups, 
                           self.sensor_input, self.input_syn, 
                           self.dopamine_group, self.reward_group, self.reward_synapse, 
                           self.monitors, self.hit_success_group, self.hit_success_syn, 
                           self.learn_op)

    # --- Methods ---

    def _get_randomness(self, value, size, percent=0.1):
        # Random per-neuron perturbation, used to add heterogeneity to parameters
        random_factor = (rand(size) - 0.5) * 2
        return value * percent * random_factor

    def _get_area_properties(self, area_idx):
        # Maps an area index to a label; this topology only has input and output areas
        if area_idx == 0: return "Sensory"
        else: return "Motor"

    def _create_neurons(self):
        # Namespace of constants Brian2 resolves against when evaluating the equations
        neuron_namespace = {
            'Cm': self.Cm,
            'g_leak': self.g_leak,
            'E_leak': self.E_leak,
            'E_exc': self.E_exc,
            'E_inh': self.E_inh,
            'tau_exc': self.tau_exc,
            'tau_inh': self.tau_inh,
            'tau_thresh': self.tau_thresh,
            'tau_m': self.tau_m,  
            'thresh_inc': self.thresh_inc, 
            'v_reset': self.v_reset
        }

        for a, area_size in enumerate(self.topology):
            area_type = self._get_area_properties(a)

            # Split into excitatory/inhibitory counts; the input layer is exc-only
            # so it can directly represent the sensory encoding
            if a != 0:
                n_exc = int(area_size * 0.8) # (80/20)
                n_inh = area_size - n_exc
                if n_exc < 1: n_exc = 1
                if n_inh < 1: n_inh = 1
            else:
                n_exc = area_size
                n_inh = 0

            # --- Excitatory neurons ---
            name_str_exc = f"Area{a}_{area_type}_Exc"
            # Separate reset string: clip() keeps the adaptive threshold from drifting permanently out of reach
            reset_str = 'v = v_reset; v_thresh = clip(v_thresh + thresh_inc, v_thresh_base, -45*mV)'
            exc_neurons = NeuronGroup(n_exc, self.eqs_neuron,
                                    threshold='v > v_thresh',
                                    reset=reset_str,
                                    refractory=self.refractory_exc,
                                    method='euler',
                                    name=name_str_exc,
                                    namespace=neuron_namespace) 
            
            # Initialize state
            exc_neurons.v = self.E_leak
            exc_neurons.v_thresh_base = self.v_thresh_base + self._get_randomness(self.v_thresh_base, n_exc, 0.05)
            exc_neurons.v_thresh = self.v_thresh_base
            exc_neurons.sigma_noise = self.sigma_noise
            
            self.exc_groups.append(exc_neurons)
            self.neuron_groups.append(exc_neurons)

            # --- Inhibitory neurons ---
            if n_inh > 0:
                name_str_inh = f"Area{a}_{area_type}_Inh"
                
                inh_neurons = NeuronGroup(n_inh, self.eqs_neuron,
                                        threshold='v > -45*mV', 
                                        reset='v = v_reset',
                                        refractory=self.refractory_inh,
                                        method='euler',
                                        name=name_str_inh,
                                        namespace=neuron_namespace)
                
                inh_neurons.v = self.E_leak
                inh_neurons.v_thresh = -45*mV
                inh_neurons.v_thresh_base = -45*mV
                inh_neurons.sigma_noise = self.sigma_noise * 0.5 
                
                self.inh_groups.append(inh_neurons)
                self.neuron_groups.append(inh_neurons)
            else:
                self.inh_groups.append(None)

    def _create_input_and_reward_groups(self):
        eqs_sensor = '''
        dv_sensor/dt = rate : 1
        rate : Hz
        '''
        self.sensor_input = NeuronGroup(self.topology[0], eqs_sensor, 
                                        threshold='v_sensor > 1', 
                                        reset='v_sensor = 0', 
                                        method='euler', 
                                        name='sensor_input')
        self.sensor_input.rate = 0 * Hz

        # High conductance (60 nS) reliably forces the target neuron to spike
        self.input_syn = Synapses(self.sensor_input, self.exc_groups[0],
                                on_pre='g_exc_post += 60*nS', 
                                name='input_synapse')
        
        N_src = self.topology[0]
        N_tgt = self.exc_groups[0].N
        self.input_syn.connect(j=f'int(i * {N_tgt} / {N_src})')

        # --- Dopamine and reward ---
        dopa_namespace = {'tau_D': self.tau_D}
        # Single neuron used purely to integrate the dopamine equations
        self.dopamine_group = NeuronGroup(1, self.eqs_dopamine, name='Dopamine_Group',
                                        method='exact', namespace=dopa_namespace)
        self.dopamine_group.D = 0
        
        self.reward_group = SpikeGeneratorGroup(1, indices=[], times=[]*ms, name='Reward_Input')
        self.reward_synapse = Synapses(self.reward_group, self.dopamine_group, on_pre='D_post += 1.0')
        self.reward_synapse.connect(i=0, j=0)

        self.hit_success_group = SpikeGeneratorGroup(1, indices=[], times=[]*ms, name='Hit_Success_Input')
        self.hit_success_syn = Synapses(self.hit_success_group, self.dopamine_group, on_pre='avg_hit_rate += 0.5')
        self.hit_success_syn.connect(i=0, j=0)

    def _create_synapses(self):
        # Namespace for R-STDP synapses (Exc -> Exc)
        syn_rstdp_namespace = {
            'tau_plus': self.tau_plus,   
            'tau_minus': self.tau_minus,  
            'A_plus': self.A_plus,
            'A_minus': self.A_minus,
            'w_max': self.w_max,
            'w_min': self.w_min,
            'learning_rate': self.learning_rate,
            'nS': nS,     
            'clip': clip,   
            'c_max': self.c_max
        }
        # Namespace for iSTDP synapses (Inh -> Exc, homeostasis)
        istdp_namespace = {
            'tau_istdp': 20*ms,  
            'eta_istdp': 0.02,   # Fast learning rate for inhibition
            'w_max_inh': 15*nS,  # Strong inhibition if necessary
            'w_min_inh': 0.1*nS,
            'tau_forget': 20*second,
            'nS': nS,
            'clip': clip
        }   

        for id_src, src_exc in enumerate(self.exc_groups):
            
            is_motor = (id_src == len(self.exc_groups) - 1)

            # --- Local inhibitory circuit (Exc <-> Inh) ---
            src_inh = self.inh_groups[id_src]
            if src_inh is not None:
                # 1a. Exc -> Inh (recruitment, static weights)

                #p_ei = 0.8 if is_motor else 0.5
                p_ei = 0.5

                S_ei = Synapses(src_exc, src_inh, on_pre='g_exc_post += 3*nS', name=f'Local_Exc_Inh_{id_src}')
                S_ei.connect(p=p_ei)
                self.synapse_groups.append(S_ei)
                

                # 1b. Inh -> Exc (iSTDP feedback / balance)
                on_pre_istdp = '''
                g_inh_post += w
                apre_inh += 1.0
                w = clip(w + eta_istdp * (apost_inh - alpha_istdp) * nS, w_min_inh, w_max_inh)
                '''
                
                on_post_istdp = '''
                apost_inh += 1.0
                w = clip(w + eta_istdp * apre_inh * nS, w_min_inh, w_max_inh)
                '''
                
                S_ie = Synapses(src_inh, src_exc, 
                                model=self.eqs_istdp,
                                on_pre=on_pre_istdp,
                                on_post=on_post_istdp,
                                namespace=istdp_namespace,
                                method='euler',
                                name=f'Local_Inh_Exc_{id_src}')
                
                S_ie.connect(p=0.8) 
                S_ie.w = '1.0*nS + rand() * 4.0*nS'
                
                S_ie.alpha_istdp = 0.05
                self.synapse_groups.append(S_ie)

            """Recurrent synapses in every layer
            # ---Excitatory FEEDFORWARD & RECURRENT(within same layer) (Exc -> Exc)---
            for id_tgt, tgt_exc in enumerate(self.exc_groups):
                distance = id_tgt - id_src
                
                is_recurrent = (distance == 0) 
                is_feedforward = (distance == 1)

                if is_recurrent or is_feedforward:
                    syn = Synapses(src_exc, tgt_exc,
                                model=self.eqs_rstdp,
                                namespace=syn_rstdp_namespace,
                                on_pre='''
                                g_exc_post += w          
                                apre += A_plus            
                                c = clip(c + apost, -c_max, c_max)
                                ''',
                                on_post='''
                                apost += A_minus
                                c = clip(c + apre, -c_max, c_max)
                                ''',
                                name=f'Syn_{id_src}_to_{id_tgt}')
                    
                    if is_recurrent:
                        # Gaussian distributian
                        syn.connect(condition='i != j', p='0.8 * exp(-((i-j)**2) / (2 * 4.0**2))')
                    else:
                        # all to all connection
                        syn.connect(p=1)
                    
                    syn.w = '4.0*nS + rand() * 4.0*nS'
                    syn.c = 0 
                    syn.tau_c = 40 * ms 
                    
                    syn.dopa_idx = 0
                    syn.d_syn = linked_var(self.dopamine_group, 'D', index='dopa_idx')
                    syn.delay = '1*ms + rand()*2*ms'
                    
                    self.synapse_groups.append(syn)
            """
            
            
            # --- Excitatory feedforward & recurrent connections (Exc -> Exc) ---
            # Active configuration: recurrence is restricted (see disabled block above for the fully-recurrent variant)
            for id_tgt, tgt_exc in enumerate(self.exc_groups):
                distance = id_tgt - id_src

                is_recurrent = (distance == 0)
                is_feedforward = (distance == 1)

                # Recurrence is disabled for the input and motor layers; hidden layers could use it if added later
                is_valid_recurrent = is_recurrent and not (id_src == 0 or is_motor)

                # if is_recurrent or is_feedforward:
                if is_valid_recurrent or is_feedforward:
                    syn = Synapses(src_exc, tgt_exc,
                                model=self.eqs_rstdp,
                                namespace=syn_rstdp_namespace,
                                on_pre='''
                                g_exc_post += w          
                                apre += A_plus            
                                c = clip(c + apost, -c_max, c_max)
                                ''',
                                on_post='''
                                apost += A_minus
                                c = clip(c + apre, -c_max, c_max)
                                ''',
                                name=f'Syn_{id_src}_to_{id_tgt}')
                    
                    if is_valid_recurrent:
                        # Gaussian falloff by distance
                        syn.connect(condition='i != j', p='0.8 * exp(-((i-j)**2) / (2 * 4.0**2))')
                    else:
                        # Fully connected (feedforward)
                        syn.connect(p=1)
                    
                    syn.w = '4.0*nS + rand() * 4.0*nS'
                    syn.c = 0 
                    syn.tau_c = self.tau_c
                    
                    syn.dopa_idx = 0
                    syn.d_syn = linked_var(self.dopamine_group, 'D', index='dopa_idx')
                    syn.delay = '1*ms + rand()*2*ms'
                    
                    self.synapse_groups.append(syn)

    def _create_monitors(self):
        self.spike_monitors = [SpikeMonitor(self.sensor_input, name='Spikes_Input')]
        for area in self.neuron_groups:
            if area is not None:
                self.spike_monitors.append(SpikeMonitor(area, name=f'Spikes_{area.name}'))

        self.monitors = self.spike_monitors

    def apply_episodic_normalization(self):
        """Pull each post-neuron's incoming weight sum softly toward w_target; called on every miss."""
        target_syns = [s for s in self.synapse_groups if 'Syn_' in s.name]
        for syn in target_syns:
            w_arr = np.array(syn.w / nS)
            j_arr = np.array(syn.j)
            
            if len(w_arr) == 0: continue
            
            w_max_val = self.w_max / nS
            w_min_val = self.w_min / nS 

            unique_posts = np.unique(j_arr)
            for p_idx in unique_posts:
                mask = (j_arr == p_idx)
                current_sum = np.sum(w_arr[mask])
                N_connections = np.sum(mask)
                
                if N_connections == 0 or current_sum == 0: continue
                
                target_sum = N_connections * (self.w_target / nS)
                scaling_factor = target_sum / current_sum
                
                soft_factor = 1.0 + 0.2 * (scaling_factor - 1.0) 
                
                new_w = w_arr[mask] * soft_factor
                
                w_arr[mask] = np.clip(new_w, w_min_val, w_max_val)
                
            syn.w = w_arr * nS

    def _update_weights(self):
        target_syns = [s for s in self.synapse_groups if 'Syn_' in s.name]

        # Periodic weight update: R-STDP learning plus passive decay
        @network_operation(dt=10*ms)
        def learn_weights():
            
            w_max_val = self.w_max / nS
            w_min_val = self.w_min / nS

            for syn in target_syns:
                w_arr = np.array(syn.w / nS)
                c_arr = np.array(syn.c)
                d_arr = np.array(syn.d_syn)
                
                if len(w_arr) == 0: continue

                w_arr = w_arr * (1.0 - self.decay_factor)

                current_gain = self.dopamine_group.gain[0]
                
                dw_raw = c_arr * d_arr * self.learning_rate * current_gain

                soft_bound = np.where(dw_raw > 0, w_max_val - w_arr, w_arr - w_min_val)
                
                new_w = np.clip(w_arr + dw_raw * soft_bound, w_min_val, w_max_val)
                
                syn.w = new_w * nS

        self.learn_op = learn_weights
    
    def create_game_loop(self, game_instance, visual_mode=True, telemetry_filename="brain_telemetry.h5"):
        self.game = game_instance
        self.visual_mode = visual_mode
        self.render_every_n = 1 if visual_mode else 50
        self.step_count = 0
        self.action_counts = {0: 0, 1: 0, 2: 0}  # 0 = hold, 1 = up, 2 = down
        self.reset_count = 0
        self.total_reward_count = 0
        self.v_current = 0.0

        try:
            self.motor_syn = [s for s in self.synapse_groups if 'Syn_' in s.name and f'to_{len(self.topology)-1}' in s.name][0]
        except IndexError:
            self.motor_syn = None
            print("WARNING: No motor synapses found")

        # Initialize telemetry configuration
        telemetry_config = {
            'dopamine': {
                'object': self.dopamine_group, 
                'vars': ['D', 'avg_hit_rate']
            },
            'motor_synapses': {
                'object': self.motor_syn, 
                'vars': ['w', 'c']
            }
        }

        # 1. Register iSTDP synapses ("the brake")
        try:
            istdp_syn = [s for s in self.synapse_groups if 'Local_Inh_Exc' in s.name][0]
            telemetry_config['istdp_synapses'] = {
                'object': istdp_syn,
                'vars': ['w']
            }
        except IndexError:
            pass

        # 2. Excitatory areas (including conductances)
        for i, exc_group in enumerate(self.exc_groups):
            area_type = self._get_area_properties(i)
            telemetry_config[f'Area_{i}_{area_type}_Exc'] = {
                'object': exc_group,
                'vars': ['v', 'v_thresh', 'g_exc', 'g_inh']
            }

        # 3. Inhibitory areas
        for i, inh_group in enumerate(self.inh_groups):
            if inh_group is not None:
                area_type = self._get_area_properties(i)
                telemetry_config[f'Area_{i}_{area_type}_Inh'] = {
                    'object': inh_group,
                    'vars': ['v']
                }
        
        self.telemetry = BrainTelemetry(telemetry_filename, telemetry_config, chunk_size=200)
        self.prev_ball_x = 0.0

        @network_operation(dt=50*ms)
        def flight_recorder(t):
            # Compute the current chunk height from game state
            chunk_height = self.game.height / self.game.num_chunks

            # Determine the discrete ball chunk
            ball_chunk = int(self.game.ball_y / chunk_height)
            ball_chunk = max(0, min(self.game.num_chunks - 1, ball_chunk))

            # Determine the discrete paddle chunk
            paddle_center = self.game.paddle_y + (self.game.paddle_h / 2.0)
            paddle_chunk = int(paddle_center / chunk_height)
            paddle_chunk = max(0, min(self.game.num_chunks - 1, paddle_chunk))

            # Tracking error = absolute chunk distance (0, 1, 2, ...)
            dist_chunk = abs(paddle_chunk - ball_chunk)

            current_dopa = self.dopamine_group.D[0]

            # Current curriculum level
            current_level = self.game.level

            # Cast to float for HDF5 compatibility
            self.telemetry.gather_data(t, current_dist=float(dist_chunk), reward=current_dopa, v_current=self.v_current, level=current_level)
            
        self.net.add(flight_recorder)

        num_inputs = self.topology[0]
        sigma = 4.0 
        max_rate = 100
        x_grid = np.arange(num_inputs)
        dt_sim = 10*ms 

        @network_operation(dt=dt_sim)
        def game_loop(t):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.net.stop()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.net.stop()
                    elif event.key == pygame.K_t:
                        self.visual_mode = not self.visual_mode
                        self.render_every_n = 1 if self.visual_mode else 50
                        self.game.set_speed_mode(not self.visual_mode)
                        print(f"Visual Mode: {self.visual_mode}")

            # 2. Process sensory input (ball position only)
            ball_pos_norm = self.game.get_state()
            center_idx_ball = ball_pos_norm * (num_inputs - 1)
            rates_ball = np.exp(-((x_grid - center_idx_ball)**2) / (2 * sigma**2)) * max_rate
            self.sensor_input.rate = rates_ball * Hz

            # 3. Decode motor output (chunked winner-take-all with hysteresis)
            motor_exc_group = self.exc_groups[-1]
            motor_mon = next(m for m in self.spike_monitors if m.source == motor_exc_group)
            
            window = 50*ms
            recent_spikes_idx = motor_mon.i[motor_mon.t > t - window]
            
            chunk_height_px = self.game.height / self.game.num_chunks
            
            # Determine the paddle's current chunk
            paddle_center = self.game.paddle_y + (self.game.paddle_h / 2.0)
            current_chunk = int(paddle_center / chunk_height_px)
            current_chunk = max(0, min(self.game.num_chunks - 1, current_chunk))  # Clamp to valid range

            if len(recent_spikes_idx) > 0:
                # Bin spikes into the current number of chunks
                counts, _ = np.histogram(recent_spikes_idx, bins=self.game.num_chunks, range=(0, 100))
                winning_chunk = np.argmax(counts)

                # Hysteresis filter: bias toward staying in the current chunk
                hysteresis_factor = 1.25  # Empirically chosen sweet spot

                if winning_chunk != current_chunk:
                    # Only switch if the new chunk has a clear spike-count majority
                    if counts[winning_chunk] > counts[current_chunk] * hysteresis_factor:
                        target_y = winning_chunk * chunk_height_px
                    else:
                        # Not enough dominance: stay in the current chunk
                        target_y = current_chunk * chunk_height_px
                else:
                    # Already the current chunk: stay
                    target_y = winning_chunk * chunk_height_px
            else:
                # No recent spikes: hold position
                target_y = current_chunk * chunk_height_px

            # 4. Advance game state
            reward, done, hit, C = self.game.step(target_y)

            # 5. Update dopamine level
            new_dopamine = self.dopamine_group.D[0] + reward
            self.dopamine_group.D[0] = np.clip(new_dopamine, -2.0, 2.0)

            # 6. Handle hit/miss outcome
            if hit:
                self.total_reward_count += 1
            if done:
                self.reset_count += 1
                self.game.reset()
                self.apply_episodic_normalization()

            # 7. Render game
            self.step_count += 1
            if self.step_count % self.render_every_n == 0:
                vis_data = {
                    'input_rates': rates_ball,
                    'output_spikes': np.unique(recent_spikes_idx).astype(int).tolist(),
                    'reward_signal': self.dopamine_group.D[0]
                }
                self.game.render(network_data=vis_data)
        
        self.net.add(game_loop)
    






