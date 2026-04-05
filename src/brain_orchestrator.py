import time
from typing import Dict, List, Any

from core.hyperbitnet import HyperBitnet
from core.sensory_encoder import SensoryEncoder
from core._native_loader import load_native_core

native = load_native_core()

class BrainOrchestrator:
    def __init__(self, n_nodes: int = 64):
        self.n_nodes = n_nodes
        self.hyper_net = HyperBitnet(n_nodes=self.n_nodes)
        self.encoder = SensoryEncoder(net=self.hyper_net)
        self.spiking_network = native.SpikingNetwork()
        self.current_time_ms = 0.0

        # Initialize LIFNeurons (one for each input node)
        for i in range(self.n_nodes):
            neuron = native.LIFNeuron(tau=20.0, v_thresh=1.0, refractory_period=5.0)
            self.spiking_network.add_neuron(i, neuron)
            
    def pulse(self, intent_value: int) -> Dict[str, Any]:
        """
        Executes a 100ms processing window.
        1. Injects intent state into HyperBitnet.
        2. Encodes states into spikes.
        3. Simulates the SpikingNetwork for 100ms.
        """
        window_duration_ms = 100.0
        
        # 1. Update Quantum and Classical states
        self.hyper_net.inject_state(intent_value)
        
        intent_level = sum(self.hyper_net.states) / self.n_nodes
        quantum_coherence = sum(self.hyper_net.quantum_states) / self.n_nodes
        
        # 2. Encode into relative spike times and translate to absolute time
        relative_events = self.encoder.encode(duration_ms=window_duration_ms)
        
        for node_id, rel_t_ms in relative_events:
            abs_t_ms = self.current_time_ms + rel_t_ms
            # Weight = 1.0 to guarantee potential threshold hit (v_thresh=1.0)
            self.spiking_network.schedule_event(abs_t_ms, node_id, weight=1.1)
            
        # 3. Track previous firing times to see who fired in this window
        prev_fire_times = {
            i: self.spiking_network.neurons[i].last_fire_t 
            for i in range(self.n_nodes)
        }
        
        # Run native simulation
        self.spiking_network.run_until_empty()
        
        # Calculate exactly who fired
        fired_neuron_ids = []
        for i in range(self.n_nodes):
            current_fire_t = self.spiking_network.neurons[i].last_fire_t
            if current_fire_t > prev_fire_times[i]:
                fired_neuron_ids.append(i)
                
        # Advance time
        self.current_time_ms += window_duration_ms

        return {
            "intent_level": intent_level,
            "quantum_coherence": quantum_coherence,
            "fired_neuron_ids": fired_neuron_ids
        }
