import math
import heapq

class LIFNeuron:
    def __init__(self, tau=20.0, v_thresh=1.0, refractory_period=5.0):
        self.tau = tau
        self.v_thresh = v_thresh
        self.refractory_period = refractory_period
        
        self.v_m = 0.0
        self.last_update_t = 0.0
        self.last_fire_t = float('-inf')

    def receive_spike(self, current_t, weight):
        if current_t - self.last_fire_t < self.refractory_period:
            return False
            
        dt = current_t - self.last_update_t
        if dt < 0.0:
            raise ValueError("current_t must be monotonically non-decreasing")
            
        # Leaky decay
        self.v_m = self.v_m * math.exp(-dt / self.tau)
        self.last_update_t = current_t
        
        # Integrate
        self.v_m += weight
        
        # Fire
        if self.v_m >= self.v_thresh:
            self.v_m = 0.0
            self.last_fire_t = current_t
            return True
            
        return False

class SpikingNetwork:
    def __init__(self):
        self._event_counter = 0
        self.event_queue = []
        self.neurons = {}
        self.synapses = {}

    def add_neuron(self, node_id, neuron_instance):
        self.neurons[node_id] = neuron_instance

    def add_connection(self, pre_id, post_id, weight, delay_ms):
        if pre_id not in self.synapses:
            self.synapses[pre_id] = []
        self.synapses[pre_id].append((post_id, weight, delay_ms))

    def schedule_event(self, time_ms, target_id, weight=0.0):
        heapq.heappush(self.event_queue, (time_ms, self._event_counter, target_id, weight))
        self._event_counter += 1

    def pop_next_event(self):
        if not self.event_queue:
            raise IndexError("pop from empty event_queue")
        return heapq.heappop(self.event_queue)

    def run_until_empty(self):
        while self.event_queue:
            time_ms, event_id, target_id, weight = self.pop_next_event()

            neuron = self.neurons[target_id]
            fired = neuron.receive_spike(time_ms, weight)

            if fired and target_id in self.synapses:
                for post_id, conn_weight, delay_ms in self.synapses[target_id]:
                    self.schedule_event(time_ms + delay_ms, post_id, conn_weight)

    def run_and_trace(self):
        """
        Execute all pending events and return a deterministic firing trace.

        Each trace item is a dict with:
        - time_ms: firing timestamp
        - event_id: tie-breaker sequence id from the event queue
        - node_id: neuron identifier that fired
        - input_weight: input pulse that triggered processing
        """
        trace = []
        while self.event_queue:
            time_ms, event_id, target_id, weight = self.pop_next_event()

            neuron = self.neurons[target_id]
            fired = neuron.receive_spike(time_ms, weight)

            if fired:
                trace.append(
                    {
                        "time_ms": float(time_ms),
                        "event_id": int(event_id),
                        "node_id": target_id,
                        "input_weight": float(weight),
                    }
                )

                if target_id in self.synapses:
                    for post_id, conn_weight, delay_ms in self.synapses[target_id]:
                        self.schedule_event(time_ms + delay_ms, post_id, conn_weight)

        return trace
