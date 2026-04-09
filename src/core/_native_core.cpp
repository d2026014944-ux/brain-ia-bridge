#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

static inline double quantize_ternary(double weight) {
    if (weight > 0.5) {
        return 1.0;
    }
    if (weight < -0.5) {
        return -1.0;
    }
    return 0.0;
}

class LIFNeuron {
public:
    double tau;
    double v_thresh;
    double refractory_period;

    double v_m;
    double last_update_t;
    double last_fire_t;

    LIFNeuron(double tau_ = 20.0, double v_thresh_ = 1.0, double refractory_period_ = 5.0)
        : tau(tau_),
          v_thresh(v_thresh_),
          refractory_period(refractory_period_),
          v_m(0.0),
          last_update_t(0.0),
          last_fire_t(-std::numeric_limits<double>::infinity()) {}

    bool receive_spike(double current_t, double weight) {
        if (current_t - last_fire_t < refractory_period) {
            return false;
        }

        const double dt = current_t - last_update_t;
        if (dt < 0.0) {
            throw py::value_error("current_t must be monotonically non-decreasing");
        }

        v_m = v_m * std::exp(-dt / tau);
        last_update_t = current_t;

        v_m += weight;

        if (v_m >= v_thresh) {
            v_m = 0.0;
            last_fire_t = current_t;
            return true;
        }

        return false;
    }
};

class SynapseSTDP {
public:
    static constexpr double tau = 20.0;
    static constexpr double A_plus = 0.02;
    static constexpr double A_minus = 0.025;
    static constexpr double W_MAX = 1.0;
    static constexpr double W_MIN = 0.0;

    double weight;
    std::optional<double> last_pre_t_ms;
    std::optional<double> last_post_t_ms;

    explicit SynapseSTDP(double weight_ = 0.5)
        : weight(std::max(W_MIN, std::min(W_MAX, weight_))),
          last_pre_t_ms(std::nullopt),
          last_post_t_ms(std::nullopt) {}

    void apply_delta(double delta) {
        weight = std::max(W_MIN, std::min(W_MAX, weight + delta));
    }

    double compute_stdp_delta(double dt) const {
        if (dt > 0.0) {
            return A_plus * std::exp(-dt / tau);
        }
        return -A_minus * std::exp(dt / tau);
    }

    void register_pre_spike(double time) {
        last_pre_t_ms = time;
        if (!last_post_t_ms.has_value()) {
            return;
        }
        const double dt = last_post_t_ms.value() - last_pre_t_ms.value();
        apply_delta(compute_stdp_delta(dt));
    }

    void register_post_spike(double time) {
        last_post_t_ms = time;
        if (!last_pre_t_ms.has_value()) {
            return;
        }
        const double dt = last_post_t_ms.value() - last_pre_t_ms.value();
        apply_delta(compute_stdp_delta(dt));
    }

    void update(double pre_t_ms, double post_t_ms) {
        last_pre_t_ms = pre_t_ms;
        last_post_t_ms = post_t_ms;
        const double dt = last_post_t_ms.value() - last_pre_t_ms.value();
        apply_delta(compute_stdp_delta(dt));
    }
};

class SynapseSTDPBitNet {
public:
    static constexpr double tau = 20.0;
    static constexpr double A_plus = 0.02;
    static constexpr double A_minus = 0.025;
    static constexpr double W_MAX = 1.0;
    static constexpr double W_MIN = -1.0;
    static constexpr double PHASE_UPPER = 0.25;
    static constexpr double PHASE_LOWER = -0.25;
    static constexpr double PHASE_RELEASE_POS = 0.75;
    static constexpr double PHASE_RELEASE_NEG = -0.75;

    double weight;
    double phase_accumulator;
    std::optional<double> last_pre_t_ms;
    std::optional<double> last_post_t_ms;

    explicit SynapseSTDPBitNet(double weight_ = 0.0)
        : weight(quantize_ternary(weight_)),
          phase_accumulator(std::max(W_MIN, std::min(W_MAX, weight_))),
          last_pre_t_ms(std::nullopt),
          last_post_t_ms(std::nullopt) {}

    void apply_delta(double delta) {
        phase_accumulator = std::max(W_MIN, std::min(W_MAX, phase_accumulator + delta));

        // Discrete phase transitions with mild hysteresis:
        // 0 -> +/-1 occurs near zero-cross thresholds;
        // +/-1 -> 0 requires sufficient opposite pressure.
        if (weight > 0.0) {
            if (phase_accumulator <= PHASE_LOWER) {
                weight = -1.0;
                return;
            }
            if (phase_accumulator < PHASE_RELEASE_POS) {
                weight = 0.0;
                return;
            }
            weight = 1.0;
            return;
        }

        if (weight < 0.0) {
            if (phase_accumulator >= PHASE_UPPER) {
                weight = 1.0;
                return;
            }
            if (phase_accumulator > PHASE_RELEASE_NEG) {
                weight = 0.0;
                return;
            }
            weight = -1.0;
            return;
        }

        if (phase_accumulator >= PHASE_UPPER) {
            weight = 1.0;
            return;
        }
        if (phase_accumulator <= PHASE_LOWER) {
            weight = -1.0;
            return;
        }
        weight = 0.0;
    }

    double compute_stdp_delta(double dt) const {
        if (dt > 0.0) {
            return A_plus * std::exp(-dt / tau);
        }
        return -A_minus * std::exp(dt / tau);
    }

    void register_pre_spike(double time) {
        last_pre_t_ms = time;
        if (!last_post_t_ms.has_value()) {
            return;
        }
        const double dt = last_post_t_ms.value() - last_pre_t_ms.value();
        apply_delta(compute_stdp_delta(dt));
    }

    void register_post_spike(double time) {
        last_post_t_ms = time;
        if (!last_pre_t_ms.has_value()) {
            return;
        }
        const double dt = last_post_t_ms.value() - last_pre_t_ms.value();
        apply_delta(compute_stdp_delta(dt));
    }

    void update(double pre_t_ms, double post_t_ms) {
        last_pre_t_ms = pre_t_ms;
        last_post_t_ms = post_t_ms;
        const double dt = last_post_t_ms.value() - last_pre_t_ms.value();
        apply_delta(compute_stdp_delta(dt));
    }

    void propagate_many(std::size_t n_pairs, double dt_pre_post_ms = 1.0) {
        const double delta = compute_stdp_delta(dt_pre_post_ms);
        for (std::size_t i = 0; i < n_pairs; ++i) {
            apply_delta(delta);
        }
    }
};

struct EventRecord {
    double time_ms;
    std::int64_t event_id;
    py::object target_id;
    double weight;
};

struct EdgeBinaryRecord {
    int pre;
    int post;
    float weight;
    float delay;
};

class SpikingNetwork {
public:
    py::dict neurons;
    py::dict synapses;
    bool learning_enabled;

    explicit SpikingNetwork(bool learning_enabled_ = true)
        : learning_enabled(learning_enabled_),
          _event_counter(0) {}

    void add_neuron(py::object node_id, py::object neuron_instance) {
        neurons[node_id] = neuron_instance;
    }

    void add_connection(py::object pre_id, py::object post_id, double weight, double delay_ms) {
        if (!synapses.contains(pre_id)) {
            synapses[pre_id] = py::list();
        }
        py::list edges = synapses[pre_id].cast<py::list>();
        const std::size_t edge_index = static_cast<std::size_t>(py::len(edges));
        const double quantized_weight = quantize_ternary(weight);
        edges.append(py::make_tuple(post_id, quantized_weight, delay_ms));

        const std::string edge_key = make_edge_key(pre_id, post_id, edge_index);
        stdp_synapses.insert_or_assign(edge_key, SynapseSTDPBitNet(quantized_weight));
    }

    void connect(py::object pre, py::object post, double weight, double delay) {
        add_connection(pre, post, weight, delay);
    }

    void schedule_event(double time_ms, py::object target_id, double weight = 0.0) {
        EventRecord ev{time_ms, _event_counter, target_id, weight};
        _event_counter += 1;
        event_queue.push_back(ev);
        std::push_heap(event_queue.begin(), event_queue.end(), event_cmp);
    }

    py::tuple pop_next_event() {
        if (event_queue.empty()) {
            throw py::index_error("pop from empty event_queue");
        }

        std::pop_heap(event_queue.begin(), event_queue.end(), event_cmp);
        EventRecord ev = event_queue.back();
        event_queue.pop_back();
        return py::make_tuple(ev.time_ms, ev.event_id, ev.target_id, ev.weight);
    }

    void run_until_empty() {
        while (!event_queue.empty()) {
            py::tuple event = pop_next_event();
            const double time_ms = event[0].cast<double>();
            py::object target_id = event[2];
            const double weight = event[3].cast<double>();

            if (weight == 0.0) {
                continue;
            }

            py::object neuron = neurons[target_id];
            bool fired = neuron.attr("receive_spike")(time_ms, weight).cast<bool>();

            if (fired && synapses.contains(target_id)) {
                py::list outgoing = synapses[target_id].cast<py::list>();
                const std::size_t edge_count = static_cast<std::size_t>(py::len(outgoing));

                for (std::size_t edge_index = 0; edge_index < edge_count; ++edge_index) {
                    py::handle edge_h = outgoing[edge_index];
                    py::tuple edge = edge_h.cast<py::tuple>();
                    py::object post_id = edge[0];
                    double conn_weight = quantize_ternary(edge[1].cast<double>());
                    const double delay_ms = edge[2].cast<double>();

                    if (learning_enabled) {
                        const std::string edge_key = make_edge_key(target_id, post_id, edge_index);
                        auto it = stdp_synapses.find(edge_key);
                        if (it == stdp_synapses.end()) {
                            it = stdp_synapses.insert({edge_key, SynapseSTDPBitNet(conn_weight)}).first;
                        }

                        // Discrete STDP phase transition with ternary quantization.
                        it->second.update(time_ms, time_ms + delay_ms);
                        conn_weight = quantize_ternary(it->second.weight);
                        outgoing[edge_index] = py::make_tuple(post_id, conn_weight, delay_ms);
                    }

                    if (conn_weight > 0.0) {
                        schedule_event(time_ms + delay_ms, post_id, 1.0);
                    } else if (conn_weight < 0.0) {
                        schedule_event(time_ms + delay_ms, post_id, -1.0);
                    }
                }
            }
        }
    }

    py::list run_and_trace() {
        py::list trace;
        while (!event_queue.empty()) {
            py::tuple event = pop_next_event();
            const double time_ms = event[0].cast<double>();
            const std::int64_t event_id = event[1].cast<std::int64_t>();
            py::object target_id = event[2];
            const double weight = event[3].cast<double>();

            if (weight == 0.0) {
                continue;
            }

            py::object neuron = neurons[target_id];
            bool fired = neuron.attr("receive_spike")(time_ms, weight).cast<bool>();

            if (fired) {
                py::dict row;
                row["time_ms"] = time_ms;
                row["event_id"] = event_id;
                row["node_id"] = target_id;
                row["input_weight"] = weight;
                trace.append(row);

                if (synapses.contains(target_id)) {
                    py::list outgoing = synapses[target_id].cast<py::list>();
                    const std::size_t edge_count = static_cast<std::size_t>(py::len(outgoing));

                    for (std::size_t edge_index = 0; edge_index < edge_count; ++edge_index) {
                        py::handle edge_h = outgoing[edge_index];
                        py::tuple edge = edge_h.cast<py::tuple>();
                        py::object post_id = edge[0];
                        double conn_weight = quantize_ternary(edge[1].cast<double>());
                        const double delay_ms = edge[2].cast<double>();

                        if (learning_enabled) {
                            const std::string edge_key = make_edge_key(target_id, post_id, edge_index);
                            auto it = stdp_synapses.find(edge_key);
                            if (it == stdp_synapses.end()) {
                                it = stdp_synapses.insert({edge_key, SynapseSTDPBitNet(conn_weight)}).first;
                            }

                            it->second.update(time_ms, time_ms + delay_ms);
                            conn_weight = quantize_ternary(it->second.weight);
                            outgoing[edge_index] = py::make_tuple(post_id, conn_weight, delay_ms);
                        }

                        if (conn_weight > 0.0) {
                            schedule_event(time_ms + delay_ms, post_id, 1.0);
                        } else if (conn_weight < 0.0) {
                            schedule_event(time_ms + delay_ms, post_id, -1.0);
                        }
                    }
                }
            }
        }

        return trace;
    }

    py::list get_synaptic_strengths() const {
        py::list out;
        for (auto item : synapses) {
            py::object pre_id = py::reinterpret_borrow<py::object>(item.first);
            py::list outgoing = item.second.cast<py::list>();
            for (py::handle edge_h : outgoing) {
                py::tuple edge = edge_h.cast<py::tuple>();
                py::dict row;
                row["pre_id"] = pre_id;
                row["post_id"] = edge[0];
                row["weight"] = edge[1];
                row["delay_ms"] = edge[2];
                out.append(row);
            }
        }
        return out;
    }

    py::list get_event_queue() const {
        py::list out;
        for (const auto& ev : event_queue) {
            out.append(py::make_tuple(ev.time_ms, ev.event_id, ev.target_id, ev.weight));
        }
        return out;
    }

    void save_weights(const std::string& path) const {
        std::size_t connection_count = 0;
        for (auto item : synapses) {
            py::list outgoing = item.second.cast<py::list>();
            connection_count += static_cast<std::size_t>(py::len(outgoing));
        }

        std::vector<EdgeBinaryRecord> records;
        records.reserve(connection_count);

        for (auto item : synapses) {
            py::object pre_id_obj = py::reinterpret_borrow<py::object>(item.first);
            int pre_id;
            try {
                pre_id = pre_id_obj.cast<int>();
            } catch (const py::cast_error&) {
                throw py::type_error("save_weights requires integer neuron IDs");
            }

            py::list outgoing = item.second.cast<py::list>();
            for (py::handle edge_h : outgoing) {
                py::tuple edge = edge_h.cast<py::tuple>();
                int post_id;
                try {
                    post_id = edge[0].cast<int>();
                } catch (const py::cast_error&) {
                    throw py::type_error("save_weights requires integer neuron IDs");
                }

                const auto weight = static_cast<float>(edge[1].cast<double>());
                const auto delay = static_cast<float>(edge[2].cast<double>());
                records.push_back(EdgeBinaryRecord{pre_id, post_id, weight, delay});
            }
        }

        std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            throw py::value_error("could not open file for binary save: " + path);
        }

        std::vector<char> io_buffer(1 << 20);
        out.rdbuf()->pubsetbuf(io_buffer.data(), static_cast<std::streamsize>(io_buffer.size()));

        const std::uint64_t count = static_cast<std::uint64_t>(records.size());
        out.write(reinterpret_cast<const char*>(&count), static_cast<std::streamsize>(sizeof(count)));
        if (!records.empty()) {
            const std::size_t bytes_to_write = sizeof(EdgeBinaryRecord) * records.size();
            out.write(
                reinterpret_cast<const char*>(records.data()),
                static_cast<std::streamsize>(bytes_to_write));
        }

        if (!out.good()) {
            throw py::value_error("binary save failed while writing: " + path);
        }
    }

    void load_weights(const std::string& path) {
        std::ifstream in(path, std::ios::binary | std::ios::in);
        if (!in.is_open()) {
            throw py::value_error("could not open file for binary load: " + path);
        }

        std::vector<char> io_buffer(1 << 20);
        in.rdbuf()->pubsetbuf(io_buffer.data(), static_cast<std::streamsize>(io_buffer.size()));

        std::uint64_t count = 0;
        in.read(reinterpret_cast<char*>(&count), static_cast<std::streamsize>(sizeof(count)));
        if (!in.good() && !in.eof()) {
            throw py::value_error("binary load failed while reading header: " + path);
        }
        if (in.gcount() != static_cast<std::streamsize>(sizeof(count))) {
            throw py::value_error("invalid binary file header: " + path);
        }

        const std::uint64_t max_count =
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(EdgeBinaryRecord));
        if (count > max_count) {
            throw py::value_error("binary file too large to load safely: " + path);
        }

        std::vector<EdgeBinaryRecord> records(static_cast<std::size_t>(count));
        if (!records.empty()) {
            const std::size_t bytes_to_read = sizeof(EdgeBinaryRecord) * records.size();
            in.read(
                reinterpret_cast<char*>(records.data()),
                static_cast<std::streamsize>(bytes_to_read));

            if (!in.good() && !in.eof()) {
                throw py::value_error("binary load failed while reading payload: " + path);
            }
            if (in.gcount() != static_cast<std::streamsize>(bytes_to_read)) {
                throw py::value_error("binary payload is truncated or corrupted: " + path);
            }
        }

        neurons = py::dict();
        synapses = py::dict();
        stdp_synapses.clear();
        event_queue.clear();
        _event_counter = 0;

        auto ensure_neuron = [this](int node_id) {
            py::int_ py_node_id(node_id);
            if (!neurons.contains(py_node_id)) {
                neurons[py_node_id] = py::cast(LIFNeuron());
            }
        };

        for (const auto& rec : records) {
            ensure_neuron(rec.pre);
            ensure_neuron(rec.post);

            py::int_ pre_id_obj(rec.pre);
            py::int_ post_id_obj(rec.post);
            if (!synapses.contains(pre_id_obj)) {
                synapses[pre_id_obj] = py::list();
            }

            py::list outgoing = synapses[pre_id_obj].cast<py::list>();
            const std::size_t edge_index = static_cast<std::size_t>(py::len(outgoing));
            const double persisted_weight = static_cast<double>(rec.weight);
            const double persisted_delay = static_cast<double>(rec.delay);
            outgoing.append(py::make_tuple(post_id_obj, persisted_weight, persisted_delay));

            const std::string edge_key = make_edge_key(pre_id_obj, post_id_obj, edge_index);
            stdp_synapses.insert_or_assign(edge_key, SynapseSTDPBitNet(persisted_weight));
        }
    }

private:
    std::int64_t _event_counter;
    std::vector<EventRecord> event_queue;
    std::unordered_map<std::string, SynapseSTDPBitNet> stdp_synapses;

    std::string make_edge_key(const py::object& pre_id, const py::object& post_id, std::size_t edge_index) const {
        return py::str(pre_id).cast<std::string>() + "->" + py::str(post_id).cast<std::string>() + "#" + std::to_string(edge_index);
    }

    static bool event_cmp(const EventRecord& a, const EventRecord& b) {
        if (a.time_ms != b.time_ms) {
            return a.time_ms > b.time_ms;
        }
        return a.event_id > b.event_id;
    }
};

PYBIND11_MODULE(_native_core, m) {
    py::class_<LIFNeuron> lif_cls(m, "LIFNeuron");
    lif_cls
        .def(py::init<double, double, double>(), py::arg("tau") = 20.0, py::arg("v_thresh") = 1.0, py::arg("refractory_period") = 5.0)
        .def("receive_spike", &LIFNeuron::receive_spike, py::arg("current_t"), py::arg("weight"))
        .def_readwrite("tau", &LIFNeuron::tau)
        .def_readwrite("v_thresh", &LIFNeuron::v_thresh)
        .def_readwrite("refractory_period", &LIFNeuron::refractory_period)
        .def_readwrite("v_m", &LIFNeuron::v_m)
        .def_readwrite("last_update_t", &LIFNeuron::last_update_t)
        .def_readwrite("last_fire_t", &LIFNeuron::last_fire_t);

    py::class_<SynapseSTDP> syn_cls(m, "SynapseSTDP");
    syn_cls
        .def(py::init<double>(), py::arg("weight") = 0.5)
        .def("_apply_delta", &SynapseSTDP::apply_delta, py::arg("delta"))
        .def("_compute_stdp_delta", &SynapseSTDP::compute_stdp_delta, py::arg("dt"))
        .def("register_pre_spike", &SynapseSTDP::register_pre_spike, py::arg("time"))
        .def("register_post_spike", &SynapseSTDP::register_post_spike, py::arg("time"))
        .def("update", &SynapseSTDP::update, py::arg("pre_t_ms"), py::arg("post_t_ms"))
        .def_readwrite("weight", &SynapseSTDP::weight)
        .def_property(
            "last_pre_t_ms",
            [](const SynapseSTDP& s) -> py::object {
                if (!s.last_pre_t_ms.has_value()) {
                    return py::none();
                }
                return py::float_(s.last_pre_t_ms.value());
            },
            [](SynapseSTDP& s, py::object value) {
                if (value.is_none()) {
                    s.last_pre_t_ms = std::nullopt;
                } else {
                    s.last_pre_t_ms = value.cast<double>();
                }
            })
        .def_property(
            "last_post_t_ms",
            [](const SynapseSTDP& s) -> py::object {
                if (!s.last_post_t_ms.has_value()) {
                    return py::none();
                }
                return py::float_(s.last_post_t_ms.value());
            },
            [](SynapseSTDP& s, py::object value) {
                if (value.is_none()) {
                    s.last_post_t_ms = std::nullopt;
                } else {
                    s.last_post_t_ms = value.cast<double>();
                }
            });
    syn_cls.attr("tau") = SynapseSTDP::tau;
    syn_cls.attr("A_plus") = SynapseSTDP::A_plus;
    syn_cls.attr("A_minus") = SynapseSTDP::A_minus;
    syn_cls.attr("W_MAX") = SynapseSTDP::W_MAX;
    syn_cls.attr("W_MIN") = SynapseSTDP::W_MIN;

    py::class_<SynapseSTDPBitNet> bitnet_syn_cls(m, "SynapseSTDPBitNet");
    bitnet_syn_cls
        .def(
            py::init([](double weight, py::object, py::object, py::object, py::object) {
                return SynapseSTDPBitNet(weight);
            }),
            py::arg("weight") = 0.0,
            py::arg("quantization") = py::none(),
            py::arg("weight_mode") = py::none(),
            py::arg("bitnet") = py::none(),
            py::arg("ternary") = py::none())
        .def(
            py::init([](double w, py::object, py::object, py::object, py::object) {
                return SynapseSTDPBitNet(w);
            }),
            py::arg("w"),
            py::arg("quantization") = py::none(),
            py::arg("weight_mode") = py::none(),
            py::arg("bitnet") = py::none(),
            py::arg("ternary") = py::none())
        .def("_apply_delta", &SynapseSTDPBitNet::apply_delta, py::arg("delta"))
        .def("_compute_stdp_delta", &SynapseSTDPBitNet::compute_stdp_delta, py::arg("dt"))
        .def("register_pre_spike", &SynapseSTDPBitNet::register_pre_spike, py::arg("time"))
        .def("register_post_spike", &SynapseSTDPBitNet::register_post_spike, py::arg("time"))
        .def("update", &SynapseSTDPBitNet::update, py::arg("pre_t_ms"), py::arg("post_t_ms"))
        .def("propagate_many", &SynapseSTDPBitNet::propagate_many, py::arg("n_pairs"), py::arg("dt_pre_post_ms") = 1.0)
        .def_readwrite("weight", &SynapseSTDPBitNet::weight)
        .def_readwrite("phase_accumulator", &SynapseSTDPBitNet::phase_accumulator)
        .def_property(
            "last_pre_t_ms",
            [](const SynapseSTDPBitNet& s) -> py::object {
                if (!s.last_pre_t_ms.has_value()) {
                    return py::none();
                }
                return py::float_(s.last_pre_t_ms.value());
            },
            [](SynapseSTDPBitNet& s, py::object value) {
                if (value.is_none()) {
                    s.last_pre_t_ms = std::nullopt;
                } else {
                    s.last_pre_t_ms = value.cast<double>();
                }
            })
        .def_property(
            "last_post_t_ms",
            [](const SynapseSTDPBitNet& s) -> py::object {
                if (!s.last_post_t_ms.has_value()) {
                    return py::none();
                }
                return py::float_(s.last_post_t_ms.value());
            },
            [](SynapseSTDPBitNet& s, py::object value) {
                if (value.is_none()) {
                    s.last_post_t_ms = std::nullopt;
                } else {
                    s.last_post_t_ms = value.cast<double>();
                }
            })
        .def_property_readonly("is_bitnet", [](const SynapseSTDPBitNet&) { return true; })
        .def_property_readonly("quantization", [](const SynapseSTDPBitNet&) { return std::string("bitnet_1_58"); });
    bitnet_syn_cls.attr("tau") = SynapseSTDPBitNet::tau;
    bitnet_syn_cls.attr("A_plus") = SynapseSTDPBitNet::A_plus;
    bitnet_syn_cls.attr("A_minus") = SynapseSTDPBitNet::A_minus;
    bitnet_syn_cls.attr("W_MAX") = SynapseSTDPBitNet::W_MAX;
    bitnet_syn_cls.attr("W_MIN") = SynapseSTDPBitNet::W_MIN;
    bitnet_syn_cls.attr("PHASE_UPPER") = SynapseSTDPBitNet::PHASE_UPPER;
    bitnet_syn_cls.attr("PHASE_LOWER") = SynapseSTDPBitNet::PHASE_LOWER;
    bitnet_syn_cls.attr("PHASE_RELEASE_POS") = SynapseSTDPBitNet::PHASE_RELEASE_POS;
    bitnet_syn_cls.attr("PHASE_RELEASE_NEG") = SynapseSTDPBitNet::PHASE_RELEASE_NEG;

    py::class_<SpikingNetwork>(m, "SpikingNetwork")
        .def(py::init<bool>(), py::arg("learning_enabled") = true)
        .def("add_neuron", &SpikingNetwork::add_neuron, py::arg("node_id"), py::arg("neuron_instance"))
        .def("add_connection", &SpikingNetwork::add_connection, py::arg("pre_id"), py::arg("post_id"), py::arg("weight"), py::arg("delay_ms"))
        .def("connect", &SpikingNetwork::connect, py::arg("pre"), py::arg("post"), py::arg("weight"), py::arg("delay"))
        .def(
            "save_weights",
            [](const SpikingNetwork& net, py::object path_like) {
                net.save_weights(py::str(path_like));
            },
            py::arg("path"))
        .def(
            "load_weights",
            [](SpikingNetwork& net, py::object path_like) {
                net.load_weights(py::str(path_like));
            },
            py::arg("path"))
        .def(
            "save_snapshot",
            [](const SpikingNetwork& net, py::object path_like) {
                net.save_weights(py::str(path_like));
            },
            py::arg("path"))
        .def(
            "load_snapshot",
            [](SpikingNetwork& net, py::object path_like) {
                net.load_weights(py::str(path_like));
            },
            py::arg("path"))
        .def("schedule_event", &SpikingNetwork::schedule_event, py::arg("time_ms"), py::arg("target_id"), py::arg("weight") = 0.0)
        .def("pop_next_event", &SpikingNetwork::pop_next_event)
        .def("run_until_empty", &SpikingNetwork::run_until_empty)
        .def("run_and_trace", &SpikingNetwork::run_and_trace)
        .def("get_synaptic_strengths", &SpikingNetwork::get_synaptic_strengths)
        .def_readwrite("neurons", &SpikingNetwork::neurons)
        .def_readwrite("synapses", &SpikingNetwork::synapses)
        .def_readwrite("learning_enabled", &SpikingNetwork::learning_enabled)
        .def_property_readonly("event_queue", &SpikingNetwork::get_event_queue);
}
