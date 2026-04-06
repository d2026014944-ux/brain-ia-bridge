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
        edges.append(py::make_tuple(post_id, weight, delay_ms));

        const std::string edge_key = make_edge_key(pre_id, post_id, edge_index);
        stdp_synapses.insert_or_assign(edge_key, SynapseSTDP(weight));
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

            py::object neuron = neurons[target_id];
            bool fired = neuron.attr("receive_spike")(time_ms, weight).cast<bool>();

            if (fired && synapses.contains(target_id)) {
                py::list outgoing = synapses[target_id].cast<py::list>();
                const std::size_t edge_count = static_cast<std::size_t>(py::len(outgoing));

                for (std::size_t edge_index = 0; edge_index < edge_count; ++edge_index) {
                    py::handle edge_h = outgoing[edge_index];
                    py::tuple edge = edge_h.cast<py::tuple>();
                    py::object post_id = edge[0];
                    double conn_weight = edge[1].cast<double>();
                    const double delay_ms = edge[2].cast<double>();

                    if (learning_enabled) {
                        const std::string edge_key = make_edge_key(target_id, post_id, edge_index);
                        auto it = stdp_synapses.find(edge_key);
                        if (it == stdp_synapses.end()) {
                            it = stdp_synapses.insert({edge_key, SynapseSTDP(conn_weight)}).first;
                        }

                        // Hebbian update on every propagation through this connection.
                        it->second.update(time_ms, time_ms + delay_ms);
                        conn_weight = it->second.weight;
                        outgoing[edge_index] = py::make_tuple(post_id, conn_weight, delay_ms);
                    }

                    schedule_event(time_ms + delay_ms, post_id, conn_weight);
                }
            }
        }
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
            add_connection(
                py::int_(rec.pre),
                py::int_(rec.post),
                static_cast<double>(rec.weight),
                static_cast<double>(rec.delay));
        }
    }

private:
    std::int64_t _event_counter;
    std::vector<EventRecord> event_queue;
    std::unordered_map<std::string, SynapseSTDP> stdp_synapses;

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
        .def("get_synaptic_strengths", &SpikingNetwork::get_synaptic_strengths)
        .def_readwrite("neurons", &SpikingNetwork::neurons)
        .def_readwrite("synapses", &SpikingNetwork::synapses)
        .def_readwrite("learning_enabled", &SpikingNetwork::learning_enabled)
        .def_property_readonly("event_queue", &SpikingNetwork::get_event_queue);
}
