/* FP8 process state owner, checkpoint interface, and Python bindings. */
#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "linear.h"
#include "state.h"

namespace astrai {
namespace fp8 {

State& state() {
    static State s;
    return s;
}

/*
 * Checkpoint snapshot (A1) and test observability
 */

/*
 * One ring's checkpoint slice: the buffer plus which scale pair is current (a
 * restore that assumed pair 0 would hand the next forward the wrong multiplier).
 * ``restore_ring`` recomputes the reciprocal rather than trusting the slot —
 * snapshots predating it carry 0 there.
 */
py::dict ring_state_dict(const ScaleRing& r) {
    py::dict d;
    d["state"] = r.state.detach().clone();
    d["idx"] = r.idx;
    d["cur"] = r.cur; // which scale pair is current (double-buffered)
    d["initialized"] = r.initialized;
    return d;
}

/*
 * Shape matches the Python snapshot (version / entries with shape+dtype plus the
 * three rings), so the trainer's checkpoint-extras bridge is unchanged. v2 adds
 * the slot name/role; a loader ignoring them still binds by (shape, dtype).
 */
py::dict fp8_state_dict() {
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    /*
     * Save only what a resume can bind: an orphan (dead weight) is unreachable,
     * and in ``order`` it would shift later bindings. Collect first — drop_meta
     * erases from ``order``, so pruning while walking it invalidates the iterator.
     */
    std::vector<std::shared_ptr<Fp8Meta>> dead;
    for (const auto& meta : st.order)
        if (!meta->alive())
            dead.push_back(meta);
    for (const auto& meta : dead)
        drop_meta(st, meta);
    /*
     * Slotted entries first, in slot order (they bind by name, so the file reads
     * as the module list); unslotted ones trail in registration order.
     */
    std::vector<std::shared_ptr<Fp8Meta>> slotted, rest;
    for (const auto& meta : st.order)
        (meta->slot >= 0 ? slotted : rest).push_back(meta);
    std::sort(slotted.begin(), slotted.end(),
              [](const std::shared_ptr<Fp8Meta>& a, const std::shared_ptr<Fp8Meta>& b) {
                  return a->slot < b->slot;
              });
    const auto entry_of = [](const std::shared_ptr<Fp8Meta>& meta) {
        py::dict e;
        e["shape"] = py::cast(meta->shape);
        e["dtype"] = torch_dtype_str(meta->dtype);
        e["slot_name"] = meta->slot_name;
        e["role"] = meta->role;
        e["w"] = ring_state_dict(meta->w);
        e["x"] = ring_state_dict(*meta->x);
        e["g"] = ring_state_dict(meta->g);
        return e;
    };
    py::list entries;
    for (const auto& meta : slotted)
        entries.append(entry_of(meta));
    for (const auto& meta : rest)
        entries.append(entry_of(meta));
    py::dict out;
    out["version"] = 2;
    out["entries"] = entries;
    return out;
}

void fp8_load_state_dict(py::dict sd) {
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    st.pending.clear();
    for (auto entry : sd["entries"].cast<py::list>())
        st.pending.push_back(entry.cast<py::dict>());
    st.generation += 1; // the cast cache keys on it
    for (auto& meta : st.order)
        restore_pending_locked(meta);
}

/*
 * Publish the Python slot table (id -> module path, role glob); replaces the
 * whole map. Names are metadata: fp8_reset drops training state, not the model's
 * shape, so it leaves them alone.
 */
void fp8_set_slots(py::list entries) {
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    std::unordered_map<int64_t, SlotInfo> slots;
    for (auto item : entries) {
        auto tuple = item.cast<py::tuple>();
        if (tuple.size() != 3)
            continue;
        SlotInfo info;
        info.name = tuple[1].cast<std::string>();
        info.role = tuple[2].cast<std::string>();
        slots.emplace(tuple[0].cast<int64_t>(), std::move(info));
    }
    st.slots = std::move(slots);
}

void fp8_reset() {
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    st.by_key.clear();
    st.by_slot.clear();
    st.order.clear();
    st.pending.clear();
    st.generation += 1;
    st.act_cache.clear();
}

void fp8_set_act_cache(bool enabled) {
    State& st = state();
    st.act_cache_enabled.store(enabled, std::memory_order_relaxed);
    if (!enabled)
        st.act_cache.clear();
}

void fp8_clear_act_cache() { state().act_cache.clear(); }

py::dict fp8_debug_meta(const Tensor& w, int64_t history_len, int64_t margin, int64_t slot) {
    auto meta = get_meta(w, history_len, margin, false, slot);
    auto ring = [](const ScaleRing& r) {
        py::dict d;
        d["state"] = r.state;
        d["hist"] = r.hist;
        d["scale"] = r.scale();
        d["scale_recip"] = r.scale_recip();
        d["idx"] = r.idx;
        d["cur"] = r.cur;
        d["initialized"] = r.initialized;
        return d;
    };
    py::dict out;
    out["w"] = ring(meta->w);
    out["x"] = ring(*meta->x);
    out["g"] = ring(meta->g);
    out["cast_version"] = meta->cast.version;
    out["has_cast"] = meta->cast.w8.defined();
    out["slot"] = meta->slot;
    out["slot_name"] = meta->slot_name;
    out["role"] = meta->role;
    return out;
}

py::dict fp8_debug_stats() {
    State& st = state();
    py::dict d;
    d["quantize"] = st.n_quantize.load(std::memory_order_relaxed);
    d["gemm"] = st.n_gemm.load(std::memory_order_relaxed);
    d["cast_hit"] = st.n_cast_hit.load(std::memory_order_relaxed);
    d["cast_miss"] = st.n_cast_miss.load(std::memory_order_relaxed);
    d["act_hit"] = st.n_act_hit.load(std::memory_order_relaxed);
    d["act_miss"] = st.n_act_miss.load(std::memory_order_relaxed);
    d["act_entries"] = static_cast<int64_t>(st.act_cache.size());
    d["act_cache_bytes"] = st.act_cache.byte_size();
    d["metas"] = static_cast<int64_t>(st.order.size());
    return d;
}

void fp8_debug_reset_stats() {
    State& st = state();
    st.n_quantize = 0;
    st.n_gemm = 0;
    st.n_cast_hit = 0;
    st.n_cast_miss = 0;
    st.n_act_hit = 0;
    st.n_act_miss = 0;
}

void bind_fp8(py::module& m) {
    m.def("fp8_linear", &fp8_linear, py::arg("x"), py::arg("w"), py::arg("bias") = py::none(),
          py::arg("update_rings") = true, py::arg("need_bias_grad") = false,
          py::arg("dynamic") = false, py::arg("history_len") = 16, py::arg("margin") = 0,
          py::arg("fmt_a") = at::kFloat8_e4m3fn, py::arg("fmt_b") = at::kFloat8_e5m2,
          py::arg("slot") = -1);
    m.def("fp8_state_dict", &fp8_state_dict);
    m.def("fp8_load_state_dict", &fp8_load_state_dict);
    m.def("fp8_set_slots", &fp8_set_slots, py::arg("entries"));
    m.def("fp8_reset", &fp8_reset);
    m.def("fp8_set_act_cache", &fp8_set_act_cache, py::arg("enabled"));
    m.def("fp8_clear_act_cache", &fp8_clear_act_cache);
    m.def("fp8_debug_meta", &fp8_debug_meta, py::arg("w"), py::arg("history_len") = 16,
          py::arg("margin") = 0, py::arg("slot") = -1);
    m.def("fp8_debug_stats", &fp8_debug_stats);
    m.def("fp8_debug_reset_stats", &fp8_debug_reset_stats);
}

} // namespace fp8
} // namespace astrai
