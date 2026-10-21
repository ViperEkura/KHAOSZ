#pragma once
/* Native GEMM planning, configuration and introspection contracts. */
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <api/native/device.h>

namespace astrai {
namespace gemm {

// Dtype-class ids the plan-table rows key on.
enum class GemmPerfClass : int { kW16A16 = 0, kW8A16, kW8A8, kF8A8 };

struct GemmRecipe {
    int cta;      // TileClass ordinal — the row-file serialization key
    int k_stages; // prefetched K tiles; ring has one extra slot
    int k_tile;   // K elements held in one pipeline stage
    int bm, bn;   // CTA geometry
    int wm, wn;   // warp tiling (the recipe name's W<x>x<y>)
    int threads;  // the manifest entry's warp tiling (first match wins)
    int smem;     // ring bytes at this staging pair's operand widths
};

struct KernelResources {
    GemmRecipe effective{}; // after warp widening and output-reclaim substitution
    int resident = 0;       // CUDA occupancy limit, not observed execution occupancy
    int registers = 0;
    int local_bytes = 0;
};

struct PlanQuery {
    int64_t m = 0;
    int64_t n = 0;
    int64_t k = 0;
    int64_t batch = 1;
    int perf_class = -1; // GemmPerfClass id; -1 matches any
    int crosswise = 0;   // direct-load operand count, see gemm_dispatch
    int ba = 2;          // operand element bytes
    int bb = 2;
    int out_elem_bytes = 2; // bf16 output by default
    int mma_k = 16;         // promoted MMA instruction K extent
    bool tma = true;        // effective staging selected for this launch
    DeviceFacts dev{};
    bool rank3a = false, rank3b = false;
    bool contiguous = false; // congruous rows and sector-aligned base pointers
    KernelResources (*resources)(const GemmRecipe&, const PlanQuery&) = nullptr;
};

struct PlanDecision {
    GemmRecipe recipe;
    int raster;
    const char* source;
};

struct LaunchPlan {
    PlanDecision decision;
    bool tma;
};

int plan_raster(const PlanQuery& q, int bm, int bn);
PlanDecision plan_dispatch(const PlanQuery& q);

struct GemmCapabilities {
    int cc;
    bool mma, fp8, tma, mx;
    const char* targets;
};
GemmCapabilities capabilities();

/*
 * Planner introspection without launching or timing kernels. `source` names the deciding row tier, the
 * rest is the recipe in dispatch-key form; `crosswise` stays an int (0/1 for
 * the Python tooling).
 */
struct PlanProbe {
    std::string source;
    int cta = 0;
    int k_stages = 0;
    int raster = 0;
    int k_tile = 0;
    int perf_class = -1;
    int crosswise = 0;
    bool tma = false;
    // Heuristic metadata: key (cta, k_stages, k_tile), effective (bm, bn, k_tile, wm, wn,
    // threads), resident CTAs, registers/thread, local bytes/thread.
    std::vector<std::vector<int>> resources;
};

/*
 * Runtime configuration (the backing of astrai.extension.policy.gemm.plan). Every field is
 * tri-state: absent leaves the knob unchanged, a value wins over the env seed;
 * pybind maps an absent dict key to "absent".
 */

/*
 * Which row tier a `rows` patch addresses; lookup ranks override > injected >
 * the compiled-in tables.
 */
enum class RowTier : int {
    Override = 0,
    Injected = 1,
};

struct GemmConfigPatch {
    /*
     * Row-file path or inline rows (one per line:
     * m_min m_max n_min n_max perf_class crosswise cta k_stages raster [k_tile]);
     * the addressed tier is replaced wholesale, empty string clears it.
     */
    std::optional<std::string> rows;
    std::optional<RowTier> tier; // which tier `rows` addresses
    // All row tiers off: the planner chain falls through to model/degraded.
    std::optional<bool> table_off;
    /*
     * 0/1/2/3 = table / hybrid / model / heuristic; -1 = unset (env seed decides, hybrid
     * unseeded).
     */
    std::optional<int> planner_mode;
    std::optional<bool> log;
    /*
     * Positive enables: false forces cp.async staging / knocks the sm_120a
     * block-scale cell out (the A/B knobs).
     */
    std::optional<bool> staging_tma;
    std::optional<bool> staging_mx;
};

/*
 * configure(patch) returns the resulting state; feeding it back re-installs it
 * exactly — save/restore round-trips in one call.
 */
struct GemmConfigState {
    std::string planner;   // resolved planner mode name
    int planner_mode = -1; // -1 = unset (env seed decides)
    /*
     * Installed row counts: 0 means the source parsed to no rows (e.g. a
     * mistyped path) rather than staying silent.
     */
    int override_rows = 0;
    int injected_rows = 0;
    std::string override_source; // what that tier was installed from
    std::string injected_source;

    bool log = false;
    bool table_off = false;
    bool staging_tma = true;
    bool staging_mx = true;
};

// Apply `patch` and return the resulting state in one call.
GemmConfigState configure(const GemmConfigPatch& patch);
GemmConfigState config_state();
bool parse_planner_mode(const std::string& name, int& out);

/*
 * The dispatch vocabulary, for the Python tooling and the sweep's C++ emitter
 * — neither keeps a second copy of the names.
 */

/*
 * Every ladder's recipes, deduped on the dispatch key, in dispatch order. A
 * row is (crosswise, ba, bb, cta, k_stages, k_tile, bm, bn, wm, wn, threads, smem);
 * its numbers spell the canonical name
 * Tile_<bm>x<bn>x<k_tile>_W<wm>x<wn>_S<k_stages> — the Python tooling's join key.
 */
std::vector<std::vector<int>> tile_vocabulary();

/*
 * TileClass spellings in enum order — a row's cta ordinal expands to these in
 * the compiled-in tables.
 */
std::vector<const char*> tile_class_names();

} // namespace gemm
} // namespace astrai
