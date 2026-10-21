#pragma once
/* Host-only GEMM row contract shared by gemm/plan/planner.cpp and gemm/plan/table.cpp. */
#include <cstdint>
#include <optional>

#include <policy/gemm/config.h>
#include <policy/gemm/manifest.h>

namespace astrai {
namespace gemm {

inline constexpr int kTableRowK = 64;
struct TableRow {
    TileClass cta;
    int64_t m_min;
    int64_t m_max;
    int64_t n_min;
    int64_t n_max;
    int perf_class;
    int crosswise;
    int k_stages;
    int raster;
    int k_tile = kTableRowK;
    int64_t k_min = 0;
    int64_t k_max = 0;
    int min_ctas_per_sm = 0;
    int min_wave_permille = 0;
};

std::optional<TableRow> plan_override_row(const PlanQuery& q);
std::optional<TableRow> plan_injected_row(const PlanQuery& q);
std::optional<TableRow> plan_builtin_row(const PlanQuery& q);
bool builtin_rows_match_device(const DeviceFacts& dev);
const TableRow* builtin_plan_table(int perf_class, int& count);
int plan_resident_ctas(TileClass cta, int k_stages, int k_tile, const PlanQuery& q);
int gemm_planner_mode();
bool gemm_table_off();

} // namespace gemm
} // namespace astrai
