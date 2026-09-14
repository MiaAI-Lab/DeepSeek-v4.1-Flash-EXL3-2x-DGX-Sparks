// Host side for the exl3_moe_x variants. Self-contained: no DevCtx; the lock/scheduler buffer is
// passed in from Python (int32, zeroed once, self-resetting like the shipped kernel's).
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../util.h"
#include "../util.cuh"
#include "exl3_moe_x.cuh"

struct MoexVariant { int tk, shs, frs, minb, bits, tn; void* kernel; int smem; const char* name; };

template<int TK, int SHS, int FRS, int MINB, int BITS, int TN>
constexpr int moex_smem_bytes()
{
    constexpr int a_stage = 16 * TK;                          // halfs
    constexpr int b_stage = (TK / 16) * (TN / 16) * 16 * BITS; // uint16
    constexpr int frags_n = 2 * (TN / 16) / 8;
    constexpr int c_size = 4 * 256 * frags_n;                 // floats
    return SHS * (2 * a_stage + 2 * b_stage) + 4 * c_size;
}

#define MOEX_VARIANT(BITS, TK, SHS, FRS, MINB) \
    { TK, SHS, FRS, MINB, BITS, 128, (void*) exl3_moe_x_kernel<BITS, 128, 2, TK, SHS, FRS, MINB>, \
      moex_smem_bytes<TK, SHS, FRS, MINB, BITS, 128>(), "k" #BITS "_tk" #TK "_shs" #SHS "_frs" #FRS "_minb" #MINB }

// Round-2 outcome (results/screen/perf/moex-round2.txt): two shapes are within ~10 % of the
// box's sustainable bandwidth; runtime-K (switch) instances spill and lose the whole gain, so
// every K this model uses gets its own compile-time instance. Variants are addressed by
// (shape, K): shape 0 = shipped (K=3 only, reference), shape 1 = (32,8,2,1) 1 block/SM,
// shape 2 = (32,6,2,2) 2 blocks/SM. exl3_moe_x(..., shape, groups, group_size) picks K itself.
static MoexVariant g_variants[] =
{
    MOEX_VARIANT(3, 32, 3, 3, 1),    // 0: shape 0, K=3 (shipped shape, reference)
    MOEX_VARIANT(2, 32, 8, 2, 1),    // 1: shape 1, K=2
    MOEX_VARIANT(3, 32, 8, 2, 1),    // 2: shape 1, K=3
    MOEX_VARIANT(4, 32, 8, 2, 1),    // 3: shape 1, K=4
    MOEX_VARIANT(2, 32, 6, 2, 2),    // 4: shape 2, K=2
    MOEX_VARIANT(3, 32, 6, 2, 2),    // 5: shape 2, K=3
    MOEX_VARIANT(4, 32, 6, 2, 2),    // 6: shape 2, K=4
};
static int moex_variant_for(int shape, int K)
{
    if (shape == 0) return K == 3 ? 0 : -1;
    if (shape == 1) return K == 2 ? 1 : K == 3 ? 2 : K == 4 ? 3 : -1;
    if (shape == 2) return K == 2 ? 4 : K == 3 ? 5 : K == 4 ? 6 : -1;
    return -1;
}
static const int g_num_variants = sizeof(g_variants) / sizeof(g_variants[0]);
static bool g_attr_set[64] = {};

std::vector<int64_t> moex_info(int variant)
{
    TORCH_CHECK(variant >= 0 && variant < g_num_variants, "bad variant");
    MoexVariant& v = g_variants[variant];
    int block = 256 * v.tk / 16;
    if (!g_attr_set[variant])
    {
        cudaFuncSetAttribute(v.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, v.smem);
        g_attr_set[variant] = true;
    }
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, v.kernel);
    int max_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, v.kernel, block, v.smem);
    int device; cudaGetDevice(&device);
    int num_sms; cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    return { (int64_t) attr.numRegs, (int64_t) v.smem, (int64_t) max_blocks, (int64_t) block, (int64_t) num_sms, (int64_t) attr.localSizeBytes, (int64_t) attr.maxThreadsPerBlock };
}

std::string moex_name(int variant) { return g_variants[variant].name; }
int moex_num_variants() { return g_num_variants; }

void exl3_moe_x
(
    const at::Tensor& hidden_state, const at::Tensor& output_state, const at::Tensor& expert_count,
    const at::Tensor& token_sorted, const at::Tensor& weight_sorted,
    const at::Tensor& temp_state_g, const at::Tensor& temp_state_u,
    const at::Tensor& temp_intermediate_g, const at::Tensor& temp_intermediate_u,
    const int act_function, const int K_gate, const int K_up, const int K_down,
    const at::Tensor& gate_ptrs_trellis, const at::Tensor& gate_ptrs_suh, const at::Tensor& gate_ptrs_svh,
    const at::Tensor& up_ptrs_trellis, const at::Tensor& up_ptrs_suh, const at::Tensor& up_ptrs_svh,
    const at::Tensor& down_ptrs_trellis, const at::Tensor& down_ptrs_suh, const at::Tensor& down_ptrs_svh,
    const bool gate_mcg, const bool gate_mul1, const bool up_mcg, const bool up_mul1,
    const bool down_mcg, const bool down_mul1, const float act_limit,
    const at::Tensor& locks_buf, const int variant_in, const int num_groups, const int group_size
)
{
    const at::cuda::OptionalCUDAGuard device_guard(hidden_state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(K_gate == K_up && K_up == K_down, "moex: gate/up/down must share K");
    const int shape = variant_in;
    const int variant = moex_variant_for(shape, K_gate);
    TORCH_CHECK(variant >= 0, "moex: no instance for shape ", shape, " K=", K_gate);
    MoexVariant& v = g_variants[variant];
    TORCH_CHECK(gate_mul1 && up_mul1 && down_mul1 && !gate_mcg, "moex: mul1 codebook only");
    TORCH_CHECK_DTYPE(hidden_state, kHalf);
    TORCH_CHECK_DTYPE(output_state, kFloat);
    TORCH_CHECK_DTYPE(locks_buf, kInt);
    TORCH_CHECK(locks_buf.numel() >= MOE_SCHED_OFFSET + MOE_SCHED_INTS, "locks buffer too small");
    size_t hidden_dim = hidden_state.size(1);
    size_t num_experts = expert_count.size(0) - 1;
    size_t bsz = hidden_state.size(0);
    size_t num_experts_per_tok = token_sorted.size(0) / bsz;
    size_t max_tokens_per_expert = temp_state_g.size(1);
    size_t concurrency = temp_state_g.size(0);
    size_t intermediate_dim = temp_intermediate_g.size(2);
    TORCH_CHECK(hidden_dim % 128 == 0 && intermediate_dim % 128 == 0, "dims must be multiples of 128");
    TORCH_CHECK((int) concurrency >= num_groups, "temps hold fewer groups than launched");
    TORCH_CHECK(num_groups <= MOE_MAX_GROUPS, "too many groups");

    int block = 256 * v.tk / 16;
    if (!g_attr_set[variant])
    {
        cudaFuncSetAttribute(v.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, v.smem);
        g_attr_set[variant] = true;
    }
    int max_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, v.kernel, block, v.smem);
    int device; cudaGetDevice(&device);
    int num_sms; cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    TORCH_CHECK(num_groups * group_size <= max_blocks * num_sms,
                "grid ", num_groups * group_size, " blocks cannot be co-resident (", max_blocks, " blocks/SM x ", num_sms, " SMs)");

    const half* _hidden_state = (const half*) hidden_state.data_ptr();
    half* _tsg = (half*) temp_state_g.data_ptr(); half* _tsu = (half*) temp_state_u.data_ptr();
    half* _tig = (half*) temp_intermediate_g.data_ptr(); half* _tiu = (half*) temp_intermediate_u.data_ptr();
    float* _out = (float*) output_state.data_ptr();
    const uint16_t** _gt = (const uint16_t**) gate_ptrs_trellis.data_ptr();
    const half** _gsuh = (const half**) gate_ptrs_suh.data_ptr(); const half** _gsvh = (const half**) gate_ptrs_svh.data_ptr();
    const uint16_t** _ut = (const uint16_t**) up_ptrs_trellis.data_ptr();
    const half** _usuh = (const half**) up_ptrs_suh.data_ptr(); const half** _usvh = (const half**) up_ptrs_svh.data_ptr();
    const uint16_t** _dt = (const uint16_t**) down_ptrs_trellis.data_ptr();
    const half** _dsuh = (const half**) down_ptrs_suh.data_ptr(); const half** _dsvh = (const half**) down_ptrs_svh.data_ptr();
    const int64_t* _ec = (const int64_t*) expert_count.data_ptr();
    const int64_t* _ts = (const int64_t*) token_sorted.data_ptr();
    const half* _ws = (const half*) weight_sorted.data_ptr();
    int* _locks = (int*) locks_buf.data_ptr();
    dim3 grid(group_size, 1, num_groups);

    #define LAUNCH(BITS, TK, SHS, FRS, MINB) \
        exl3_moe_x_kernel<BITS, 128, 2, TK, SHS, FRS, MINB><<<grid, block, v.smem, stream>>>( \
            _hidden_state, _tsg, _tsu, _tig, _tiu, _out, _gt, _gsuh, _gsvh, _ut, _usuh, _usvh, _dt, _dsuh, _dsvh, \
            _ec, _ts, _ws, (int) hidden_dim, (int) intermediate_dim, (int) num_experts, (int) num_experts_per_tok, \
            (int) max_tokens_per_expert, num_groups, act_limit, act_function, K_gate, K_up, K_down, _locks)
    switch (variant)
    {
        case 0: LAUNCH(3, 32, 3, 3, 1); break;
        case 1: LAUNCH(2, 32, 8, 2, 1); break;
        case 2: LAUNCH(3, 32, 8, 2, 1); break;
        case 3: LAUNCH(4, 32, 8, 2, 1); break;
        case 4: LAUNCH(2, 32, 6, 2, 2); break;
        case 5: LAUNCH(3, 32, 6, 2, 2); break;
        case 6: LAUNCH(4, 32, 6, 2, 2); break;
    }
    #undef LAUNCH
    cuda_check(cudaPeekAtLastError());
}

int moex_locks_ints() { return MOE_SCHED_OFFSET + MOE_SCHED_INTS; }
int moex_variant_index(int shape, int K) { return moex_variant_for(shape, K); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("exl3_moe_x", &exl3_moe_x, "exl3_moe_x");
    m.def("moex_info", &moex_info, "moex_info");
    m.def("moex_name", &moex_name, "moex_name");
    m.def("moex_num_variants", &moex_num_variants, "moex_num_variants");
    m.def("moex_locks_ints", &moex_locks_ints, "moex_locks_ints");
    m.def("moex_variant_index", &moex_variant_index, "moex_variant_index");
}
