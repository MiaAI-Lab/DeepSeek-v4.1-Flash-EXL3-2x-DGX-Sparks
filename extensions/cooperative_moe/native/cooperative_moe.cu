// Isolated DS4.1 fixed-shape specialization of the upstream two-stage MoE kernel.
// Compile-time constants match this C ABI; the parameter struct is unchanged.
#include <cuda_runtime.h>
#include <cmath>
#define GOAL50_COOP_NATIVE_ONLY 1
#define EXL3_MOE_COOP_DEFINE_ROT 1
#include "quant/goal50_fixed_coop_kernel.cuh"

namespace ns = goal50_fixed_coop_ns;
constexpr int H = 5120, I = 1152, TOPK = 6, ROWS_MAX = 8;
constexpr int SLOTS_MAX = TOPK * ROWS_MAX;
static bool prepared[3][3] = {};

struct Selected { void* a; void* b; int sa; int sb; bool wa; bool wb; };

template<int K> Selected select_t(int geometry) {
    // Match upstream Blackwell auto selection: Hi>=4096 wide; I<2048 narrow.
    bool wa = geometry != 0, wb = geometry == 1;
    return {
        wa ? (void*) ns::exl3_moe_coop_a_kernel<K,2,true>
           : (void*) ns::exl3_moe_coop_a_kernel<K,2,false>,
        wb ? (void*) ns::exl3_moe_coop_b_kernel<K,2,true>
           : (void*) ns::exl3_moe_coop_b_kernel<K,2,false>,
        ns::smem_a_bytes<K>(H), ns::smem_b_bytes<K>(), wa, wb
    };
}
Selected select(int bits, int geometry) {
    if (bits==2) return select_t<2>(geometry);
    if (bits==3) return select_t<3>(geometry);
    return select_t<4>(geometry);
}

extern "C" int goal50_coop_abi() { return 1; }
extern "C" int goal50_coop_info(int bits, int geometry, int* info) {
    if (!info || bits<2 || bits>4 || geometry<0 || geometry>2) return int(cudaErrorInvalidValue);
    int device;
    cudaError_t err=cudaGetDevice(&device); if(err!=cudaSuccess) return int(err);
    cudaDeviceProp prop;
    err=cudaGetDeviceProperties(&prop,device); if(err!=cudaSuccess) return int(err);
    if(prop.major!=12 || prop.minor!=1) return int(cudaErrorInvalidDevice);
    Selected s=select(bits,geometry);
    void* funcs[3]={s.a,s.b,(void*)ns::exl3_moe_coop_rot_kernel};
    int smems[3]={s.sa,s.sb,0};
    for(int n=0;n<3;n++) {
        if(smems[n]>48*1024) {
            err=cudaFuncSetAttribute(funcs[n],cudaFuncAttributeMaxDynamicSharedMemorySize,smems[n]);
            if(err!=cudaSuccess) return int(err);
        }
        cudaFuncAttributes attr;
        err=cudaFuncGetAttributes(&attr,funcs[n]); if(err!=cudaSuccess) return int(err);
        int blocks;
        err=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,funcs[n],MOE_COOP_THREADS,smems[n]);
        if(err!=cudaSuccess) return int(err);
        if(blocks<1) return int(cudaErrorLaunchOutOfResources);
        info[n*5+0]=MOE_COOP_THREADS; info[n*5+1]=smems[n];
        info[n*5+2]=attr.numRegs; info[n*5+3]=int(attr.localSizeBytes); info[n*5+4]=blocks;
    }
    info[15]=prop.multiProcessorCount;
    info[16]=int(exl3_moe_coop_ctr_len(SLOTS_MAX,ROWS_MAX,I,H));
    info[17]=int(sizeof(MoeCoopParams));
    prepared[bits-2][geometry]=true;
    return 0;
}

// Pointer order: x, ids, weights; 9 gate/up/down tables (trellis,suh,svh);
// had_gate,had_up,gu_gate,gu_up,activation,down_partials,counters,output.
// Buffers have capacity 48 routed slots and 8 output rows, prepared before capture.
extern "C" int goal50_coop_launch(void** t, int bits, int rows, int experts,
    float limit, int geometry, int gu_f32, void* stream_ptr) {
    if(!t || bits<2 || bits>4 || rows<1 || rows>ROWS_MAX || experts<1 || experts>384 ||
       geometry<0 || geometry>2 || gu_f32!=0 || !std::isfinite(limit) || limit<0 ||
       !prepared[bits-2][geometry]) return int(cudaErrorInvalidValue);
    for(int n=0;n<20;n++) if(!t[n]) return int(cudaErrorInvalidValue);
    MoeCoopParams p={};
    p.x=(half*)t[0]; p.sel=(int64_t*)t[1]; p.rw=(half*)t[2];
    p.g_trellis=(int64_t*)t[3]; p.g_suh=(int64_t*)t[4]; p.g_svh=(int64_t*)t[5];
    p.u_trellis=(int64_t*)t[6]; p.u_suh=(int64_t*)t[7]; p.u_svh=(int64_t*)t[8];
    p.d_trellis=(int64_t*)t[9]; p.d_suh=(int64_t*)t[10]; p.d_svh=(int64_t*)t[11];
    p.had_g=(half*)t[12]; p.had_u=(half*)t[13]; p.gu_g=t[14]; p.gu_u=t[15];
    p.act_out=(half*)t[16]; p.d_out=(float*)t[17]; p.ctr_a=(int*)t[18]; p.out=(float*)t[19];
    p.x_stride=H; p.bsz=rows; p.topk=TOPK; p.H=H; p.Hi=H; p.I=I; p.Ho=H; p.H_out=H;
    p.min_expert=0; p.max_expert=experts; p.n_local=experts;
    p.act=MOE_COOP_ACT_SILU; p.act_limit=limit; p.gated=true;
    p.a_global=rows>1; p.gu_f32=bool(gu_f32); p.ksplit_a=p.ksplit_b=1;
    p.slots_max=SLOTS_MAX; p.rows_max=ROWS_MAX; p.out_stride=H;
    p.ctr_a_len=SLOTS_MAX*(I/128); p.ctr_b_len=ROWS_MAX*(H/128);
    p.ctr_b=p.ctr_a+p.ctr_a_len; p.runs=p.ctr_b+p.ctr_b_len;
    Selected s=select(bits,geometry);
    cudaStream_t stream=(cudaStream_t)stream_ptr;
    void* args[]={&p};
    if(p.a_global) {
        int items=rows*TOPK*(H/128)*2;
        ns::exl3_moe_coop_rot_kernel<<<CEIL_DIVIDE(items,MOE_COOP_THREADS/32),MOE_COOP_THREADS,0,stream>>>(p);
        auto err=cudaGetLastError(); if(err!=cudaSuccess) return int(err);
    }
    int ga=rows*TOPK*2*(I/(s.wa?128:MOE_COOP_COLS));
    int gb=rows*TOPK*(H/(s.wb?128:MOE_COOP_COLS));
    auto err=cudaLaunchKernel(s.a,dim3(ga),dim3(MOE_COOP_THREADS),args,s.sa,stream);
    if(err!=cudaSuccess) return int(err);
    err=cudaLaunchKernel(s.b,dim3(gb),dim3(MOE_COOP_THREADS),args,s.sb,stream);
    if(err!=cudaSuccess) return int(err);
    return int(cudaGetLastError());
}
