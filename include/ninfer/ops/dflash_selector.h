#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstdint>

namespace ninfer::ops {

// DFlash 2 path-selector family (three ops, one header): the 27B drafter proposes a 16-wide
// candidate path per draft column and walks the k dependent draft positions through precomputed
// 16x16 edge-score tables. All three ops run over draft columns C = k*B only, with k in [1,7]
// draft tokens and B in [1,8] batch rows; the exact registered domain is the set
// {1,2,3,4,5,6,7,8,9,10,12,14,15,16,18,20,21,24,25,28,30,32,35,36,40,42,48,49,56}. The anchor
// column's logits and hidden state are never fed to this family. The vocabulary is fixed at
// 248320 rows, the candidate width at 16, and the codebook rank at 256. Every op is
// graph-capturable (no host syncs, no device-side allocation), writes all of its output, reads
// its inputs unchanged, and owns no workspace or persistent state.

/**
 * Op: dflash_selector_topk
 *
 * Per-column top-16 over the full 248320-row vocab. For each draft column c the 16 (value,
 * token) pairs rank by value descending with equal values broken by lower token id (the smaller
 * id wins and occupies the higher-rank slot), the same candidate ordering the sampling op uses:
 *
 *   candidates[j,c] = j-th ranked token id,              0 <= j < 16
 *   unary[j,c]      = float(logits[candidates[j,c],c])
 *
 * `logits` is contiguous BF16 [248320,C] (full-vocab draft logits from the target output head);
 * `candidates` is contiguous I32 [16,C] (rank 0 = best); `unary` is contiguous FP32 [16,C]
 * (the candidates' logits, upcast to FP32). The registered domain is the exact set C = k*B with
 * k in [1,7] and B in [1,8] (family note above). Inputs are unchanged; every output element is
 * overwritten.
 */
void dflash_selector_topk(const Tensor& logits, Tensor& candidates, Tensor& unary,
                          cudaStream_t stream);

/**
 * Op: dflash_selector_scores
 *
 * Edge scores for the dependent walk. For draft position t, predecessor slot p, candidate slot
 * c, and batch row b:
 *
 *   scores[t,p,c,b] = unary[c, t*B+b]
 *         + sum_d pred_codebook[pred(p,t),d] * hidden_proj[d, t*B+b] * succ_codebook[cand, d]
 *
 * where d ranges over [0,256) and cand = candidates[t*B+b, c]. `hidden_proj` is the draft
 * hidden already projected to the 256-rank space, contiguous BF16 [256,k*B], indexed by batch
 * row (not by candidate). The predecessor chain is
 *
 *   pred(p,t) = anchors[b]                when t = 0
 *              = candidates[(t-1)*B+b, p] when t > 0
 *
 * At t = 0 every p-slot reads the same predecessor row, so scores[0,p,c,b] is the same value
 * for all p; all 16 slots are materialized so the walk can read any of them.
 *
 * `candidates` is contiguous I32 [16,k*B]; `anchors` is contiguous I32 [B]; `unary` is
 * contiguous FP32 [16,k*B]; `pred_codebook` and `succ_codebook` are contiguous BF16 [248320,256];
 * `scores` is contiguous FP32 [k,16,16,B] with k*B equal to the draft-column count C. The
 * compute is FP32 end-to-end: the BF16 codebook and hidden rows are upcast before the
 * rank-256 contraction (the reference computes the einsum in model-dtype BF16; the FP32
 * arithmetic is the deliberate precision hardening, spec doc 06 section 7). The registered
 * domain is the exact set C = k*B with k in [1,7] and B in [1,8]. Inputs are unchanged; every
 * scores element is overwritten.
 */
void dflash_selector_scores(const Tensor& candidates, const Tensor& hidden_proj,
                            const Tensor& anchors, const Tensor& unary,
                            const Tensor& pred_codebook, const Tensor& succ_codebook,
                            Tensor& scores, cudaStream_t stream);

/**
 * Op: dflash_selector_walk
 *
 * Walks the k dependent draft positions, one 16-candidate path per batch row. For row b with
 * pred = 0 (the anchor slot) and t = 0..k-1:
 *
 *   row[c] = scores[t,pred,c,b],          0 <= c < 16
 *
 *   greedy (configs[b].temperature <= 0):
 *     idx = argmax_c row[c], ties broken by the lower candidates[t*B+b,c] token id
 *   sampled (temperature > 0):
 *     idx = one multinomial draw over softmax(row[c]/temperature) on the 16 slots
 *
 *   drafts[t*B+b] = candidates[t*B+b, idx];  pred = idx
 *
 * `scores` is contiguous FP32 [k,16,16,B]; `candidates` is contiguous I32 [16,k*B]; `drafts` is
 * contiguous I32 [k*B]. `configs` is a HOST pointer to B SamplingConfig rows; only temperature
 * and seed are consumed (temperature-only sampling is the deliberate divergence of spec doc 06
 * decision D8: no top_p/top_k/min_p/penalties) and the launcher reads the B rows once during
 * launch setup into kernel parameters, so the launch stays graph-capturable. Sampled draws use
 * the counter-based RNG idiom with purpose kSamplePurposeDflashSelector: the draw for row b at
 * draft position t is
 *
 *   sampling_uniform(configs[b].seed, b, kSamplePurposeDflashSelector, t)
 *
 * (the row in the position slot, the draft position in the subkey slot, following the
 * sampling-draw subkeying) picked by inverse-CDF over the 16 softmax weights in the same
 * convention as the sampling op. Greedy rows perform no draws, so greedy identity is
 * unaffected. One kernel, one CTA per batch row (B <= 8), at most k <= 7 sequential steps
 * inside (spec doc 06 decision D9). The registered domain is the exact set k*B with k in [1,7]
 * and B in [1,8].
 */
void dflash_selector_walk(const Tensor& scores, const Tensor& candidates,
                          const SamplingConfig* configs, Tensor& drafts, cudaStream_t stream);

} // namespace ninfer::ops