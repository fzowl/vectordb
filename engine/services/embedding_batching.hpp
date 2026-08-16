#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace vectordb {
namespace engine {

// Maximum number of inputs sent to the embedding provider in a single request.
// VoyageAI accepts up to 1000 inputs per request; this mirrors the reference
// client's default batch size.
constexpr size_t EmbeddingMaxBatchSize = 1000;

// Target chunk size (in tokens) used for VoyageAI contextualized document
// inputs. It equals the per-chunk context window of the voyage-context-* models,
// so that each independent input string resolves to exactly one chunk and hence
// exactly one embedding. See GetContextualizedChunkingParams for details.
constexpr int kContextualizedDocChunkSize = 32000;

// Rough token-count estimate for a text. There is no server-side tokenizer
// available here, so we approximate ~4 characters per token (the widely used
// heuristic for English-like text). Used only to build token-aware batches; a
// slightly conservative estimate simply yields smaller batches.
size_t EstimateTokenCount(const std::string& text);

// Per-request total token limit for a model, matching VoyageAI's documented
// limits. Accepts names with an optional "provider/" prefix
// (e.g. "voyageai/voyage-4-lite"). Unknown models fall back to 120000, the
// conservative default also used by the reference client.
size_t GetModelTokenLimit(const std::string& model_name);

// Token-aware batching of a list of texts (given their estimated token counts)
// into contiguous [start, end) index ranges. A batch is closed before adding an
// item that would push the running token total past max_tokens_per_batch, unless
// the batch is still empty (so a single oversized text is emitted alone). A
// batch is also closed once it reaches max_items_per_batch items. Mirrors the
// reference client's greedy _build_batches logic.
std::vector<std::pair<size_t, size_t>> BuildTokenAwareBatches(
    const std::vector<size_t>& token_counts,
    size_t max_tokens_per_batch,
    size_t max_items_per_batch);

// Chunking parameters for a VoyageAI contextualized embedding request.
struct ContextualizedChunkingParams {
  bool enable_auto_chunking;  // whether to send enable_auto_chunking=true
  bool has_chunk_size;        // whether chunk_size should be sent at all
  int chunk_size;             // meaningful only when has_chunk_size is true
};

// Decide the contextualized chunking parameters for a given input_type.
//
// Design (deliberate): every input string is embedded as its OWN independent
// document. Document-side inputs are sent as a flat list of strings with
// enable_auto_chunking=true and chunk_size=32000, so each string resolves to
// exactly one chunk and one deterministic per-input embedding. Cross-input
// contextualization is intentionally NOT used, because generic embed callers
// pass unrelated texts.
//
// The API rejects auto-chunking when input_type is "query", so the query path
// must not send it (and must not send chunk_size).
ContextualizedChunkingParams GetContextualizedChunkingParams(
    const std::string& input_type);

}  // namespace engine
}  // namespace vectordb
