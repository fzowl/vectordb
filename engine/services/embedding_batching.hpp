#pragma once

#include <string>

namespace vectordb {
namespace engine {

// Target chunk size (in tokens) used for VoyageAI contextualized document
// inputs. It equals the per-chunk context window of the voyage-context-* models,
// so that each independent input string resolves to exactly one chunk and hence
// exactly one embedding. See GetContextualizedChunkingParams for details.
constexpr int kContextualizedDocChunkSize = 32000;

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
