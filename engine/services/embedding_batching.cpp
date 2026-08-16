#include "services/embedding_batching.hpp"

#include <unordered_map>

namespace vectordb {
namespace engine {

size_t EstimateTokenCount(const std::string& text) {
  if (text.empty()) {
    return 0;
  }
  // ~4 characters per token, rounded up.
  return (text.size() + 3) / 4;
}

size_t GetModelTokenLimit(const std::string& model_name) {
  // Strip an optional "provider/" prefix, e.g. "voyageai/voyage-4-lite".
  std::string name = model_name;
  auto slash = name.find('/');
  if (slash != std::string::npos) {
    name = name.substr(slash + 1);
  }

  // Per-request total token limits from the VoyageAI docs. Anything not listed
  // falls back to the conservative 120000 default below.
  static const std::unordered_map<std::string, size_t> limits = {
      {"voyage-4-lite", 1000000},
      {"voyage-4", 320000},
      {"voyage-3.5-lite", 1000000},
      {"voyage-3.5", 320000},
      {"voyage-3-lite", 1000000},
      {"voyage-3", 320000},
      {"voyage-2", 320000},
      {"voyage-02", 320000},
  };
  auto it = limits.find(name);
  return it != limits.end() ? it->second : 120000;
}

std::vector<std::pair<size_t, size_t>> BuildTokenAwareBatches(
    const std::vector<size_t>& token_counts,
    size_t max_tokens_per_batch,
    size_t max_items_per_batch) {
  std::vector<std::pair<size_t, size_t>> batches;
  const size_t n = token_counts.size();
  size_t i = 0;
  while (i < n) {
    const size_t start = i;
    size_t batch_tokens = 0;
    while (i < n) {
      // Item-count cap.
      if (i - start >= max_items_per_batch) {
        break;
      }
      const size_t t = token_counts[i];
      // Token cap, but always allow at least one item so a single oversized
      // text still goes through on its own.
      if (i > start && batch_tokens + t > max_tokens_per_batch) {
        break;
      }
      batch_tokens += t;
      ++i;
    }
    batches.emplace_back(start, i);
  }
  return batches;
}

ContextualizedChunkingParams GetContextualizedChunkingParams(
    const std::string& input_type) {
  ContextualizedChunkingParams params;
  // Auto-chunking is only valid for document inputs; the API rejects it for
  // queries.
  params.enable_auto_chunking = (input_type != "query");
  params.has_chunk_size = params.enable_auto_chunking;
  params.chunk_size = params.enable_auto_chunking ? kContextualizedDocChunkSize : 0;
  return params;
}

}  // namespace engine
}  // namespace vectordb
