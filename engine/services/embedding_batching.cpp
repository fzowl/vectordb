#include "services/embedding_batching.hpp"

#include <unordered_map>

namespace vectordb {
namespace engine {

size_t EstimateTokenCount(const std::string& text) {
  if (text.empty()) {
    return 0;
  }
  // ~4 characters per token, rounded up. This is the widely used heuristic for
  // English-like text and is only used to build token-aware batches, so a
  // slightly low estimate simply yields larger (but still valid) batches.
  //
  // Limitation: text.size() counts bytes, not characters. For CJK/multibyte
  // content this OVER-counts bytes per character but still UNDER-counts tokens
  // (such scripts pack far fewer characters per token), so a batch built under
  // the limit can still exceed the true server-side token count and, combined
  // with the fixed retry budget, fail. If that becomes a problem for
  // non-Latin-heavy workloads, lower the divisor here (a more conservative
  // estimate just yields smaller batches).
  return (text.size() + 3) / 4;
}

size_t GetModelTokenLimit(const std::string& model_name) {
  // Strip an optional "provider/" prefix, e.g. "voyageai/voyage-4-lite".
  std::string name = model_name;
  auto slash = name.find('/');
  if (slash != std::string::npos) {
    name = name.substr(slash + 1);
  }

  // Per-request TOTAL token limits ("batch token limit") from the VoyageAI docs
  // (https://docs.voyageai.com/docs/embeddings and .../contextualized-chunk-embeddings).
  // Models whose batch token limit is not published there are intentionally
  // omitted and fall back to the conservative 120000 default below (safe: it
  // only yields smaller batches).
  static const std::unordered_map<std::string, size_t> limits = {
      // Current generation.
      {"voyage-4-large", 120000},
      {"voyage-4", 320000},
      {"voyage-4-lite", 1000000},
      {"voyage-finance-2", 120000},
      {"voyage-law-2", 120000},
      // Contextualized models: 120K total tokens when auto-chunking is enabled.
      {"voyage-context-4", 120000},
      {"voyage-context-3", 120000},
      // Previous generation (still API-accessible).
      {"voyage-3-large", 120000},
      {"voyage-3.5", 320000},
      {"voyage-3.5-lite", 1000000},
      {"voyage-code-3", 120000},
      {"voyage-multilingual-2", 120000},
      {"voyage-large-2-instruct", 120000},
      {"voyage-2", 320000},
      // Note: voyage-code-4, voyage-4-nano, voyage-3, voyage-3-lite and
      // voyage-large-2 do not publish a batch token limit and use the fallback.
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
