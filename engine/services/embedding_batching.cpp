#include "services/embedding_batching.hpp"

namespace vectordb {
namespace engine {

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
