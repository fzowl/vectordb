#include "services/embedding_batching.hpp"

#include <gtest/gtest.h>

using vectordb::engine::GetContextualizedChunkingParams;
using vectordb::engine::kContextualizedDocChunkSize;

// --- Contextualized chunking parameters -------------------------------------

TEST(ContextualizedChunking, DocumentPathEnablesAutoChunking) {
  auto params = GetContextualizedChunkingParams("document");
  EXPECT_TRUE(params.enable_auto_chunking);
  EXPECT_TRUE(params.has_chunk_size);
  EXPECT_EQ(params.chunk_size, kContextualizedDocChunkSize);
  EXPECT_EQ(params.chunk_size, 32000);
}

TEST(ContextualizedChunking, QueryPathDisablesAutoChunking) {
  // The API rejects auto-chunking for queries, so it must not be sent, and
  // chunk_size must be dropped.
  auto params = GetContextualizedChunkingParams("query");
  EXPECT_FALSE(params.enable_auto_chunking);
  EXPECT_FALSE(params.has_chunk_size);
}
