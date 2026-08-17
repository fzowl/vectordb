#include "services/embedding_batching.hpp"

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

using vectordb::engine::BuildTokenAwareBatches;
using vectordb::engine::EstimateTokenCount;
using vectordb::engine::GetContextualizedChunkingParams;
using vectordb::engine::GetModelTokenLimit;
using vectordb::engine::kContextualizedDocChunkSize;

using Batches = std::vector<std::pair<size_t, size_t>>;

// --- Token-aware batching ---------------------------------------------------

TEST(TokenAwareBatching, SplitsAtTokenBoundary) {
  // Four items of 40 tokens each, token cap 100, item cap high: batches of two.
  std::vector<size_t> counts = {40, 40, 40, 40};
  auto batches = BuildTokenAwareBatches(counts, /*max_tokens=*/100, /*max_items=*/1000);
  Batches expected = {{0, 2}, {2, 4}};
  EXPECT_EQ(batches, expected);
}

TEST(TokenAwareBatching, StopsBeforeExceedingTokenCap) {
  // 60 + 60 = 120 > 100, so the second item starts a new batch.
  std::vector<size_t> counts = {60, 60, 10};
  auto batches = BuildTokenAwareBatches(counts, /*max_tokens=*/100, /*max_items=*/1000);
  Batches expected = {{0, 1}, {1, 3}};  // second batch: 60 + 10 = 70 <= 100
  EXPECT_EQ(batches, expected);
}

TEST(TokenAwareBatching, SingleOversizedTextGoesAlone) {
  // The 500-token item exceeds the cap on its own; it must still be emitted as
  // its own batch rather than looping forever or being dropped.
  std::vector<size_t> counts = {10, 500, 10};
  auto batches = BuildTokenAwareBatches(counts, /*max_tokens=*/100, /*max_items=*/1000);
  Batches expected = {{0, 1}, {1, 2}, {2, 3}};
  EXPECT_EQ(batches, expected);
}

TEST(TokenAwareBatching, FirstItemAlwaysAcceptedEvenIfOversized) {
  std::vector<size_t> counts = {1000000};
  auto batches = BuildTokenAwareBatches(counts, /*max_tokens=*/100, /*max_items=*/1000);
  Batches expected = {{0, 1}};
  EXPECT_EQ(batches, expected);
}

TEST(TokenAwareBatching, RespectsItemCountCap) {
  // Tokens well under the cap, but the item cap of 2 forces splitting.
  std::vector<size_t> counts = {1, 1, 1, 1, 1};
  auto batches = BuildTokenAwareBatches(counts, /*max_tokens=*/1000000, /*max_items=*/2);
  Batches expected = {{0, 2}, {2, 4}, {4, 5}};
  EXPECT_EQ(batches, expected);
}

TEST(TokenAwareBatching, EmptyInputYieldsNoBatches) {
  std::vector<size_t> counts = {};
  auto batches = BuildTokenAwareBatches(counts, /*max_tokens=*/100, /*max_items=*/10);
  EXPECT_TRUE(batches.empty());
}

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

// --- Per-model token limits -------------------------------------------------

TEST(ModelTokenLimit, KnownModels) {
  EXPECT_EQ(GetModelTokenLimit("voyage-4-lite"), 1000000u);
  EXPECT_EQ(GetModelTokenLimit("voyage-4"), 320000u);
  EXPECT_EQ(GetModelTokenLimit("voyage-4-large"), 120000u);
  EXPECT_EQ(GetModelTokenLimit("voyage-3.5-lite"), 1000000u);
  // Contextualized models: 120K total tokens with auto-chunking enabled.
  EXPECT_EQ(GetModelTokenLimit("voyage-context-4"), 120000u);
  EXPECT_EQ(GetModelTokenLimit("voyage-context-3"), 120000u);
}

TEST(ModelTokenLimit, StripsProviderPrefix) {
  EXPECT_EQ(GetModelTokenLimit("voyageai/voyage-4-lite"), 1000000u);
  EXPECT_EQ(GetModelTokenLimit("voyageai/voyage-4"), 320000u);
}

TEST(ModelTokenLimit, UnknownModelFallsBackToDefault) {
  // Models with no published batch token limit use the conservative fallback.
  EXPECT_EQ(GetModelTokenLimit("voyage-code-4"), 120000u);
  EXPECT_EQ(GetModelTokenLimit("voyage-4-nano"), 120000u);
  EXPECT_EQ(GetModelTokenLimit("voyageai/some-future-model"), 120000u);
}

// --- Token estimation -------------------------------------------------------

TEST(TokenEstimate, EmptyIsZero) {
  EXPECT_EQ(EstimateTokenCount(""), 0u);
}

TEST(TokenEstimate, ApproximatelyFourCharsPerToken) {
  EXPECT_EQ(EstimateTokenCount("abcd"), 1u);
  EXPECT_EQ(EstimateTokenCount("abcde"), 2u);  // rounds up
  EXPECT_EQ(EstimateTokenCount(std::string(400, 'x')), 100u);
}
