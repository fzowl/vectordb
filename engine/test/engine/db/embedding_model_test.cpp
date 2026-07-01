#include "utils/common_util.hpp"

#include <gtest/gtest.h>

#include <string>

using vectordb::server::CommonUtil;

TEST(VoyageAIModelRouting, ContextualizedModels) {
  // voyage-context-4 (and its predecessor) route to the contextualized endpoint.
  EXPECT_TRUE(CommonUtil::IsVoyageAIContextualizedModel("voyage-context-4"));
  EXPECT_TRUE(CommonUtil::IsVoyageAIContextualizedModel("voyage-context-3"));
  EXPECT_TRUE(CommonUtil::IsVoyageAIContextualizedModel("voyageai/voyage-context-4"));

  EXPECT_FALSE(CommonUtil::IsVoyageAIContextualizedModel("voyage-multimodal-3"));
  EXPECT_FALSE(CommonUtil::IsVoyageAIContextualizedModel("voyage-3"));
  EXPECT_FALSE(CommonUtil::IsVoyageAIContextualizedModel("openai/text-embedding-3-small"));
}

TEST(VoyageAIModelRouting, MultimodalModels) {
  EXPECT_TRUE(CommonUtil::IsVoyageAIMultimodalModel("voyage-multimodal-3"));
  EXPECT_TRUE(CommonUtil::IsVoyageAIMultimodalModel("voyageai/voyage-multimodal-3"));

  EXPECT_FALSE(CommonUtil::IsVoyageAIMultimodalModel("voyage-context-4"));
  EXPECT_FALSE(CommonUtil::IsVoyageAIMultimodalModel("voyage-3"));
}
