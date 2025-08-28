use crate::types::*;
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

/// Trait for chat-based text generation providers
#[async_trait]
pub trait ChatTextGeneration: Send + Sync {
    /// Get the provider's name/identifier
    fn name(&self) -> &str;

    /// Get the model being used
    fn model(&self) -> &str;

    /// Generate a single chat response (non-streaming)
    async fn generate(&self, request: ChatRequest) -> Result<ChatResponse>;

    /// Generate a streaming chat response
    async fn generate_stream(
        &self,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatStreamChunk>> + Send>>>;

    /// Check if the provider supports tool calling
    fn supports_tools(&self) -> bool {
        false
    }

    /// Check if the provider supports vision/images
    fn supports_vision(&self) -> bool {
        false
    }

    /// Check if the provider supports system messages
    fn supports_system_messages(&self) -> bool {
        true
    }

    /// Get maximum token limit for this provider/model
    fn max_tokens(&self) -> Option<u32> {
        Some(4096)
    }

    /// Validate that a request is compatible with this provider
    fn validate_request(&self, request: &ChatRequest) -> Result<()> {
        if request.tools.is_some() && !self.supports_tools() {
            return Err(AiError::InvalidRequest {
                message: format!("Provider {} does not support tool calling", self.name()),
            });
        }

        // Check for unsupported message types and content
        for message in &request.messages {
            match message {
                Message::System { .. } if !self.supports_system_messages() => {
                    return Err(AiError::InvalidRequest {
                        message: format!(
                            "Provider {} does not support system messages",
                            self.name()
                        ),
                    });
                }
                Message::User { content, .. } => {
                    for part in content {
                        if let UserContent::Image { .. } = part
                            && !self.supports_vision()
                        {
                            return Err(AiError::InvalidRequest {
                                message: format!(
                                    "Provider {} does not support vision/images",
                                    self.name()
                                ),
                            });
                        }
                    }
                }
                Message::Assistant { content, .. } => {
                    for part in content {
                        if let AssistantContent::ToolCall { .. } = part
                            && !self.supports_tools()
                        {
                            return Err(AiError::InvalidRequest {
                                message: format!(
                                    "Provider {} does not support tool calls",
                                    self.name()
                                ),
                            });
                        }
                    }
                }
                Message::Tool { .. } if !self.supports_tools() => {
                    return Err(AiError::InvalidRequest {
                        message: format!("Provider {} does not support tool results", self.name()),
                    });
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// Trait for embedding generation providers
#[async_trait]
pub trait EmbeddingGeneration: Send + Sync {
    /// Get the provider's name/identifier
    fn name(&self) -> &str;

    /// Get the model being used
    fn model(&self) -> &str;

    /// Generate embeddings for text inputs
    async fn generate_embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse>;

    /// Get the dimension of embeddings produced by this model
    fn embedding_dimension(&self) -> u32;
}

/// Trait for image generation providers
#[async_trait]
pub trait ImageGeneration: Send + Sync {
    /// Get the provider's name/identifier
    fn name(&self) -> &str;

    /// Get the model being used
    fn model(&self) -> &str;

    /// Generate images from text prompts
    async fn generate_image(&self, request: ImageRequest) -> Result<ImageResponse>;

    /// Check if the provider supports image editing
    fn supports_image_editing(&self) -> bool {
        false
    }
}
