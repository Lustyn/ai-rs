# AI SDK for Rust

A comprehensive, type-safe AI SDK for Rust that provides unified abstractions for AI providers, streaming protocols, and cross-language client support.

## 🎯 Vision

This SDK aims to be the definitive Rust library for AI integration, offering:

- **Provider Abstraction**: Seamless switching between AI providers (OpenAI, Anthropic, local models)
- **Type Safety**: Compile-time guarantees for tool calling and message handling
- **Streaming Support**: Real-time bidirectional communication with AI services
- **Cross-Language**: Generate TypeScript clients with full type safety from Rust definitions

## 🚀 Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ai-rs = "0.1.0"
dotenv = "0.15.0"  # For environment variables
tokio = { version = "1.0", features = ["full"] }
```

### Basic Usage

```rust
use ai_rs::*;
use dotenv::dotenv;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    
    // Create provider configuration
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;
    let config = AnthropicConfig::new(api_key, "claude-3-5-sonnet-20241022");
    let provider = AnthropicProvider::new(config)?;
    
    // Build a chat request
    let request = ChatRequest::new()
        .system("You are a helpful AI assistant.")
        .user("What is the capital of France?")
        .temperature(0.7)
        .max_tokens(100);
    
    // Generate response
    let response = provider.generate(request).await?;
    println!("Response: {}", response.message.content[0].text);
    
    Ok(())
}
```

### Streaming Example

```rust
use tokio_stream::StreamExt;

// Create streaming request
let mut stream = provider.generate_stream(request).await?;

print!("Assistant: ");
while let Some(chunk) = stream.next().await {
    match chunk? {
        ChatStreamChunk { delta, .. } => {
            if let MessageDelta::Assistant { content: Some(content) } = delta {
                if let AssistantContent::Text { text } = content {
                    print!("{}", text);
                    std::io::Write::flush(&mut std::io::stdout())?;
                }
            }
        }
    }
}
println!();
```

## 🏗️ Architecture

### Provider Abstraction Layer

The SDK uses trait-based abstractions to support multiple AI providers:

```rust
#[async_trait]
pub trait ChatTextGeneration: Send + Sync {
    async fn generate(&self, request: ChatRequest) -> Result<ChatResponse>;
    async fn generate_stream(&self, request: ChatRequest) -> Result<StreamType>;
    
    fn supports_tools(&self) -> bool;
    fn supports_vision(&self) -> bool;
    fn max_tokens(&self) -> Option<u32>;
}
```

### Type-Safe Message Building

Messages support flexible content types with compile-time safety:

```rust
// String literals
let msg = Message::user("Hello world");

// Rich content
let msg = Message::user(vec![
    UserContent::Text { text: "Describe this image:".to_string() },
    UserContent::Image { image: ImageContent { ... } }
]);

// Builder pattern
let request = ChatRequest::new()
    .system("You are helpful")
    .user("What is Rust?")
    .assistant("Rust is a systems programming language")
    .temperature(0.8);
```

### Streaming Protocol

Real-time streaming with type-safe message deltas:

```rust
pub enum MessageDelta {
    Assistant { content: Option<AssistantContent> },
    Tool { tool_result: Option<ToolResult> },
    // ...
}

pub struct ChatStreamChunk {
    pub id: String,
    pub delta: MessageDelta,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
}
```

## 🔧 Supported Providers

### Anthropic Claude
- ✅ Text generation
- ✅ Streaming
- ✅ Tool calling
- ✅ Vision (images)
- ✅ System messages

### OpenAI (Planned)
- 🚧 GPT models
- 🚧 Tool calling
- 🚧 Vision
- 🚧 Embeddings

### Local Models (Planned)
- 🚧 Ollama integration
- 🚧 Custom model support

## 🛠️ Features

### Current (Phase 1)
- [x] Provider abstraction traits
- [x] Anthropic Claude integration
- [x] Type-safe message building
- [x] Streaming support
- [x] Environment-based configuration
- [x] Comprehensive error handling

### Planned (Phase 2-4)
- [ ] Tool calling framework with compile-time validation
- [ ] OpenAI provider implementation
- [ ] Local model support (Ollama)
- [ ] TypeScript client generation
- [ ] WebSocket streaming for web clients
- [ ] Advanced retry and rate limiting
- [ ] Embeddings and vector operations
- [ ] Multi-modal support (audio, video)

## 🤝 Contributing

We welcome contributions! This project is in active development.

### Development Setup

```bash
git clone https://github.com/Lustyn/ai-rs.git
cd ai-rs
cargo build
cargo test
```

### Adding New Providers

1. Implement the `ChatTextGeneration` trait
2. Add provider-specific configuration
3. Handle provider-specific message formats
4. Add comprehensive tests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- Documentation (coming soon)
- Crates.io (coming soon)

---

**Note**: This SDK is in early development. APIs may change before 1.0 release. We follow semantic versioning for stability guarantees.
