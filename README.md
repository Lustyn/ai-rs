# AI SDK for Rust

A comprehensive, type-safe AI SDK for Rust that provides unified abstractions for AI providers, streaming protocols, tool calling, and agent workflows. Build intelligent agents with automatic tool calling, state management, and Human-in-the-Loop (HITL) capabilities.

## 🎯 Vision

This SDK aims to be the definitive Rust library for AI integration, offering:

- **Provider Abstraction**: Seamless switching between AI providers (Anthropic, OpenAI, local models)
- **Type Safety**: Compile-time guarantees for tool calling and message handling
- **Agent Framework**: High-level agent abstractions with streaming and tool integration
- **Tool System**: Type-safe tool calling with automatic JSON schema generation
- **Streaming Support**: Real-time bidirectional communication with AI services
- **State Management**: Stateful tool handlers with shared application state
- **HITL Support**: Human-in-the-Loop scenarios with client-side tool handling

## 🚀 Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ai-rs = "0.1.0"
dotenv = "0.15.0"  # For environment variables
tokio = { version = "1.0", features = ["full"] }
schemars = "0.8"  # For tool schema generation
serde = { version = "1.0", features = ["derive"] }
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

    // Create a generate configuration
    let config = GenerateConfig::new(provider)
        .messages(vec![
            Message::system("You are a helpful AI assistant."),
            Message::user("What is the capital of France?")
        ])
        .max_tokens(1000)
        .temperature(0.7);

    // Generate response
    let response = generate_text(config).await?;
    println!("Response: {:?}", response.final_message);

    Ok(())
}
```

### Streaming Example

```rust
use tokio_stream::StreamExt;

// Create streaming configuration
let config = StreamConfig::new(provider)
    .messages(vec![
        Message::system("You are a helpful assistant."),
        Message::user("Tell me about Rust programming")
    ])
    .max_tokens(1000);

// Start streaming
let mut stream = stream_text(config).await?;

print!("Assistant: ");
while let Some(chunk) = stream.next().await {
    let chunk = chunk?;
    if let MessageDelta::Assistant { content: Some(content) } = chunk.chunk.delta {
        if let AssistantContent::Text { text } = content {
            print!("{}", text);
            std::io::Write::flush(&mut std::io::stdout())?;
        }
    }
}
println!();
```

## 🏗️ Architecture

### Agent Framework

The SDK provides high-level agent abstractions for building conversational AI:

```rust
// Create tools with state
let app_state = AppState::new();
let tool_router = ToolRouter::new()
    .register("calculator", Some("Calculate mathematical expressions".to_string()), calculator_tool)
    .register("weather", Some("Get weather information".to_string()), weather_tool)
    .register("save_note", Some("Save a note with timestamp".to_string()), save_note_tool)
    .with_state(app_state);

// Create configuration with tools
let config = GenerateConfig::new(provider)
    .messages(vec![
        Message::system("You are a helpful assistant with access to tools."),
        Message::user("Calculate 15 * 23 and save the result as a note")
    ])
    .tools(tool_router)
    .max_tokens(2000)
    .run_until(MaxSteps::new(5));

// Generate with tool execution
let response = generate_text(config).await?;
```

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

### Tool System

Type-safe tool calling with automatic schema generation:

```rust
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use ai_rs::tools::{State, ToolRouter};

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct CalculatorParams {
    expression: String,
}

#[derive(Clone, Debug)]
struct AppState {
    calculations: Vec<String>,
}

// Tool handler with shared state
fn calculator_tool(State(mut state): State<AppState>, params: CalculatorParams) -> String {
    let result = evaluate_expression(&params.expression).unwrap_or(0.0);
    let result_str = format!("The result is: {}", result);
    state.calculations.push(result_str.clone());
    result_str
}

// Register tools with automatic schema generation
let tool_router = ToolRouter::new()
    .register("calculator", Some("Calculate mathematical expressions".to_string()), calculator_tool)
    .with_state(AppState { calculations: Vec::new() });
```

### Type-Safe Message Building

Messages support flexible content types with compile-time safety:

```rust
// String literals
let msg = Message::user("Hello world");

// Rich content
let msg = Message::User {
    content: vec![
        UserContent::Text { text: "Describe this image:".to_string() },
        UserContent::Image { image: ImageContent {
            url: Some("https://example.com/image.jpg".to_string()),
            base64: None,
            mime_type: Some("image/jpeg".to_string()),
        }}
    ],
    metadata: None,
};

// Builder pattern with configuration
let config = StreamConfig::new(provider)
    .messages(vec![Message::system("You are helpful")])
    .max_tokens(1000)
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
- 🚧 Vision
- ✅ System messages
- ✅ Agent framework integration

### OpenAI (Planned)

- 🚧 GPT models
- 🚧 Tool calling
- 🚧 Vision
- 🚧 Embeddings

### Local Models (Planned)

- 🚧 Ollama integration
- 🚧 Custom model support

## 🛠️ Features

### Current (Phase 1-2)

- [x] Provider abstraction traits
- [x] Anthropic Claude integration
- [x] Type-safe message building
- [x] Streaming support
- [x] Environment-based configuration
- [x] Comprehensive error handling
- [x] **Tool calling framework with automatic schema generation**
- [x] **Agent framework with state management**
- [x] **Human-in-the-Loop (HITL) tool support**
- [x] **Stateful tool handlers with shared application state**

### Planned (Phase 3-4)

- [ ] OpenAI provider implementation
- [ ] Local model support (Ollama)
- [ ] TypeScript client generation
- [ ] WebSocket streaming for web clients
- [ ] Advanced retry and rate limiting
- [ ] Embeddings and vector operations
- [ ] Multi-modal support (audio, video)
- [ ] Tool composition and chaining
- [ ] Agent memory and context persistence

## 📚 Examples

The SDK includes comprehensive examples demonstrating various usage patterns:

### Running Examples

```bash
# Set your API key
export ANTHROPIC_API_KEY="your-api-key-here"

# Run all examples
cargo run --example agents

# Run specific examples
cargo run --example agents 1 # Basic agent usage
cargo run --example agents 2 # HITL tool calling
cargo run --example agents 3 # Automatic tool execution
```

### Example Scenarios

1. **Basic Agents** (`basic_agents.rs`): Simple conversation with and without streaming
2. **Client-Side Tools** (`client_tools.rs`): Human-in-the-Loop scenarios where tools are defined but handled client-side
3. **Tool Calling** (`tool_calling.rs`): Automatic tool execution with calculator, weather, and note-saving capabilities

### Tool Examples

The examples include several tool implementations:

- **Calculator**: Mathematical expression evaluation
- **Weather**: Mock weather information retrieval
- **Save Note**: Persistent note storage with timestamps

## 🤝 Contributing

We welcome contributions! This project is in active development.

### Development Setup

```bash
git clone https://github.com/Lustyn/ai-rs.git
cd ai-rs
cargo build
cargo test

# Run examples (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY="your-key"
cargo run --example agents
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

## 🚀 Getting Started

To get started quickly:

1. **Set up your environment**:

   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

2. **Run the examples**:

   ```bash
   cargo run --example agents
   ```

3. **Try different scenarios**:

   - Basic conversation: `cargo run --example agents 1`
   - HITL tools: `cargo run --example agents 2`
   - Tool calling: `cargo run --example agents 3`

4. **Build your own agent**:

   ```rust
   let config = GenerateConfig::new(provider)
       .messages(vec![Message::user("Hello, world!")])
       .tools(your_tool_router)
       .max_tokens(2000);

   let response = generate_text(config).await?;
   println!("Response: {:?}", response.final_message);
   ```

---

**Note**: This SDK is in active development. The tool system and agent framework are stable, but APIs may evolve before 1.0 release. We follow semantic versioning for stability guarantees.
