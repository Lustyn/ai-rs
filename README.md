# AI SDK for Rust

A modular, type-safe Rust SDK for AI providers with focus on async/await support and extensible tool calling.

## 🏗️ Architecture

This SDK is now organized into three main crates for better modularity and maintainability:

### Core (`ai-core`)
Foundational types, traits, and abstractions:
- Core types: `Message`, `ChatRequest`, `ChatResponse`, etc.
- Provider traits: `ChatTextGeneration`, `EmbeddingGeneration`, `ImageGeneration`
- Tool system: Type-safe tool definitions and execution
- Error handling: Comprehensive error types

### Providers (`ai-anthropic`, `ai-openai`, etc.)
Provider-specific implementations:
- **`ai-anthropic`**: Anthropic Claude API implementation
- More providers coming soon (OpenAI, Google, etc.)

### Agent Framework (`ai-agent`)
High-level agent execution:
- Configurable termination strategies (`MaxSteps`, `StopOnReason`)
- Tool calling orchestration
- Streaming and non-streaming execution

## 🎯 Benefits of This Architecture

- **🧩 Modular**: Use only the crates you need - no bloated dependencies
- **🔒 Type Safety**: Compile-time guarantees for tool calling and message handling
- **⚡ Performance**: Each crate optimized for its specific purpose
- **🔄 Extensible**: Easy to add new providers without touching core logic
- **📦 Composable**: Mix and match providers and agents as needed

## 🚀 Quick Start

### Installation

Add the crates you need to your `Cargo.toml`:

```toml
[dependencies]
# Core types and traits (always needed)
ai-core = { path = "crates/core" }
# Choose your providers
ai-anthropic = { path = "crates/anthropic" }
# Agent framework (optional, for high-level orchestration)  
ai-agent = { path = "crates/agent" }

# Supporting libraries
tokio = { version = "1.0", features = ["full"] }
schemars = "0.8"  # For tool schema generation
serde = { version = "1.0", features = ["derive"] }
```

### Basic Usage

```rust
use ai_core::*;
use ai_anthropic::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Create provider configuration
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("API key required");
    let config = AnthropicConfig::new(api_key, "claude-3-5-sonnet-20241022");
    let provider = AnthropicProvider::new(config)?;

    // Simple chat without agents
    let request = ChatRequest::new()
        .system("You are a helpful AI assistant.")
        .user("What is the capital of France?")
        .temperature(0.7);

    let response = provider.generate(request).await?;
    println!("Response: {:?}", response);

    Ok(())
}
```

### Agent Framework Usage

```rust
use ai_core::*;
use ai_anthropic::*;
use ai_agent::*;

// Using the high-level agent framework
let config = GenerateConfig::new(provider)
    .messages(vec![
        Message::system("You are a helpful AI assistant."),
        Message::user("What is the capital of France?")
    ])
    .max_tokens(1000)
    .temperature(0.7)
    .run_until(MaxSteps::new(3));

// Generate with agent orchestration
let response = generate_text(config).await?;
println!("Agent completed in {} steps", response.steps);
```

## 🛠️ Tool Calling Example

```rust
use ai_core::{*, errors::{ToolExecutionError, ToolResult}};
use ai_anthropic::*;
use ai_agent::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone)]
struct AppState {
    calculator_history: Vec<String>,
}

#[derive(Deserialize, JsonSchema)]
struct CalculatorInput {
    expression: String,
}

// Tool handler that can return errors
fn calculator(
    State(mut state): State<AppState>, 
    input: CalculatorInput
) -> ToolResult<serde_json::Value> {
    // Your calculation logic here
    let result = 42.0; // Placeholder
    
    state.calculator_history.push(format!("{} = {}", input.expression, result));
    
    Ok(serde_json::json!({
        "result": result,
        "expression": input.expression
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = AnthropicConfig::new("api-key", "claude-3-5-sonnet-20241022");
    let provider = AnthropicProvider::new(config)?;
    
    // Create tool router
    let router = ToolRouter::new()
        .register("calculator", Some("Perform calculations".to_string()), calculator)
        .with_state(AppState { calculator_history: Vec::new() });
    
    // Use with agent
    let config = GenerateConfig::new(provider)
        .messages(vec![
            Message::system("You are a calculator assistant."),
            Message::user("What's 15 * 23?"),
        ])
        .tools(router)
        .run_until(MaxSteps::new(5));
        
    let response = generate_text(config).await?;
    println!("Agent completed in {} steps", response.steps);
    
    Ok(())
}
```

## 🤖 Agent Framework

The agent framework provides powerful orchestration with configurable termination strategies:

```rust
use ai_agent::*;

// Stop after maximum steps
let config = GenerateConfig::new(provider)
    .run_until(MaxSteps::new(5));

// Stop on specific finish reasons
let config = GenerateConfig::new(provider)
    .run_until(StopOnReason::stop_on_finish());

// Combine strategies
let combined = RunUntilFirst::new(
    MaxSteps::new(10),
    StopOnReason::stop_on_finish()
);
let config = GenerateConfig::new(provider).run_until(combined);
```

## 📦 Crate Structure

The new modular architecture organizes code into focused crates:

```
ai-rs/
├── crates/
│   ├── core/          # Core types, traits, and tools system
│   │   ├── errors.rs  # Comprehensive error handling  
│   │   ├── types.rs   # Message types, requests/responses
│   │   ├── provider.rs # Provider traits
│   │   └── tools.rs   # Type-safe tool system
│   ├── anthropic/     # Anthropic Claude implementation
│   │   └── provider.rs
│   └── agent/         # High-level agent orchestration
│       └── agent.rs
├── examples/          # Comprehensive examples
└── Cargo.toml         # Workspace configuration
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
