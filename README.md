# AI SDK for Rust

A modular, type-safe Rust SDK for building AI-powered applications with support for multiple providers, tool calling, and agent workflows.

## ✨ Features

- **🔒 Type Safety**: Compile-time guarantees for messages, tools, and responses
- **🧩 Modular Design**: Use only the components you need
- **⚡ Async/Await**: Built on tokio with full async support
- **🛠️ Tool Calling**: Type-safe function calling with automatic JSON schema generation
- **🤖 Agent Framework**: High-level orchestration with configurable execution strategies
- **📡 Streaming**: Real-time response streaming
- **👥 Human-in-the-Loop**: Support for client-side tool execution
- **🔄 Multi-Provider**: Unified interface across different AI providers

## 📦 Architecture

The SDK is organized into focused crates:

### `ai-core`
Core types and abstractions used by all other components:
- Message types and conversation handling
- Provider traits for different AI capabilities
- Type-safe tool system with schema generation
- Comprehensive error handling

### `ai-anthropic` 
Anthropic Claude API implementation:
- Claude Sonnet, Haiku, and Opus support
- Streaming and non-streaming generation
- Tool calling and vision capabilities
- Rate limiting and error handling

### `ai-agent`
High-level agent execution framework:
- Configurable termination strategies
- Automatic tool calling orchestration
- Multi-step conversation management
- Streaming agent execution

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

### Basic Provider Usage

```rust
use ai_core::*;
use ai_anthropic::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Create provider configuration
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("API key required");
    let config = AnthropicConfig::new(api_key, "claude-3-5-sonnet");
    let provider = AnthropicProvider::new(config)?;

    // Simple chat request
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

## 🛠️ Tool Calling

Create type-safe tools with automatic schema generation:

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
    let config = AnthropicConfig::new("api-key", "claude-3-5-sonnet");
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

## 🤖 Agent Execution Strategies

Configure how agents should terminate:

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

## 📦 Project Structure

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

## 🔧 Supported Providers

### Anthropic Claude

- ✅ Text generation (Claude Sonnet, Haiku, Opus)
- ✅ Streaming responses
- ✅ Tool calling
- ✅ Vision capabilities
- ✅ System messages
- ✅ Agent framework integration

### Coming Soon

- **OpenAI**: GPT-4, GPT-3.5, tool calling, vision, embeddings
- **Local Models**: Ollama integration, custom model support
- **Google**: Gemini support

## 📚 Examples

Run the included examples to see the SDK in action:

```bash
# Set your API key
export ANTHROPIC_API_KEY="your-api-key-here"

# Run all examples
cargo run --bin agents

# Run specific examples
cargo run --bin provider_usage  # Basic provider usage
cargo run --bin mixed_tools     # Tool system demo
cargo run --bin agents         # Agent framework demo
```

### Example Scenarios

- **provider_usage.rs**: Basic provider usage without agents
- **mixed_tools.rs**: Tool system with fallible and infallible tools
- **agents/**: Advanced agent examples with tool calling and HITL scenarios

## 🧪 Development

```bash
# Check all crates
cargo check --workspace

# Run tests
cargo test --workspace

# Build release
cargo build --workspace --release
```

## 🚧 Roadmap

- **Enhanced Tool System**: Tool composition, chaining, and async tools
- **More Providers**: OpenAI, Google, local models via Ollama
- **Advanced Features**: Embeddings, image generation, voice synthesis
- **Performance**: Caching, rate limiting, connection pooling
- **Integration**: Web frameworks, CLI tools, desktop apps

## 🤝 Contributing

Contributions welcome! Please see our contributing guidelines for details on:

- Code style and testing requirements
- Adding new providers
- Extending the tool system
- Documentation improvements

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.