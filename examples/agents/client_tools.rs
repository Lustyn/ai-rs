use ai_agent::*;
use ai_anthropic::*;
use ai_core::*;
use dotenv::dotenv;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;

/// Example of client-side tools (Human-in-the-Loop scenarios)
/// These tools have definitions but no handlers - the client must handle them
pub async fn run_client_tools_example() -> Result<()> {
    dotenv().ok();
    println!("=== Client-Side Tools Example (HITL) ===\n");

    // Create provider
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY environment variable must be set");
    let config = AnthropicConfig::new(api_key, "claude-3-5-sonnet-20241022");
    let provider = AnthropicProvider::new(config)?;

    // Define client-side tool schemas
    let approval_schema = schemars::schema_for!(ApprovalRequest);
    let user_input_schema = schemars::schema_for!(UserInputRequest);

    // Create tool router with client-side tool definitions (no handlers)
    let router = ToolRouter::new()
        .register_definition(
            "request_approval",
            Some("Request approval from user for a sensitive action".to_string()),
            Some(approval_schema),
        )
        .register_definition(
            "request_user_input",
            Some("Request additional input from the user".to_string()),
            Some(user_input_schema),
        )
        .with_state(());

    let messages = vec![
        Message::system(
            "You are a helpful assistant. When you need to perform sensitive actions like deleting files or making purchases, use the request_approval tool. When you need more information from the user, use the request_user_input tool.",
        ),
        Message::user(
            "I want to delete all my old photos to free up space. Can you help me with that?",
        ),
    ];

    println!("Starting conversation with client-side tools...\n");

    // Use streaming to handle tool calls as they come
    let config = StreamConfig::new(provider)
        .messages(messages)
        .tools(router)
        .run_until(MaxSteps::new(5));

    match stream_text(config).await {
        Ok(mut stream) => {
            let mut pending_tool_calls = Vec::new();

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(agent_chunk) => {
                        // Collect tool calls
                        if let MessageDelta::Assistant {
                            content: Some(AssistantContent::ToolCall { tool_call }),
                        } = &agent_chunk.chunk.delta
                        {
                            pending_tool_calls.push(tool_call.clone());
                        }

                        // Print text content
                        if let MessageDelta::Assistant {
                            content: Some(AssistantContent::Text { text }),
                        } = &agent_chunk.chunk.delta
                            && !text.is_empty()
                        {
                            print!("{}", text);
                            std::io::Write::flush(&mut std::io::stdout()).unwrap();
                        }

                        // Handle final chunk - process any tool calls
                        if agent_chunk.is_final {
                            println!(); // New line after content

                            // Process tool calls that need client handling
                            for tool_call in &pending_tool_calls {
                                println!("\n🔧 Tool Call: {}", tool_call.name);
                                println!(
                                    "Arguments: {}",
                                    serde_json::to_string_pretty(&tool_call.arguments).unwrap()
                                );

                                match tool_call.name.as_str() {
                                    "request_approval" => {
                                        let approval: ApprovalRequest =
                                            serde_json::from_value(tool_call.arguments.clone())?;
                                        let user_response =
                                            handle_approval_request(&approval).await;
                                        println!(
                                            "✅ User response: {}",
                                            if user_response { "APPROVED" } else { "DENIED" }
                                        );
                                    }
                                    "request_user_input" => {
                                        let input_req: UserInputRequest =
                                            serde_json::from_value(tool_call.arguments.clone())?;
                                        let user_input =
                                            handle_user_input_request(&input_req).await;
                                        println!("✅ User input: {}", user_input);
                                    }
                                    _ => {
                                        println!("❌ Unknown tool: {}", tool_call.name);
                                    }
                                }
                            }

                            // In a real application, you would:
                            // 1. Continue the conversation with tool results
                            // 2. Add tool result messages to continue the agent loop
                            // For this example, we'll just show the tool calls were handled
                            if !pending_tool_calls.is_empty() {
                                println!(
                                    "\n💡 In a real app, you would continue the conversation with these tool results"
                                );
                                break;
                            }

                            pending_tool_calls.clear();
                        }
                    }
                    Err(e) => {
                        println!("\nStream error: {}", e);
                        break;
                    }
                }
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    Ok(())
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ApprovalRequest {
    /// The action that requires approval
    action: String,
    /// Detailed description of what will happen
    description: String,
    /// Risk level (low, medium, high)
    risk_level: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct UserInputRequest {
    /// What information is needed
    prompt: String,
    /// Type of input expected (text, number, choice, etc.)
    input_type: String,
    /// Optional validation rules or choices
    validation: Option<String>,
}

/// Simulate handling an approval request
async fn handle_approval_request(request: &ApprovalRequest) -> bool {
    println!("\n🚨 APPROVAL REQUIRED 🚨");
    println!("Action: {}", request.action);
    println!("Description: {}", request.description);
    println!("Risk Level: {}", request.risk_level);

    // In a real application, this would:
    // - Show a UI dialog
    // - Send a notification
    // - Wait for user input
    // - Return the actual user decision

    // For demo purposes, auto-deny high-risk actions
    match request.risk_level.as_str() {
        "high" => {
            println!("⚠️  High-risk action auto-denied for safety");
            false
        }
        _ => {
            println!("✅ Medium/low-risk action auto-approved for demo");
            true
        }
    }
}

/// Simulate handling a user input request
async fn handle_user_input_request(request: &UserInputRequest) -> String {
    println!("\n💬 USER INPUT REQUESTED 💬");
    println!("Prompt: {}", request.prompt);
    println!("Expected type: {}", request.input_type);
    if let Some(validation) = &request.validation {
        println!("Validation: {}", validation);
    }

    // In a real application, this would:
    // - Show an input dialog
    // - Wait for user to type/select
    // - Validate the input
    // - Return the actual user input

    // For demo purposes, return a simulated response
    match request.input_type.as_str() {
        "choice" => "option_a".to_string(),
        "number" => "42".to_string(),
        _ => "This is simulated user input for the demo".to_string(),
    }
}
