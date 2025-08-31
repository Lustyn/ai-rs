use ai_agent::*;
use ai_anthropic::*;
use ai_core::{
    errors::{ToolExecutionError, ToolResult},
    *,
};
use dotenv::dotenv;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Example of tools with actual handlers that execute automatically
pub async fn run_tool_calling_example() -> Result<()> {
    dotenv().ok();
    println!("=== Tool Calling Example ===\n");

    // Create provider
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY environment variable must be set");
    let config = AnthropicConfig::new(api_key, "claude-3-5-sonnet-20241022");
    let provider = AnthropicProvider::new(config)?;

    // Create application state
    let state = AppState {
        calculator_history: Vec::new(),
        weather_cache: HashMap::new(),
    };

    // Create tool router with actual handlers
    // Mix of fallible tools (that can return errors) and infallible tools
    let router = ToolRouter::new()
        .register(
            "calculator",
            Some("Perform basic mathematical calculations".to_string()),
            calculator_tool,
        )
        .register(
            "get_weather",
            Some("Get current weather for a location".to_string()),
            weather_tool,
        )
        .register(
            "save_note",
            Some("Save a note to the application state".to_string()),
            save_note_tool,
        )
        // Example of an infallible tool (always succeeds)
        .register_infallible(
            "get_time",
            Some("Get current UTC time".to_string()),
            |_input: serde_json::Value| async {
                serde_json::json!({
                    "time": chrono::Utc::now().to_rfc3339(),
                })
            },
        )
        .with_state(state);

    let messages = vec![
        Message::system(
            "You are a helpful assistant with access to tools. You can perform calculations, check weather, and save notes. Use tools when appropriate to help the user.",
        ),
        Message::user(
            "What's 15 * 23? Also, what's the weather like in San Francisco? Finally, save a note saying 'Meeting with John tomorrow at 3pm'.",
        ),
    ];

    println!("Starting conversation with tool handlers...\n");

    // Use generate_text to let the agent use tools automatically
    let config = GenerateConfig::new(provider)
        .messages(messages)
        .tools(router)
        .run_until(MaxSteps::new(10)); // Allow multiple steps for tool usage

    match generate_text(config).await {
        Ok(response) => {
            println!("=== Final Conversation ===");
            for (i, msg) in response.messages.iter().enumerate() {
                match msg {
                    Message::User { content, .. } => {
                        let text = content
                            .iter()
                            .filter_map(|c| match c {
                                UserContent::Text { text } => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join(" ");
                        println!("{}. 👤 User: {}", i + 1, text);
                    }
                    Message::Assistant { content, .. } => {
                        for c in content {
                            match c {
                                AssistantContent::Text { text } => {
                                    println!("{}. 🤖 Assistant: {}", i + 1, text);
                                }
                                AssistantContent::ToolCall { tool_call } => {
                                    println!(
                                        "{}. 🔧 Tool Call: {} with args: {}",
                                        i + 1,
                                        tool_call.name,
                                        serde_json::to_string(&tool_call.arguments)
                                            .unwrap_or_default()
                                    );
                                }
                            }
                        }
                    }
                    Message::Tool { tool_results, .. } => {
                        for result in tool_results {
                            println!(
                                "{}. ⚙️  Tool Result: {}",
                                i + 1,
                                serde_json::to_string_pretty(&result.result).unwrap_or_default()
                            );
                        }
                    }
                    _ => {}
                }
            }

            println!("\n=== Summary ===");
            println!("Completed {} steps", response.steps);
            println!("Finish reason: {:?}", response.finish_reason);
            if let Some(usage) = response.total_usage {
                println!("Total tokens: {}", usage.total_tokens);
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    Ok(())
}

/// Application state that tools can access and modify
#[derive(Clone, Debug)]
struct AppState {
    calculator_history: Vec<String>,
    weather_cache: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct CalculatorInput {
    /// Mathematical expression to evaluate (e.g., "15 * 23", "sqrt(144)")
    expression: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct WeatherInput {
    /// City name or location
    location: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SaveNoteInput {
    /// The note content to save
    note: String,
    /// Optional category for the note
    category: Option<String>,
}

/// Calculator tool handler with improved error handling
async fn calculator_tool(
    State(mut state): State<AppState>,
    input: CalculatorInput,
) -> ToolResult<serde_json::Value> {
    println!("🧮 Calculator called with: {}", input.expression);

    // Validate input
    if input.expression.trim().is_empty() {
        return Err(ToolExecutionError::InvalidInput(
            "Expression cannot be empty".to_string(),
        ));
    }

    let result = evaluate_expression(&input.expression)?;

    // Update calculator history in state
    state
        .calculator_history
        .push(format!("{} = {}", input.expression, result));

    Ok(serde_json::json!({
        "expression": input.expression,
        "result": result,
        "formatted": format!("{} = {}", input.expression, result),
        "history_count": state.calculator_history.len()
    }))
}

/// Simple expression evaluator with improved error handling
fn evaluate_expression(expr: &str) -> ToolResult<f64> {
    let expr = expr.trim();

    // Find the operator and split
    let (left, op, right) = if let Some(pos) = expr.rfind('*') {
        (&expr[..pos], '*', &expr[pos + 1..])
    } else if let Some(pos) = expr.rfind('/') {
        (&expr[..pos], '/', &expr[pos + 1..])
    } else if let Some(pos) = expr.rfind('+') {
        (&expr[..pos], '+', &expr[pos + 1..])
    } else if let Some(pos) = expr.rfind('-') {
        // Handle negative numbers by finding the last minus that's not at the start
        if pos > 0 {
            (&expr[..pos], '-', &expr[pos + 1..])
        } else {
            return Err(ToolExecutionError::InvalidInput(
                "Invalid expression format".to_string(),
            ));
        }
    } else {
        return Err(ToolExecutionError::InvalidInput(
            "No operator found. Use +, -, *, or /".to_string(),
        ));
    };

    let a: f64 = left.trim().parse().map_err(|_| {
        ToolExecutionError::InvalidInput(format!("Invalid number: '{}'", left.trim()))
    })?;
    let b: f64 = right.trim().parse().map_err(|_| {
        ToolExecutionError::InvalidInput(format!("Invalid number: '{}'", right.trim()))
    })?;

    let result = match op {
        '+' => a + b,
        '-' => a - b,
        '*' => a * b,
        '/' => {
            if b == 0.0 {
                return Err(ToolExecutionError::ExecutionError(
                    "Division by zero".to_string(),
                ));
            }
            a / b
        }
        _ => unreachable!(),
    };

    Ok(result)
}

/// Weather tool handler (simulated) with improved error handling
async fn weather_tool(
    State(mut state): State<AppState>,
    input: WeatherInput,
) -> ToolResult<serde_json::Value> {
    println!("🌤️  Weather called for: {}", input.location);

    // Simulate weather data (in real app, you'd call a weather API)
    let weather_data = match input.location.to_lowercase().as_str() {
        loc if loc.contains("san francisco") || loc.contains("sf") => {
            serde_json::json!({
                "location": "San Francisco, CA",
                "temperature": "68°F (20°C)",
                "condition": "Partly cloudy",
                "humidity": "65%",
                "wind": "12 mph W"
            })
        }
        loc if loc.contains("new york") || loc.contains("nyc") => {
            serde_json::json!({
                "location": "New York, NY",
                "temperature": "72°F (22°C)",
                "condition": "Sunny",
                "humidity": "55%",
                "wind": "8 mph SW"
            })
        }
        loc if loc.contains("london") => {
            serde_json::json!({
                "location": "London, UK",
                "temperature": "59°F (15°C)",
                "condition": "Light rain",
                "humidity": "80%",
                "wind": "15 mph NW"
            })
        }
        _ => {
            serde_json::json!({
                "location": input.location,
                "temperature": "Unknown",
                "condition": "Weather data not available for this location",
                "error": "Location not found in demo data"
            })
        }
    };

    // Cache the weather data
    state.weather_cache.insert(
        input.location.clone(),
        serde_json::to_string(&weather_data).unwrap_or_default(),
    );

    Ok(weather_data)
}

/// Save note tool handler with improved error handling
async fn save_note_tool(input: SaveNoteInput) -> ToolResult<serde_json::Value> {
    println!("📝 Saving note: {}", input.note);

    // Validate input
    if input.note.trim().is_empty() {
        return Err(ToolExecutionError::InvalidInput(
            "Note content cannot be empty".to_string(),
        ));
    }

    // In a real app, you'd save to database or file system
    // For demo, just return success
    let timestamp = chrono::Utc::now()
        .format("%Y-%m-%d %H:%M:%S UTC")
        .to_string();

    Ok(serde_json::json!({
        "success": true,
        "note": input.note,
        "category": input.category.unwrap_or_else(|| "general".to_string()),
        "saved_at": timestamp,
        "message": "Note saved successfully"
    }))
}
