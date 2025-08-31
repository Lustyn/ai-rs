use ai_core::{
    errors::{ToolExecutionError, ToolResult},
    *,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Example demonstrating both infallible (direct return) and fallible (ToolResult) tools
#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Mixed Tool Types Example ===\n");

    // Create a simple state
    let state = AppState { counter: 0 };

    // Create tool router mixing both types of tools
    let router = ToolRouter::new()
        // Infallible tool - always succeeds, returns direct value
        .register_infallible(
            "get_time",
            Some("Get current time".to_string()),
            get_current_time,
        )
        // Another infallible tool
        .register_infallible("echo", Some("Echo back the input".to_string()), echo_tool)
        // Fallible tool - can return errors
        .register(
            "divide",
            Some("Divide two numbers".to_string()),
            divide_numbers,
        )
        // Another fallible tool
        .register(
            "validate_email",
            Some("Validate an email address".to_string()),
            validate_email,
        )
        .with_state(state);

    println!("Registered tools:");
    for name in router.tool_names() {
        println!("  - {}", name);
    }

    // Test the tools
    println!("\n=== Testing Tools ===\n");

    // Test infallible tool
    println!("Testing get_time (infallible):");
    match router.execute_tool("get_time", serde_json::json!({})).await {
        Some(Ok(result)) => println!("  Result: {}", result),
        Some(Err(e)) => println!("  Error: {}", e),
        None => println!("  No handler"),
    }

    // Test fallible tool with valid input
    println!("\nTesting divide with 10/2 (fallible):");
    match router
        .execute_tool(
            "divide",
            serde_json::json!({
                "numerator": 10.0,
                "denominator": 2.0
            }),
        )
        .await
    {
        Some(Ok(result)) => println!("  Result: {}", result),
        Some(Err(e)) => println!("  Error: {}", e),
        None => println!("  No handler"),
    }

    // Test fallible tool with invalid input (division by zero)
    println!("\nTesting divide with 10/0 (fallible - should error):");
    match router
        .execute_tool(
            "divide",
            serde_json::json!({
                "numerator": 10.0,
                "denominator": 0.0
            }),
        )
        .await
    {
        Some(Ok(result)) => println!("  Result: {}", result),
        Some(Err(e)) => println!("  Error: {}", e),
        None => println!("  No handler"),
    }

    Ok(())
}

#[derive(Clone)]
struct AppState {
    #[allow(dead_code)]
    counter: i32,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct TimeInput {}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct EchoInput {
    message: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct DivideInput {
    numerator: f64,
    denominator: f64,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct EmailInput {
    email: String,
}

// Infallible tool - always returns a value
async fn get_current_time(_input: TimeInput) -> serde_json::Value {
    let now = chrono::Utc::now();
    serde_json::json!({
        "time": now.to_rfc3339(),
        "timestamp": now.timestamp(),
    })
}

// Another infallible tool
async fn echo_tool(input: EchoInput) -> serde_json::Value {
    serde_json::json!({
        "echoed": input.message,
        "length": input.message.len(),
    })
}

// Fallible tool - can return errors
async fn divide_numbers(input: DivideInput) -> ToolResult<serde_json::Value> {
    if input.denominator == 0.0 {
        return Err(ToolExecutionError::ExecutionError(
            "Cannot divide by zero".to_string(),
        ));
    }

    let result = input.numerator / input.denominator;
    Ok(serde_json::json!({
        "result": result,
        "operation": format!("{} / {} = {}", input.numerator, input.denominator, result),
    }))
}

// Another fallible tool
async fn validate_email(input: EmailInput) -> ToolResult<serde_json::Value> {
    if !input.email.contains('@') {
        return Err(ToolExecutionError::InvalidInput(
            "Email must contain @ symbol".to_string(),
        ));
    }

    if !input.email.contains('.') {
        return Err(ToolExecutionError::InvalidInput(
            "Email must contain a domain".to_string(),
        ));
    }

    Ok(serde_json::json!({
        "valid": true,
        "email": input.email,
        "timestamp": chrono::Utc::now().to_rfc3339(),
    }))
}
