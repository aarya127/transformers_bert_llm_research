# model_check.py
import traceback

def check_model_availability(MODELS, openai_client, claude_client, gemini_client):
    """
    Checks the availability of each model in the MODELS dict by making a minimal API call.
    Returns a dict of available models.
    """
    available = {}
    test_prompt = "Hello! This is a test. Please reply with 'OK'."
    for provider, model in MODELS.items():
        print(f"\nChecking provider: {provider}, model: {model}")
        try:
            if provider.startswith("openai"):
                print(f"Using OpenAI API key: {str(openai_client.api_key)[:8]}... (truncated)")
                resp = openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": test_prompt}]
                )
                if resp.choices[0].message.content:
                    available[provider] = model
            elif provider.startswith("claude"):
                print(f"Using Anthropic API key: {str(claude_client.api_key)[:8]}... (truncated)")
                resp = claude_client.messages.create(
                    model=model,
                    max_tokens=10,
                    messages=[{"role": "user", "content": test_prompt}]
                )
                if resp.content and resp.content[0].text:
                    available[provider] = model
            elif provider == "gemini":
                print(f"Using Gemini API key: (hidden for SDK)")
                resp = gemini_client.models.generate_content(
                    model=model,
                    contents=test_prompt
                )
                if hasattr(resp, 'text') and resp.text:
                    available[provider] = model
        except Exception as e:
            print(f"Warning: {provider} model '{model}' unavailable or failed test call: {e}")
            traceback.print_exc()
    return available
