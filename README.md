# AI Data Parser: LangChain vs LlamaIndex Structured Output Demo

This project demonstrates how to use **LangChain** and **LlamaIndex** with **Pydantic** models to generate structured outputs from unstructured call transcript data. It's designed as an educational resource for students learning about AI-powered data processing.

## 🎯 Project Overview

The project processes customer service call transcripts and extracts structured information including:
- Call summary
- Call intent/purpose
- Agent performance rating (1-5)
- Customer sentiment rating (1-5)
- Detailed rationales for ratings

## 🚀 Features

- ✅ **Two Framework Approaches**: Compare LangChain vs LlamaIndex implementations
- ✅ **Structured Output**: Uses Pydantic models for consistent data validation
- ✅ **Concurrent Processing**: ThreadPoolExecutor for faster batch processing
- ✅ **Error Handling**: Separate error logging to prevent batch failures
- ✅ **CSV Processing**: Handles malformed CSV data with proper parsing
- ✅ **Educational Code**: Clean, well-documented examples for learning

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Git (optional, for cloning)

## 🛠️ Setup Instructions

### Using Virtual Environment 

```bash
# Clone or download the project
git clone <repository-url>
cd ai-dataparser

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

```

## 🔑 Environment Setup

1. Create a `.env` file in the project root:
```bash
On Linux/macOS:
    touch .env
On Windows (PowerShell):
    New-Item -Path .env -ItemType File -Force
```

2. Add your OpenAI API key to the `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

3. **Important**: Never commit your `.env` file to version control!

## 📁 Project Structure

```
ai-dataparser/
├── notebook/
│   └── langchain_dataparser.ipynb
│   └── llamaindex_dataparser.ipynb
├── data/
│   └── call_transcript_sample.csv    # Sample call transcripts
├── .env                              # Environment variables (create this)
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

## 🎓 Code Architecture Explained

### 1. Pydantic Model Definition

Both approaches use the same Pydantic model for structured output:

```python
class CallAnalysis(BaseModel):
    """Analysis results for a customer service call transcript."""
    transcript_summary: str = Field(description="Summary of the call transcript in 2-4 sentences")
    call_intent: str = Field(description="Main purpose or intention of the call")
    agents_performance_rating: int = Field(ge=1, le=5, description="Agent performance rating (1-5)")
    agents_performance_rating_rationale: str = Field(description="Rationale for agent performance rating")
    customer_sentiment_rating: int = Field(ge=1, le=5, description="Customer sentiment rating (1-5)")
    customer_sentiment_rationale: str = Field(description="Rationale for customer sentiment rating")
```

**Key Features:**
- **Type Safety**: Ensures all fields have correct data types
- **Validation**: `ge=1, le=5` ensures ratings are between 1-5
- **Documentation**: Field descriptions help the LLM understand requirements
- **Docstring**: Required for LlamaIndex OpenAI Function conversion

### 2. LangChain Approach

```python
# Initialize components
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = PydanticOutputParser(pydantic_object=CallAnalysis)

# Create chain
chain = prompt | llm | parser
result = chain.invoke({"call_transcript": call_transcript})
```

**How it works:**
1. **PydanticOutputParser**:Generate structured output
2. **Chain Composition**: Uses `|` operator to connect prompt → LLM → parser
3. **Format Instructions**: Automatically adds JSON schema to prompt
4. **Validation**: Parser validates LLM output against Pydantic model

### 3. LlamaIndex Approach 

```python
# Create program
program = OpenAIPydanticProgram.from_defaults(
    output_cls=CallAnalysis,
    prompt_template_str=prompt_template_str,
    verbose=True,
    # You can also pass model_name and temperature here if needed
    # model_name="gpt-4o-mini",
    # temperature=0,
)

# Use program
result = program(call_transcript=call_transcript)
```

**How it works:**
1. **OpenAIPydanticProgram**: High-level abstraction for structured output
2. **Function Calling**: Uses OpenAI's function calling feature internally
3. **Automatic Conversion**: Converts Pydantic model to OpenAI function schema
4. **Simple API**: Single function call returns validated Pydantic object

### 4. Concurrency Implementation


Both approaches use `ThreadPoolExecutor` for parallel processing. Note the function names differ:

**LangChain:**
```python
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(process_single_transcript, row, llm, parser, prompt)
        for _, row in df.iterrows()
    ]
    for future in futures:
        success_result, error_result = future.result()
```

**LlamaIndex:**
```python
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(process_single_transcript_llamaindex, row, program)
        for _, row in df.iterrows()
    ]
    for future in futures:
        success_result, error_result = future.result()
```

**Benefits:**
- **Speed**: Process multiple transcripts simultaneously
- **Rate Limiting**: `max_workers=3` prevents API rate limit issues
- **Error Isolation**: Individual failures don't stop the entire batch

### 5. Error Handling Strategy

```python
try:
    result = chain.invoke({"call_transcript": call_transcript})
    return output_row, None
except Exception as e:
    error_row = {
        'conversation_id': conversation_id,
        'error_message': str(e),
        'error_type': type(e).__name__,
        'timestamp': datetime.now().isoformat()
    }
    return None, error_row
```

**Features:**
- **Graceful Degradation**: Errors don't crash the entire process
- **Detailed Logging**: Captures error type, message, and timestamp
- **Separate Output**: Errors saved to dedicated CSV file
- **Debugging**: Easy to identify and fix issues


### Common Issues

**Rate Limit Error**
   ```
   RateLimitError: API rate limit exceeded
   ```
   **Solution**: Reduce `max_workers` parameter or add delays


### Performance Tuning

- **Concurrent Workers**: Start with `max_workers=3`, adjust based on API limits
- **Batch Size**: Process large datasets in chunks if memory is limited
- **Model Selection**: Use `gpt-4o-mini` for cost efficiency, `gpt-4o` for better quality

## 📚 Learning Objectives

This project teaches:

1. **Structured Output Generation**: How to get consistent, validated data from LLMs
2. **Framework Comparison**: Differences between LangChain and LlamaIndex approaches
3. **Pydantic Integration**: Using Pydantic for data validation and type safety
4. **Concurrent Programming**: Implementing parallel processing for better performance
5. **Error Handling**: Building robust data processing pipelines
6. **CSV Processing**: Handling real-world messy data formats

## 🤝 Contributing

This is an educational project. Feel free to:
- Add more examples
- Improve error handling
- Add additional frameworks
- Enhance documentation

## 📄 License

---

### Additional Notes

- **Prompt Differences:** Even with the same prompt text, LangChain and LlamaIndex may format prompts/messages differently (e.g., system/user roles, format instructions), which can affect model output.
- **Model and Temperature Configuration:** Both frameworks allow you to set the model and temperature. You can pass these as arguments when initializing the LLM or program in your code.

This project is for educational purposes. Please ensure you comply with OpenAI's usage policies when using their API.

