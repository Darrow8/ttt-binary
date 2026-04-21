# Training Problem Quality and Answer Confidence Ranking

This script uses Vertex AI's GPT-OSS model to evaluate training problems for their usefulness in learning to solve a target problem.

## Overview

The `rank_problems.py` script:
- Reads a JSON file containing training problems and answers
- Takes a target problem as input
- Uses the Vertex AI API to evaluate each training problem
- Ranks **Problem Quality** (1-10): How useful is this problem for learning to solve the target problem?
- Ranks **Answer Confidence** (1-10): How confident are we in the correctness of the answer?
- Outputs results to a JSON file with rankings included

## Prerequisites

### Environment Setup

1. Set up Google Cloud credentials:
   ```bash
   export GOOGLE_CLOUD_PROJECT='your-project-id'
   export GOOGLE_CLOUD_LOCATION='us-central1'  # optional, defaults to us-central1
   ```

2. Ensure you have authenticated with Google Cloud:
   ```bash
   gcloud auth application-default login
   ```

3. Install required Python packages:
   ```bash
   pip install openai tenacity google-auth
   ```

### Environment File (Optional)

You can create a `.env` file in the parent directory (`../.env`) with:
```
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

## Input Format

### Training Problems File

The script accepts JSON files in flexible formats:

#### Format 1: List of problems
```json
[
  {
    "problem": "Find the sum of the first 50 positive integers.",
    "answer": "1275"
  },
  {
    "problem": "Evaluate: 1 + 2 + 3 + ... + 20",
    "answer": "210"
  }
]
```

#### Format 2: Object with problems array
```json
{
  "problems": [
    {
      "problem": "Find the sum of the first 50 positive integers.",
      "answer": "1275"
    }
  ]
}
```

#### Flexible Field Names

The script accepts various field names:
- Problem text: `problem`, `question`, or `prompt`
- Answer text: `answer`, `solution`, or `response`

### Target Problem

You can provide the target problem in two ways:

#### Option 1: Command-line argument
```bash
python rank_problems.py input.json --target "Find the sum of the first 100 positive integers."
```

#### Option 2: JSON file
Create a target file (e.g., `target.json`):
```json
{
  "target": "Find the sum of the first 100 positive integers.",
  "context": "This is a classic summation problem."
}
```

Then run:
```bash
python rank_problems.py input.json --target-file target.json
```

## Usage

### Basic Usage (with --target)
```bash
python rank_problems.py training_problems.json --target "Your target problem here"
```

### With Target File
```bash
python rank_problems.py training_problems.json --target-file target.json
```

This will create `training_problems_ranked.json` with the results.

### Specify Output File
```bash
python rank_problems.py training_problems.json --target-file target.json -o output.json
```

### Verbose Mode
Show detailed prompts and LLM responses:
```bash
python rank_problems.py training_problems.json --target-file target.json -v
```

### Limit Number of Problems
Process only the first N problems:
```bash
python rank_problems.py training_problems.json --target-file target.json --limit 10
```

## Output Format

The output JSON file contains the target problem and all ranked training problems:

```json
{
  "target_problem": "Find the sum of the first 100 positive integers.",
  "problems": [
    {
      "problem": "Find the sum of the first 50 positive integers.",
      "answer": "1275",
      "problem_quality": 8,
      "answer_confidence": 10,
      "ranking_response": "Problem Quality: 8\nAnswer Confidence: 10\n..."
    }
  ]
}
```

### Output Fields

- `problem_quality`: Score from 1-10 rating how useful this training problem is for learning to solve the target problem
- `answer_confidence`: Score from 1-10 rating confidence in the correctness of the answer
- `ranking_response`: (Only in verbose mode) Raw LLM response

## Scoring Criteria

### Problem Quality (1-10)

Evaluates how useful this training problem is for learning to solve the target problem:

- **1-3 (Poor)**: Unrelated, trivial, or misleading for learning the target problem
  - Teaches irrelevant skills
  - Too trivial to provide meaningful practice
  - Potentially confuses or misleads the learner
  
- **4-6 (Moderate)**: Somewhat relevant but not ideal training data
  - Partially relevant skills
  - May be too easy or too hard
  - Some connection to target problem but not optimal
  
- **7-10 (Excellent)**: Highly relevant, teaches key skills, well-posed
  - Directly teaches skills needed for target problem
  - Appropriate difficulty level
  - Provides good practice for required reasoning
  - Complements other training problems well

### Answer Confidence (1-10)

Evaluates confidence in the correctness of the answer:

- **1-3 (Low)**: Likely incorrect or incomplete
  - Mathematical errors present
  - Missing key steps or components
  - Unclear reasoning
  
- **4-6 (Moderate)**: Probably correct but some uncertainty
  - Answer appears correct
  - Some ambiguity in solution
  - Minor issues possible
  
- **7-10 (High)**: Clearly correct and complete
  - Mathematically verified
  - Complete solution with clear reasoning
  - No ambiguity or errors

## Example

```bash
# Run on example files
python rank_problems.py example_input.json --target-file example_target.json -v

# Check results
cat example_input_ranked.json
```

### Example Training Set

Given the target problem: "Find the sum of the first 100 positive integers."

Training problems might include:
- "Find the sum of the first 50 positive integers" (Quality: 8-9, directly relevant)
- "Find the sum of all even numbers from 2 to 100" (Quality: 6-7, related but different pattern)
- "What is the derivative of x^2?" (Quality: 1-2, unrelated to summation)

## Command-Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `input` | - | Input JSON file containing training problems (required) |
| `--target` | - | Target problem as a string |
| `--target-file` | - | JSON file containing the target problem |
| `--output` | `-o` | Output JSON file (default: input_ranked.json) |
| `--verbose` | `-v` | Print detailed prompts and responses |
| `--limit` | `-l` | Limit number of problems to process |

**Note**: You must provide either `--target` or `--target-file` (but not both).

## Troubleshooting

### "Target problem is required"
Make sure to provide either:
```bash
--target "Your problem here"
```
or
```bash
--target-file target.json
```

### "Set GOOGLE_CLOUD_PROJECT environment variable"
Make sure you've set the environment variable:
```bash
export GOOGLE_CLOUD_PROJECT='your-project-id'
```

### Authentication Errors
Run:
```bash
gcloud auth application-default login
```

### "LLM response has no choices"
This may indicate an API issue. Try again with verbose mode to see more details:
```bash
python rank_problems.py input.json --target-file target.json -v
```

### Could not parse scores from response
The LLM may have formatted its response differently than expected. Use verbose mode to inspect the actual response and adjust parsing if needed.

## Use Case

This tool is designed for:
- Building training sets for mathematical problem-solving models
- Filtering candidate problems by relevance to a target problem
- Ensuring answer quality in training data
- Creating progressive curricula (start with high-quality easier problems)

## Related Files

- `../Stage1/distinct_llm_prompting.py` - Original Vertex AI client implementation
- `example_input.json` - Example training problems file
- `example_target.json` - Example target problem file