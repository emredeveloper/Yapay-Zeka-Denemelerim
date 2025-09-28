import json
import csv
import re
import sys
import argparse
from typing import Dict, Any, Optional, List
from enum import Enum

import ollama

class Sentiment(Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class SchemaType(Enum):
    BASIC = "basic"
    DETAILED = "detailed"
    CUSTOM = "custom"

# Model configuration
MODEL_NAME = "llama3.2:3b"

# Predefined schemas
SCHEMAS = {
    "basic": {
        "title": "string",
        "sentiment": "positive|neutral|negative",
        "keywords": ["string"]
    },
    "detailed": {
        "title": "string",
        "sentiment": "positive|neutral|negative",
        "sentiment_score": "number",
        "keywords": ["string"],
        "summary": "string",
        "entities": {
            "people": ["string"],
            "places": ["string"],
            "organizations": ["string"]
        }
    },
    "custom": {}  # Will be filled dynamically
}

def get_system_prompt(schema_type: SchemaType, custom_schema: Optional[Dict] = None) -> str:
    """Generate system prompt based on schema type"""
    
    if schema_type == SchemaType.CUSTOM and custom_schema:
        schema_desc = json.dumps(custom_schema, ensure_ascii=False)
    else:
        schema_desc = json.dumps(SCHEMAS[schema_type.value], ensure_ascii=False)
    
    return f"""You are a helpful assistant that only outputs valid JSON.
Language: English.
Output Schema: {schema_desc}

Important:
- Always output valid JSON
- Do not include any explanations
- If you cannot determine something, use null
- Follow the schema exactly
"""

def ask_json(prompt: str, schema_type: SchemaType = SchemaType.BASIC, 
             custom_schema: Optional[Dict] = None, max_retries: int = 3) -> Dict[str, Any]:
    """
    Get JSON response from Ollama with retry mechanism
    
    Args:
        prompt: User input prompt
        schema_type: Type of schema to use
        custom_schema: Custom schema for CUSTOM type
        max_retries: Number of retries if JSON parsing fails
    
    Returns:
        Dictionary containing the parsed JSON response
    """
    
    system_prompt = get_system_prompt(schema_type, custom_schema)
    
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                format="json",
                options={
                    "temperature": 0.3,  # Lower temperature for more consistent JSON
                    "top_p": 0.9,
                }
            )
            
            content = response["message"]["content"].strip()
            
            # Clean up response - remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            parsed_data = json.loads(content)

            # Normalize per schema
            if schema_type == SchemaType.DETAILED:
                return _normalize_detailed(parsed_data, prompt)
            elif schema_type == SchemaType.BASIC:
                return _normalize_basic(parsed_data, prompt)
            else:
                # CUSTOM: return as-is; user-defined structure
                return parsed_data
            
        except json.JSONDecodeError as e:
            print(f"JSON parse error (attempt {attempt + 1}/{max_retries}): {e}", 
                  file=sys.stderr)
            if attempt == max_retries - 1:
                return {
                    "error": "Failed to parse JSON response",
                    "raw_content": content,
                    "attempts": attempt + 1
                }
                
        except Exception as e:
            print(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}", 
                  file=sys.stderr)
            if attempt == max_retries - 1:
                return {
                    "error": str(e),
                    "raw_content": content if 'content' in locals() else None,
                    "attempts": attempt + 1
                }
    
    return {"error": "Max retries exceeded"}

def analyze_multiple_texts(texts: List[str], schema_type: SchemaType = SchemaType.BASIC) -> List[Dict]:
    """Analyze multiple texts and return combined results"""
    results = []
    
    for i, text in enumerate(texts):
        print(f"Processing text {i+1}/{len(texts)}...", file=sys.stderr)
        result = ask_json(text, schema_type)
        result["original_text"] = text
        results.append(result)
    
    return results

def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "null", "none", "n/a", "na"}:
        return True
    return False

def _limit_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text

def _normalize_sentiment(value: Any) -> Optional[str]:
    if _is_nullish(value):
        return None
    s = str(value).strip().lower()
    if s in {"positive", "+", "pos"}:
        return "positive"
    if s in {"negative", "-", "neg"}:
        return "negative"
    if s in {"neutral", "neut", "neu", "0"}:
        return "neutral"
    # default fallback
    return "neutral"

def _normalize_sentiment_score(value: Any) -> Optional[float]:
    if _is_nullish(value):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    # clamp to [0,1]
    if score < 0:
        score = 0.0
    if score > 1:
        score = 1.0
    return round(score, 3)

def _clean_keyword(token: str) -> Optional[str]:
    token = token.strip().lower()
    if not token or token in {"null", "none"}:
        return None
    token = re.sub(r"[\s\-_/]+", " ", token)
    token = re.sub(r"[^a-z0-9\s]", "", token)
    token = token.strip()
    return token or None

def _normalize_keywords(value: Any, max_items: int = 5) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    if isinstance(value, list):
        candidates = value
    elif isinstance(value, str):
        candidates = [value]
    else:
        candidates = []
    for item in candidates:
        if not isinstance(item, str):
            continue
        cleaned = _clean_keyword(item)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
        if len(result) >= max_items:
            break
    return result

def _normalize_entity_field(value: Any) -> Optional[List[str]]:
    # Accept list[str], str, dict -> list of keys
    items: List[str] = []
    if isinstance(value, dict):
        # keep keys as detected entity names
        items = [str(k).strip() for k, v in value.items() if str(k).strip()]
    elif isinstance(value, list):
        items = [str(v).strip() for v in value]
    elif isinstance(value, str):
        items = [value]
    else:
        items = []
    cleaned: List[str] = []
    seen: set[str] = set()
    for it in items:
        it_lower = it.lower()
        if _is_nullish(it_lower):
            continue
        it_clean = re.sub(r"\s+", " ", it).strip()
        if not it_clean or it_clean.lower() in seen:
            continue
        seen.add(it_clean.lower())
        cleaned.append(it_clean)
    return cleaned or None

def _normalize_basic(parsed: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    title = parsed.get("title")
    if _is_nullish(title):
        title = _limit_words(" ".join(prompt.strip().split()), 8) or "Untitled"
    else:
        title = _limit_words(str(title).strip(), 8)

    sentiment = _normalize_sentiment(parsed.get("sentiment"))
    keywords = _normalize_keywords(parsed.get("keywords"), max_items=5)
    if not keywords:
        # fallback: pick first 3 distinct tokens > 3 chars from prompt
        tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", prompt.lower()) if len(t) > 3]
        dedup: List[str] = []
        seen: set[str] = set()
        for t in tokens:
            if t in seen:
                continue
            seen.add(t)
            dedup.append(t)
            if len(dedup) >= 3:
                break
        keywords = dedup or ["topic"]

    return {
        "title": title,
        "sentiment": sentiment if sentiment is not None else "neutral",
        "keywords": keywords,
    }

def _normalize_detailed(parsed: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    base = _normalize_basic(parsed, prompt)
    sentiment_score = _normalize_sentiment_score(parsed.get("sentiment_score"))
    summary_val = parsed.get("summary")
    summary = None if _is_nullish(summary_val) else str(summary_val).strip()
    entities_val = parsed.get("entities")
    if not isinstance(entities_val, dict):
        entities_val = {}
    people = _normalize_entity_field(entities_val.get("people"))
    places = _normalize_entity_field(entities_val.get("places"))
    organizations = _normalize_entity_field(entities_val.get("organizations"))
    entities = {
        "people": people,
        "places": places,
        "organizations": organizations,
    }
    return {
        **base,
        "sentiment_score": sentiment_score,
        "summary": summary,
        "entities": entities,
    }

def load_csv_texts(csv_path: str, column: str, limit: int = 10, delimiter: str = ",") -> List[str]:
    """Load up to N texts from a CSV file using a given column name."""
    texts: List[str] = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if column not in (reader.fieldnames or []):
                raise ValueError(f"Column '{column}' not found in CSV headers: {reader.fieldnames}")
            for row in reader:
                value = (row.get(column) or "").strip()
                if value:
                    texts.append(value)
                if len(texts) >= limit:
                    break
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_path}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)
    if not texts:
        print("Warning: No non-empty rows found in the specified CSV column", file=sys.stderr)
    return texts

def create_custom_schema() -> Dict[str, Any]:
    """Create a custom schema interactively"""
    print("Create custom JSON schema (leave field empty to finish):")
    schema = {}
    
    while True:
        field_name = input("Field name: ").strip()
        if not field_name:
            break
            
        field_type = input("Field type (string/number/boolean/array/object): ").strip().lower()
        
        if field_type == "array":
            item_type = input("Array item type: ").strip()
            schema[field_name] = [item_type]
        elif field_type == "object":
            print("Nested object - create sub-fields:")
            sub_schema = create_custom_schema()
            schema[field_name] = sub_schema
        else:
            schema[field_name] = field_type
    
    return schema

def main() -> None:
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Get JSON responses from Ollama")
    parser.add_argument("text", nargs="*", help="Text to analyze")
    parser.add_argument("--schema", "-s", 
                       choices=["basic", "detailed", "custom"],
                       default="basic",
                       help="Schema type to use")
    parser.add_argument("--file", "-f", 
                       help="Read input from file")
    parser.add_argument("--output", "-o",
                       help="Output file (default: stdout)")
    parser.add_argument("--batch", "-b", 
                       action="store_true",
                       help="Process multiple texts")
    parser.add_argument("--pretty", "-p",
                       action="store_true",
                       help="Pretty print JSON output")
    # CSV options
    parser.add_argument("--csv", help="CSV file path to read texts from")
    parser.add_argument("--csv-column", default="content", help="CSV column name to read (default: content)")
    parser.add_argument("--limit", type=int, default=10, help="Number of rows/texts to process (default: 10)")
    parser.add_argument("--csv-delimiter", default=",", help="CSV delimiter (default: ,)")
    
    args = parser.parse_args()
    
    # Get input text
    input_texts = []
    
    if args.csv:
        input_texts = load_csv_texts(
            csv_path=args.csv,
            column=args.csv_column,
            limit=args.limit,
            delimiter=args.csv_delimiter,
        )
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                if args.batch:
                    input_texts = [line.strip() for line in f if line.strip()]
                else:
                    input_texts = [f.read().strip()]
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found", file=sys.stderr)
            sys.exit(1)
    elif args.text:
        if args.batch:
            input_texts = args.text
        else:
            input_texts = [" ".join(args.text)]
    else:
        # Default text
        input_texts = ["The weather is great today; we could go for a picnic."]
    
    # Handle schema type
    schema_type = SchemaType(args.schema)
    custom_schema = None
    
    if schema_type == SchemaType.CUSTOM:
        custom_schema = create_custom_schema()
        if not custom_schema:
            print("No custom schema defined, using basic schema")
            schema_type = SchemaType.BASIC
    
    # Process texts
    if len(input_texts) == 1:
        result = ask_json(input_texts[0], schema_type, custom_schema)
    else:
        result = analyze_multiple_texts(input_texts, schema_type)
    
    # Output results
    output_data = {
        "model": MODEL_NAME,
        "schema_type": schema_type.value,
        "results": result if len(input_texts) > 1 else [result],
        "total_texts": len(input_texts)
    }
    
    # Write output
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, 
                         indent=2 if args.pretty else None)
            print(f"Results saved to {args.output}", file=sys.stderr)
        except Exception as e:
            print(f"Error writing to file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        json.dump(output_data, sys.stdout, ensure_ascii=False,
                 indent=2 if args.pretty else None)
        print()  # Add newline

if __name__ == "__main__":
    main()