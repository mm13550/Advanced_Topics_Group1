"""
Query decomposition for legal RAG system.

Splits a natural language user query into:
  - A clean semantic search query (concepts only)
  - Structured metadata filters (year_min, year_max, court_contains)

These are passed separately to ChromaDBRetrievalPipeline.retrieve()
so that ChromaDB can apply metadata filters natively at the database
level rather than embedding constraints into the semantic query.

Model: llama-3.1-8b-instant (fast, cheap, reliable JSON output)
"""

import json
import os
from dotenv import load_dotenv
from groq import Groq


_SYSTEM_PROMPT = """You are a query parser for a legal case retrieval system.

Given a user's natural language legal query, extract:
1. A semantic search query containing rich legal concepts and terminology.
   Remove only: date ranges, court names, and procedural instructions
   (e.g. "find", "show me", "decided after", "excluding").
   Preserve: constitutional provisions, legal doctrines, cause of action
   names, legal standards, and all other substantive legal terminology.
2. Any metadata filters for date range or court name present in the query.

Respond ONLY with a valid JSON object. No preamble, no explanation, no markdown.

JSON schema:
{
  "semantic_query": string,
  "year_min": integer or null,
  "year_max": integer or null,
  "court_contains": string or null
}

Rules:
- semantic_query must never contain years, date ranges, or court names
- semantic_query should be rich in legal terminology
- year_min and year_max are calendar years as integers (e.g. 2000)
- "after YYYY" means year_min = YYYY + 1, "before YYYY" means year_max = YYYY - 1
- "since YYYY" and "from YYYY" mean year_min = YYYY
- court_contains should be a partial court name (e.g. "Supreme", "Ninth Circuit")
- If no filters are present, return null for those fields
- If the query is purely semantic with no filters, still return all fields with null values

Examples:
Input: "Find cases involving freedom of speech decided after 1960"
Output: {"semantic_query": "freedom of speech First Amendment constitutional protection", "year_min": 1961, "year_max": null, "court_contains": null}

Input: "What standard of review applies in First Amendment cases?"
Output: {"semantic_query": "standard of review First Amendment constitutional scrutiny", "year_min": null, "year_max": null, "court_contains": null}

Input: "Find Supreme Court contract cases between 1950 and 1970"
Output: {"semantic_query": "contract formation breach enforcement obligations", "year_min": 1950, "year_max": 1970, "court_contains": "Supreme Court"}

Input: "Can police search my house without a warrant?"
Output: {"semantic_query": "Fourth Amendment warrant requirement police search seizure", "year_min": null, "year_max": null, "court_contains": null}"""

_USER_PROMPT_TEMPLATE = 'User query: "{query}"'


class QueryDecomposer:
    """
    Decomposes natural language legal queries into semantic query
    and structured metadata filters using a fast LLM call.
    """

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        """
        Initializes the decomposer with a Groq client.

        Args:
            model: Groq model ID to use for decomposition.
                   Defaults to llama-3.1-8b-instant for speed and cost.
        """
        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")

        self.client = Groq(api_key=api_key)
        self.model = model

    def decompose(self, query: str) -> dict:
        """
        Decomposes a user query into semantic query and metadata filters.

        On any failure (API error, malformed JSON, missing fields),
        falls back to returning the raw query with no filters so
        retrieval is never interrupted.

        Args:
            query: Raw natural language query from the user.

        Returns:
            Dict with keys:
              semantic_query: str  - clean query for embedding
              year_min:       int or None
              year_max:       int or None
              court_contains: str or None
              was_decomposed: bool - False if fallback was used
        """
        fallback = {
            "semantic_query": query,
            "year_min": None,
            "year_max": None,
            "court_contains": None,
            "was_decomposed": False
        }

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(query=query)}
                ],
                max_tokens=150,
                temperature=0.0
            )

            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)

            if "semantic_query" not in parsed:
                print("QueryDecomposer: missing semantic_query, returning entire query")
                return fallback

            # Sanitize types - LLM may return strings instead of ints
            year_min = parsed.get("year_min")
            year_max = parsed.get("year_max")
            court_contains = parsed.get("court_contains")

            if year_min is not None:
                year_min = int(year_min)
            if year_max is not None:
                year_max = int(year_max)
            if court_contains is not None:
                court_contains = str(court_contains).strip() or None

            return {
                "semantic_query": parsed["semantic_query"],
                "year_min": year_min,
                "year_max": year_max,
                "court_contains": court_contains,
                "was_decomposed": True
            }

        except json.JSONDecodeError:
            print(f"QueryDecomposer: JSON parse failed, using fallback. Raw: {raw!r}")
            return fallback
        except Exception as e:
            print(f"QueryDecomposer: unexpected error ({e}), using fallback")
            return fallback