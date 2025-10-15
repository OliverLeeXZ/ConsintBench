import re
import json
import time
from typing import List, Dict, Optional, Any

from openai_api import get_openai_client
from utils import ensure_json_object
from config import OPENAI_KEYWORD_MODEL

def extract_keywords_with_gpt(
    question_text: str,
    options: Optional[List[str]] = None,
    model: str = OPENAI_KEYWORD_MODEL,
    max_retries: int = 3,
    temperature: float = 0.0,
    base_url: Optional[str] = None,
    max_items: int = 8,
) -> Dict:
    """
    Output:
    {
      "products": [...],
      "models": [...],
      "functions": [...],
      "keywords": [...],   # same as ranked_keywords, in descending order of importance
      "query": "..."       # ranked_keywords joined with spaces
    }
    """
    if not question_text:
        return {"products": [], "models": [], "functions": [], "keywords": [], "query": ""}

    # ---------------- Heuristic fallback: extract "function/feature" noun phrases from original sentence ----------------
    def _heuristic_functions(text: str, max_len_tokens: int = 4) -> List[str]:
        import re

        # Common function/feature trigger words (can be expanded as needed)
        cue_words = {
            "interface","ui","ux","usability","discoverability","discovery","app discovery",
            "performance","latency","battery life","battery","charging","comfort",
            "weight distribution","price","safety","range","resolution","refresh rate",
            "field of view","fov","hand tracking","eye tracking","setup","onboarding",
            "multitasking","notifications","privacy","security","heat","thermal","audio",
            "display","compatibility","integration","gesture","voice control","search"
        }

        # Extract n-grams (1-4), only keep consecutive token fragments that appear in original text
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", text)
        lower_text = text.lower()
        # Generate n-grams
        ngrams = set()
        for n in range(1, 5):
            for i in range(0, max(0, len(tokens) - n + 1)):
                span = " ".join(tokens[i:i+n])
                if len(span.split()) <= max_len_tokens:
                    ngrams.add(span)

        # Match cue words or n-grams containing cue words
        hits = set()
        for g in ngrams:
            g_l = g.lower()
            if g_l in cue_words:
                hits.add(g)

        # Also directly check if compound phrases (e.g., "app discovery") appear in original sentence
        for cue in list(cue_words):
            if cue in lower_text:
                hits.add(cue)

        uniq = []
        seen = set()
        for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9\-']*(?:\s+[A-Za-z0-9][A-Za-z0-9\-']*){0,3}", text):
            frag = m.group(0).strip()
            frag_l = frag.lower()
            # Only collect those matching hits (case as in original)
            for h in list(hits):
                if frag_l == h.lower() and frag not in seen:
                    uniq.append(frag)
                    seen.add(frag)
        # Limit length (<=4 tokens)
        uniq = [s for s in uniq if len(s.split()) <= max_len_tokens]
        # Remove pure stopwords
        stop = {"the","a","an","to","of","for","and","or","in","on","at","is","are","what","do","does","think","user","users"}
        def _is_stop_only(s: str):
            toks = [t.lower() for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", s)]
            return all(t in stop for t in toks) or len(toks) == 0
        uniq = [s for s in uniq if not _is_stop_only(s)]
        return uniq

    opt_text = ""
    if options:
        short_opts = [o for o in options if isinstance(o, str) and len(o.split()) <= 6]
        if short_opts:
            opt_text = "Options: " + " | ".join(short_opts)

    # ---------------- Enhanced system prompt (emphasizing functions definition + counterexamples + few-shot) ----------------
    sys_prompt = (
        "You are an advanced keyword extraction system specializing in consumer electronics and user experience questions.\n\n"
        "GOAL: Extract precise, search-optimized keywords from product-related questions that will yield the most relevant search results.\n\n"
        
        "INSTRUCTIONS:\n"
        "1. Extract 5-8 high-value keywords/phrases from the provided question.\n"
        "2. PRIORITIZE: product names, specific features, technical terms, user experience aspects, comparative terms.\n"
        "3. INCLUDE: noun phrases, specific components, actions, problems, emotions, technical specifications.\n"
        "4. EXCLUDE: stop words, generic terms ('user', 'think', 'consider'), question words, auxiliary verbs.\n"
        "5. MAINTAIN: original casing and exact terms used in the questions.\n"
        "6. WEIGH: distinctive technical terms > specific features > general aspects.\n\n"

        "CATEGORIES TO EXTRACT:\n"
        "- functions: exact feature/capability/characteristic tokens or short noun phrases, e.g., "
        "'interface', 'battery life', 'hand tracking', 'charging', 'weight distribution', 'heat'.\n"
        "- use_cases: exact phrases describing usage scenarios or contexts of use, e.g., "
        "'watching movies', 'productivity', 'virtual desktop', 'gaming', 'meetings', 'app discovery'. "
        "These often appear after 'for', 'to', 'during', 'while', 'when', or as gerunds/noun phrases.\n"
        "- attitudes: evaluative or sentiment-bearing words/phrases that express opinions or feelings, e.g., "
        "'like', 'dislike', 'satisfied', 'annoyed', 'positive', 'negative'.\n\n"

        "QUALITY RULES:\n"
        "1) Specificity First: Extract the MOST SPECIFIC surface form. Prefer spans with concrete heads/modifiers: "
        "   numbers/units (e.g., '120 Hz', '2 hours'), components ('USB-C', 'hand tracking'), precise attributes "
        "   ('weight distribution', 'field of view').\n"
        "2) Prefer Specific Over Generic: Drop vague terms unless SPECIFICALLY anchored by a concrete modifier. "
        "   AVOID generic terms like: 'aspect', 'experience', 'thing', 'stuff', 'issue', 'quality', 'performance', "
        "   'function', 'capability', 'feature' when standalone. These are allowed only when specifically anchored, "
        "   e.g., 'GPU performance', 'audio quality', 'charging performance'.\n"
        "3) Choose Complete Phrases: Prefer 'battery life' over 'battery'; 'hand tracking accuracy' over 'hand tracking' "
        "   when both appear. Maximum length: 4 tokens.\n"
        "4) When two candidates overlap, keep the more specific one (with technical terms/components/specifications).\n\n"

        "EXAMPLES:\n"
        "Q: What do users think of the interface for app discovery?\n"
        "functions → ['interface']; use_cases → ['app discovery']; attitudes → []\n"
        "ranked_keywords → ['interface', 'app discovery']\n\n"
        
        "Q: Does it suffer from heat or battery life issues during gaming?\n"
        "functions → ['heat','battery life']; use_cases → ['gaming']; attitudes → []\n"
        "ranked_keywords → ['battery life', 'heat', 'gaming']\n\n"
        
        "Q: Are you satisfied or dissatisfied with charging during meetings?\n"
        "functions → ['charging']; use_cases → ['meetings']; attitudes → ['satisfied','dissatisfied']\n"
        "ranked_keywords → ['charging', 'satisfied', 'dissatisfied', 'meetings']\n\n"
        
        "Q: Which aspect of VR experience matters most: performance or hand tracking accuracy?\n"
        "functions → ['hand tracking accuracy']; use_cases → []; attitudes → []\n"
        "ranked_keywords → ['hand tracking accuracy']  # Note: 'aspect', 'VR experience', 'performance' dropped as too generic\n"
    )

    user_prompt = (
        f"Question: {question_text}\n"
        f"{opt_text}\n\n"
        "EXTRACT SEARCH-OPTIMIZED KEYWORDS:\n"
        "- Focus on technical terms and specific product features\n"
        "- Capture core functionality and user experience aspects\n"
        "- Include comparative elements and quality descriptors\n"
        "- Extract usage scenarios and problem descriptions\n\n"
        
        "OUTPUT REQUIREMENTS:\n"
        "0) Return ONLY valid JSON with NO additional text\n"
        "1) All extracted keywords MUST be exact substrings from the question (verbatim)\n"
        "2) Categories to extract:\n"
        "   - functions: specific feature/capability terms (≤4 tokens), prefer technical phrases\n"
        "   - use_cases: usage scenarios/contexts as short phrases (≤4 tokens)\n"
        "   - attitudes: opinion/sentiment words or phrases (≤3 tokens)\n"
        "3) ranked_keywords: Create a single list ordered by IMPORTANCE (highest first):\n"
        "   - Place specific functions first\n"
        "   - Then relevant use cases\n"
        "   - Then attitudes/opinions if present\n"
        "   - Include 5-8 items total for optimal search relevance\n"
        "4) Eliminate duplicates and overlapping terms (keep only the most specific)\n"
        "5) Generate 'query' by joining ranked_keywords with spaces\n\n"
        
        "Remember: The quality of search results depends directly on your keyword selection."
    )

    # ---------------- JSON Schema (including attitudes; excluding brands/models/product names) ----------------
    max_keep = max(8, min(8, max_items))
    response_schema = {
        "name": "keyword_schema",
        "schema": {
            "type": "object",
            "properties": {
                "functions": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "use_cases": {"type": "array", "items": {"type": "string"}, "minItems": 0},
                "attitudes": {"type": "array", "items": {"type": "string"}, "minItems": 0},
                "ranked_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": max_keep
                },
                "query": {"type": "string"}
            },
            "required": ["functions", "use_cases", "attitudes", "ranked_keywords", "query"],
            "additionalProperties": False
        }
    }
    client = get_openai_client(base_url=base_url)
    def _clean_list(x: Any) -> List[str]:
        import re
        if not x:
            return []
        if isinstance(x, str):
            x = [x]
        out = []
        seen = set()
        stop = {"the","a","an","to","of","for","and","or","in","on","at","is","are","what","do","does","think","user","users","please"}
        for s in x:
            if not isinstance(s, str):
                continue
            s2 = s.strip()
            if not s2:
                continue
            # Limit to ≤4 tokens
            if len(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", s2)) > 4:
                continue
            # Remove pure stopwords
            toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", s2)
            if all(t.lower() in stop for t in toks):
                continue
            if s2 not in seen:
                out.append(s2)
                seen.add(s2)
        return out

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            data = None
            try:
                resp = client.responses.create(
                    model=model,
                    temperature=temperature,
                    response_format={"type": "json_schema", "json_schema": response_schema},
                    input=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                )
                try:
                    data = json.loads(resp.output_text)
                except Exception:
                    raw = (
                        resp.output[0].content[0].text
                        if getattr(resp, "output", None)
                        else resp.choices[0].message.content
                    )
                    data = ensure_json_object(raw) or json.loads(raw)
            except TypeError:
                # Fall back to Chat Completions (JSON mode)
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                )
                raw = resp.choices[0].message.content
                data = ensure_json_object(raw) or json.loads(raw)

            products = _clean_list(data.get("products", []))
            models = _clean_list(data.get("models", []))
            functions_from_llm = _clean_list(data.get("functions", []))
            ranked = _clean_list(data.get("ranked_keywords", []))
            query = str(data.get("query", "") or "").strip()

            # ---- Heuristic fallback, ensure functions are accurate and don't miss key info ----
            heur = _heuristic_functions(question_text)
            # Merge with LLM results (maintain original case, prioritize exact form of heuristic phrases)
            func_set = []
            seen = set()
            for s in heur + functions_from_llm:
                if s not in seen:
                    func_set.append(s)
                    seen.add(s)
            functions = _clean_list(func_set)

            # If functions is still empty but can be extracted from sentence, keep at least 1-2
            if not functions and heur:
                functions = _clean_list(heur[:2])

            # If ranked is empty, prioritize functions > models > products
            if not ranked and (products or models or functions):
                ranked = (functions + models + products)[:max_keep]

            # Fallback for query
            if not query:
                query = " ".join(ranked[:max_keep])

            # Truncate
            ranked = ranked[:max_keep]

            return {
                "products": products,
                "models": models,
                "functions": functions,
                "keywords": ranked,
                "query": query,
            }

        except Exception as e:
            last_err = e
            time.sleep(1.0 + 0.2 * attempt)

    return {"products": [], "models": [], "functions": [], "keywords": [], "query": ""}
