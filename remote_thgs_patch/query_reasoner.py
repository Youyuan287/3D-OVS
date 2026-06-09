import base64
import json
import os
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path

from PIL import Image


PART_HEADS = {
    "nose",
    "hooves",
    "hoof",
    "hand",
    "hat",
    "handle",
    "leg",
    "arm",
}


@dataclass
class QueryPlan:
    query: str
    role: str = "object"
    target_terms: list[str] = field(default_factory=list)
    anchor_terms: list[str] = field(default_factory=list)
    part_terms: list[str] = field(default_factory=list)
    contrast_terms: list[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self):
        return asdict(self)


def _dedupe(items):
    out = []
    for item in items:
        item = item.strip()
        if item and item not in out:
            out.append(item)
    return out


def _last_token(text):
    tokens = text.replace("-", " ").split()
    return tokens[-1].lower() if tokens else ""


def _base_modifier(query):
    if " with " not in query:
        return None, None
    base, modifier = query.split(" with ", 1)
    return base.strip(), modifier.strip()


def heuristic_query_plan(query, candidate_labels=None):
    candidate_labels = candidate_labels or []
    base, modifier = _base_modifier(query)
    tail = _last_token(query)

    if base and modifier:
        modifier_head = _last_token(modifier)
        expanded_targets = [
            query,
            f"{modifier} on {base}",
            f"{modifier} attached to {base}",
            f"{base} carrying {modifier}",
            base,
        ]
        expanded_parts = [modifier, modifier_head]
        siblings = [
            label for label in candidate_labels
            if label != query and label.startswith(base + " with ")
        ]
        return QueryPlan(
            query=query,
            role="modifier",
            target_terms=_dedupe(expanded_targets),
            anchor_terms=_dedupe([base]),
            part_terms=_dedupe(expanded_parts),
            contrast_terms=_dedupe(siblings),
            reasoning=(
                "The query is a compositional phrase. First anchor the base object, "
                "then use the modifier to choose among visually similar instances."
            ),
        )

    if tail in PART_HEADS:
        parent = ""
        tokens = query.split()
        if len(tokens) > 1:
            parent = " ".join(tokens[:-1])
        target_terms = [query, tail]
        if parent:
            target_terms.extend([f"{tail} of {parent}", f"{parent}'s {tail}"])
        return QueryPlan(
            query=query,
            role="part",
            target_terms=_dedupe(target_terms),
            anchor_terms=_dedupe([parent]) if parent else [],
            part_terms=_dedupe([tail, query]),
            contrast_terms=[],
            reasoning=(
                "The query names a small part. Prefer fine-level superpoints and "
                "use the part noun to avoid selecting the whole parent object."
            ),
        )

    return QueryPlan(
        query=query,
        role="object",
        target_terms=[query],
        reasoning="The query is treated as an object or standalone instance.",
    )


class QueryReasoner:
    def __init__(self, backend="heuristic", cache_dir=None, model="gpt-4o", enabled=True):
        self.backend = backend
        self.model = model
        self.enabled = enabled
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def plan(self, query, candidate_labels=None, image_path=None, scene=None, frame=None):
        if not self.enabled:
            return heuristic_query_plan(query, candidate_labels)

        cached = self._read_cache(query, scene, frame)
        if cached:
            return QueryPlan(**cached)

        if self.backend == "openai":
            plan = self._openai_plan(query, candidate_labels or [], image_path)
            if plan is None:
                plan = heuristic_query_plan(query, candidate_labels)
        else:
            plan = heuristic_query_plan(query, candidate_labels)

        self._write_cache(plan, scene, frame)
        return plan

    def _cache_path(self, query, scene, frame):
        if not self.cache_dir:
            return None
        safe = query.replace("/", "_").replace("\\", "_").replace(" ", "_")
        parts = [p for p in [scene, frame] if p]
        path = self.cache_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{safe}.json"

    def _read_cache(self, query, scene, frame):
        path = self._cache_path(query, scene, frame)
        if path and path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _write_cache(self, plan, scene, frame):
        path = self._cache_path(plan.query, scene, frame)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(plan.to_dict(), f, indent=2)

    def _openai_plan(self, query, candidate_labels, image_path):
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_API_KEY")
        if not api_key or not image_path:
            return None

        try:
            from openai import OpenAI
        except Exception:
            return None

        with Image.open(image_path).convert("RGB") as img:
            img.thumbnail((768, 768))
            buf = BytesIO()
            img.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = {
            "query": query,
            "candidate_labels": candidate_labels,
            "task": (
                "Return a JSON query plan for text-prompted segmentation. "
                "Classify the query role as object, part, or modifier. "
                "For modifier phrases, identify the base object and distinguishing modifier. "
                "For part phrases, identify the parent/anchor object and the part noun. "
                "Use candidate_labels to add contrast_terms for similar alternatives."
            ),
            "schema": {
                "query": "string",
                "role": "object|part|modifier",
                "target_terms": ["strings used to retrieve the desired mask"],
                "anchor_terms": ["parent/base object terms"],
                "part_terms": ["modifier or part words"],
                "contrast_terms": ["labels that should be disambiguated against"],
                "reasoning": "short rationale, no hidden chain of thought",
            },
        }

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a visual query planner for segmentation. "
                        "Output compact JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": json.dumps(prompt, ensure_ascii=False)},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    ],
                },
            ],
        )
        try:
            data = json.loads(response.choices[0].message.content)
            heuristic = heuristic_query_plan(query, candidate_labels).to_dict()
            heuristic.update({k: data.get(k, heuristic[k]) for k in heuristic})
            return QueryPlan(**heuristic)
        except Exception:
            return None
