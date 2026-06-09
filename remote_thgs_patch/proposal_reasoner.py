import base64
import json
import os
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def mask_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def draw_som(image_rgb, masks, labels=None, alpha=0.45):
    labels = labels or [str(i + 1) for i in range(len(masks))]
    out = image_rgb.copy()
    palette = [
        (255, 0, 0), (0, 255, 255), (0, 255, 0), (255, 0, 255),
        (255, 255, 0), (0, 128, 255), (255, 128, 0), (128, 0, 255),
    ]
    for i, mask in enumerate(masks):
        if mask.sum() == 0:
            continue
        color = np.asarray(palette[i % len(palette)], dtype=np.uint8)
        out[mask] = (out[mask] * (1 - alpha) + color * alpha).astype(np.uint8)
        x0, y0, x1, y1 = mask_bbox(mask)
        cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
        cv2.circle(out, (cx, cy), 13, (255, 255, 255), -1)
        cv2.circle(out, (cx, cy), 13, (0, 0, 0), 2)
        cv2.putText(out, labels[i], (cx - 7, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return out


class ProposalReasoner:
    def __init__(self, backend="heuristic", model="gpt-4o", cache_dir=None):
        self.backend = backend
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def select(self, image_rgb, masks, metadata, query_plan, scene=None, frame=None, query=None):
        if not masks:
            return []
        self._write_debug(image_rgb, masks, metadata, query_plan, scene, frame, query)
        manual = self._read_manual_selection(scene, frame, query)
        if manual:
            return manual
        if self.backend == "openai":
            picked = self._openai_select(image_rgb, masks, metadata, query_plan)
            if picked:
                return picked
        return []

    def _read_manual_selection(self, scene, frame, query):
        base = self._debug_base(scene, frame, query)
        if base is None:
            return []
        path = Path(str(base) + "_selection.json")
        if not path.exists():
            return []
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
            return [int(i) for i in data.get("selected_ids", [])]
        except Exception:
            return []

    def _debug_base(self, scene, frame, query):
        if not self.cache_dir:
            return None
        safe_query = (query or "query").replace("/", "_").replace("\\", "_").replace(" ", "_")
        path = self.cache_dir
        for item in [scene, frame]:
            if item:
                path = path / item
        path.mkdir(parents=True, exist_ok=True)
        return path / safe_query

    def _write_debug(self, image_rgb, masks, metadata, query_plan, scene, frame, query):
        base = self._debug_base(scene, frame, query)
        if base is None:
            return
        som = draw_som(image_rgb, masks)
        Image.fromarray(som).save(str(base) + "_som.jpg")
        np.savez_compressed(str(base) + "_masks.npz", masks=np.asarray(masks, dtype=np.uint8))
        payload = {
            "query_plan": query_plan.to_dict() if hasattr(query_plan, "to_dict") else query_plan,
            "candidates": metadata,
        }
        with open(str(base) + "_candidates.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _openai_select(self, image_rgb, masks, metadata, query_plan):
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPEN_API_KEY")
        if not api_key:
            return []
        try:
            from openai import OpenAI
        except Exception:
            return []

        som = draw_som(image_rgb, masks)
        img = Image.fromarray(som)
        img.thumbnail((1024, 1024))
        buf = BytesIO()
        img.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        prompt = {
            "query_plan": query_plan.to_dict() if hasattr(query_plan, "to_dict") else query_plan,
            "candidates": metadata,
            "instruction": (
                "Choose the candidate ids that best match the query. "
                "Use the visual numbered masks, not only candidate text scores. "
                "For part queries, prefer a small part inside the anchor object. "
                "For modifier queries, distinguish against contrast terms."
            ),
            "schema": {"selected_ids": ["candidate integer ids"], "reason": "short explanation"},
        }
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a visual proposal selector. Output JSON only."},
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
            return [int(i) for i in data.get("selected_ids", [])]
        except Exception:
            return []
