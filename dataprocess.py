import argparse
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _stringify(value: Any) -> str:
	if value is None:
		return ""
	if isinstance(value, str):
		return value.strip()
	if isinstance(value, (list, tuple)):
		return "\n".join(_stringify(item) for item in value if item is not None).strip()
	if isinstance(value, dict):
		return json.dumps(value, ensure_ascii=False)
	return str(value).strip()


def _extract_input_output(record: Dict[str, Any]) -> Optional[Tuple[str, str]]:
	if "input" in record and "output" in record:
		inp = _stringify(record.get("input"))
		out = _stringify(record.get("output"))
		return inp, out

	instruction = _stringify(record.get("instruction") or record.get("prompt"))
	context = _stringify(record.get("context") or record.get("article") or record.get("text"))
	question = _stringify(record.get("question") or record.get("query"))
	raw_input = _stringify(record.get("input"))

	input_parts = [part for part in [instruction, raw_input, question, context] if part]
	inp = "\n\n".join(input_parts).strip()

	out = _stringify(
		record.get("output")
		or record.get("answer")
		or record.get("response")
		or record.get("completion")
		or record.get("label")
	)

	if not inp or not out:
		return None
	return inp, out


def _iter_records(raw: Any) -> Iterable[Dict[str, Any]]:
	if isinstance(raw, list):
		for item in raw:
			if isinstance(item, dict):
				yield item
	elif isinstance(raw, dict):
		if "data" in raw and isinstance(raw["data"], list):
			for item in raw["data"]:
				if isinstance(item, dict):
					yield item
		else:
			yield raw


def process_dataset(input_path: str, output_path: str, max_samples: Optional[int]) -> None:
	with open(input_path, "r", encoding="utf-8") as f:
		raw = json.load(f)

	processed: List[Dict[str, str]] = []
	skipped = 0

	for record in _iter_records(raw):
		extracted = _extract_input_output(record)
		if not extracted:
			skipped += 1
			continue
		inp, out = extracted
		processed.append({"input": inp, "output": out})
		if max_samples and len(processed) >= max_samples:
			break

	with open(output_path, "w", encoding="utf-8") as f:
		for row in processed:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")

	print(f"Processed: {len(processed)}")
	print(f"Skipped: {skipped}")
	print(f"Saved: {output_path}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Prepare dataset for LLM fine-tuning.")
	parser.add_argument(
		"--input",
		default="medical_meadow_wikidoc/medical_meadow_wikidoc.json",
		help="Path to the raw dataset JSON.",
	)
	parser.add_argument(
		"--output",
		default="data/medical_meadow_wikidoc.jsonl",
		help="Path to write normalized JSONL with input/output fields.",
	)
	parser.add_argument(
		"--max-samples",
		type=int,
		default=None,
		help="Limit number of samples (for quick tests).",
	)
	args = parser.parse_args()

	process_dataset(args.input, args.output, args.max_samples)


if __name__ == "__main__":
	main()
