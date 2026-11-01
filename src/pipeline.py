"""Run each ingestion step on its own:  python -m src.pipeline <step>

Steps: scrape (2) -> load (3) -> chunk (4) -> embed (5) -> index (6) -> export (6b, for Vercel), or `all`.
"""
import argparse
import importlib

STEPS = {
    "scrape": "src.scraper",
    "load": "src.loader",
    "chunk": "src.chunker",
    "embed": "src.embedder",
    "index": "src.vector_store",
    "export": "src.export_serving",   # 6b: serving index + ONNX model for Vercel
}


def run_step(name: str, force: bool = False) -> None:
    try:
        module = importlib.import_module(STEPS[name])
    except ModuleNotFoundError as e:
        if e.name == STEPS[name]:
            raise SystemExit(f"Step '{name}' is not implemented yet ({STEPS[name]}.py missing).")
        raise
    print(f"== {name} ==")
    if force and name == "scrape":
        module.run(force=True)
    else:
        module.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="MF FAQ RAG ingestion pipeline")
    parser.add_argument("step", choices=[*STEPS, "all"])
    parser.add_argument("--force", action="store_true", help="scrape: re-save pages even if unchanged")
    args = parser.parse_args()
    for name in (STEPS if args.step == "all" else [args.step]):
        run_step(name, args.force)


if __name__ == "__main__":
    main()
