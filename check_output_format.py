from pydantic import BaseModel


class AnswerFormat(BaseModel):
    query: "QueryFormat"
    prediction: "PredictionFormat"


class QueryFormat(BaseModel):
    query_id: int
    content: str


class PredictionFormat(BaseModel):
    content: str
    references: list[str]


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import json

    argparse = argparse.ArgumentParser()
    argparse.add_argument("--query_file", type=str, required=True)
    argparse.add_argument("--processed_file", type=str, required=True)
    args = argparse.parse_args()

    query_file = Path(args.query_file)
    processed_file = Path(args.processed_file)

    # check number of lines match
    with open(query_file, "r", encoding="utf-8") as f:
        query_lines = [line for line in f.readlines() if line.strip()]
    with open(processed_file, "r", encoding="utf-8") as f:
        processed_lines = [line for line in f.readlines() if line.strip()]
    if len(query_lines) != len(processed_lines):
        raise Exception(
            f"Number of lines in query file ({len(query_lines)}) does not match number of lines in processed file ({len(processed_lines)})"
        )

    # check format and collect query ids
    needed_query_ids = set()
    for line in query_lines:
        data = json.loads(line)
        needed_query_ids.add(data["query"]["query_id"])

    found_query_ids = set()
    for line_number, line in enumerate(processed_lines, 1):
        try:
            formatted_data = AnswerFormat(**json.loads(line))
            found_query_ids.add(formatted_data.query.query_id)
        except Exception as e:
            raise Exception(f"Format error in line {line_number}: {line}\n\n{e}")

    # check all query ids are present
    missing_query_ids = needed_query_ids - found_query_ids
    if missing_query_ids:
        raise Exception(f"Missing query ids in processed file: {missing_query_ids}")
