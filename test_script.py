from pathlib import Path
import argparse

import pandas as pd


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Read a parquet file and print total row count with first 5 rows."
	)
	parser.add_argument(
		"path",
		nargs="?",
		default="data/telemetry_all.parquet",
		help="Parquet file path (default: data/telemetry_all.parquet)",
	)
	args = parser.parse_args()

	parquet_path = Path(args.path)
	if not parquet_path.exists():
		raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

	df = pd.read_parquet(parquet_path)
	print(f"Toplam satir sayisi: {len(df)}")
	print("Ilk 5 satir:")
	print(df.head(5))


if __name__ == "__main__":
	main()
