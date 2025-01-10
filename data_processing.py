import pandas as pd
from typing import Any, Dict
from ydata_profiling import ProfileReport  # type: ignore
from load_data import flatten_data, load_pickle  


class DataProcessor:
    def __init__(self, pickle_path: str):
        self.pickle_path = pickle_path
        self.data = None
        self.df = None
        self.stats = None

    def load_and_flatten_data(self) -> pd.DataFrame | None:
        self.data = load_pickle(self.pickle_path) # type: ignore
        if self.data is not None:
            self.df = flatten_data(self.data)
        return self.df

    def calculate_statistics(self) -> Dict[str, Any]:
        if self.df is None or self.df.empty:
            print("[ERROR] No data available for statistics calculation.")
            return {}

        stats: Dict[str, Any] = {}
        syndrome_counts = self.df["syndrome_id"].value_counts()
        syndrome_summary = syndrome_counts.reset_index()
        syndrome_summary.columns = ["syndrome_id", "images per syndrome"]
        syndrome_summary["percentage"] = (
            syndrome_summary["images per syndrome"] / len(self.df) * 100
        )
        stats["syndrome_summary"] = syndrome_summary

        self.df["embedding_size"] = self.df["embedding"].apply(len)
        embedding_stats = self.df["embedding_size"].agg(["mean", "min", "max"])
        stats["embedding_stats"] = embedding_stats

        self.stats = stats
        return stats

    def print_statistics(self) -> None:
        if not self.stats:
            print("[ERROR] No statistics available to print.")
            return

        print("\n=== ANALYSIS REPORT ===\n")
        print("-> Count and percentage of rows per 'syndrome_id':")
        print(
            self.stats["syndrome_summary"].to_string(
                index=False, formatters={"percentage": "{:.2f}%".format}
            )
        )
        print(f"\n[INFO] Total rows: {len(self.df)}")
        print("\n-> Statistics of 'embedding' sizes:")
        print(self.stats["embedding_stats"].to_string())
        print("\n=== END OF STATISTICS ===\n")

    def generate_profile_report(self) -> None:
        if self.df is None or self.df.empty:
            print("[ERROR] No data available for profiling.")
            return

        try:
            print("\n-> Generating ProfileReport...")
            profile = ProfileReport(self.df, title="Data Profiling Report", explorative=True)
            profile.to_file("data_profiling_report.html")
            print("[SUCCESS] ProfileReport saved as 'data_profiling_report.html'.")
        except Exception as e:
            print(f"[ERROR] Failed to generate ProfileReport: {e}")

    def process(self) -> pd.DataFrame | None:
        self.load_and_flatten_data()
        if self.df is not None:
            self.calculate_statistics()
            self.print_statistics()
            self.generate_profile_report()
            return self.df
        return None
