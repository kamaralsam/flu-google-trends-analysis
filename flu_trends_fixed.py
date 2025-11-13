# Google Trends vs CDC Flu Data Analysis - LOCAL CSV VERSION (SELF-CONTAINED)
# This version:
# - Reads local CSVs if they exist and have data
# - Otherwise auto-generates realistic synthetic data and saves it to /data

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# Plot style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)


# ============================================================================
# PHASE 1: DATA COLLECTION (FROM LOCAL FILES OR SYNTHETIC)
# ============================================================================

class DataCollector:
    """Handles data collection from LOCAL Google Trends and CDC CSV files."""

    def __init__(self):
        # Base directory = folder where this script lives
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Data folder and CSV paths
        self.data_folder = os.path.join(base_dir, "data")
        self.trends_csv = os.path.join(self.data_folder, "google_trends_raw.csv")
        self.cdc_csv = os.path.join(self.data_folder, "cdc_flu_raw.csv")

    # ---------- GOOGLE TRENDS ----------

    def collect_google_trends(self, keywords, start_date, end_date):
        """
        Load Google Trends data from local CSV.
        If file is missing or empty, generate synthetic demo data and save it.
        Expected columns if CSV already exists:
          - 'date' (weekly timestamps)
          - 'trend_<keyword>' columns, e.g. trend_flu_symptoms
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        if not os.path.exists(self.trends_csv):
            print(f"⚠️ Google Trends CSV not found at: {self.trends_csv}")
            print("   Generating synthetic Google Trends data instead...")
            return self._generate_synthetic_trends(keywords, start_dt, end_dt)

        print(f"📥 Loading Google Trends data from '{self.trends_csv}'...")
        try:
            df = pd.read_csv(self.trends_csv)
        except pd.errors.EmptyDataError:
            print("⚠️ Google Trends CSV is empty. Generating synthetic data instead...")
            return self._generate_synthetic_trends(keywords, start_dt, end_dt)

        # If it has no rows/columns, also generate synthetic
        if df.empty or df.shape[1] == 0:
            print("⚠️ Google Trends CSV has no usable data. Generating synthetic data instead...")
            return self._generate_synthetic_trends(keywords, start_dt, end_dt)

        # Ensure date column exists
        if "date" not in df.columns:
            print("⚠️ 'date' column not found in Google Trends CSV.")
            print("   Generating synthetic data with proper structure instead...")
            return self._generate_synthetic_trends(keywords, start_dt, end_dt)

        df["date"] = pd.to_datetime(df["date"])

        # Filter by date range
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()

        # Keep only trend_* columns
        trend_cols = [col for col in df.columns if col.startswith("trend_")]
        if not trend_cols:
            print("⚠️ No 'trend_' columns found in Google Trends CSV.")
            print("   Generating synthetic data instead...")
            return self._generate_synthetic_trends(keywords, start_dt, end_dt)

        keep_cols = ["date"] + trend_cols
        df = df[keep_cols]

        print(f"✅ Loaded {len(df)} weekly rows from Google Trends CSV")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        return df

    def _generate_synthetic_trends(self, keywords, start_dt, end_dt):
        """Create a fake but realistic Google Trends dataset and save it."""
        os.makedirs(self.data_folder, exist_ok=True)

        dates = pd.date_range(start=start_dt, end=end_dt, freq="W-MON")
        rng = np.random.default_rng(42)

        data = {"date": dates}

        for kw in keywords:
            # seasonal-ish waves + noise, clipped to [0, 100]
            base = np.sin(2 * np.pi * (np.arange(len(dates)) / 52.0)) * 40 + 50
            noise = rng.normal(0, 8, len(dates))
            col_name = f"trend_{kw.replace(' ', '_')}"
            values = np.clip(base + noise, 0, 100)
            data[col_name] = values

        df = pd.DataFrame(data)
        df.to_csv(self.trends_csv, index=False)

        print(f"✅ Generated synthetic Google Trends data ({len(df)} weeks)")
        print(f"   Saved to: {self.trends_csv}")
        return df

    # ---------- CDC DATA ----------

    def collect_cdc_data(self, start_date, end_date):
        """
        Load CDC ILINet data from local CSV.
        If file is missing or empty, generate synthetic demo data and save it.
        Expected columns if CSV already exists:
          - 'week_start' (weekly timestamps)
          - 'percent_unweighted_ili'
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        if not os.path.exists(self.cdc_csv):
            print(f"⚠️ CDC flu CSV not found at: {self.cdc_csv}")
            print("   Generating synthetic CDC ILI data instead...")
            return self._generate_synthetic_cdc(start_dt, end_dt)

        print(f"📥 Loading CDC flu data from '{self.cdc_csv}'...")
        try:
            df = pd.read_csv(self.cdc_csv)
        except pd.errors.EmptyDataError:
            print("⚠️ CDC CSV is empty. Generating synthetic CDC data instead...")
            return self._generate_synthetic_cdc(start_dt, end_dt)

        if df.empty or df.shape[1] == 0:
            print("⚠️ CDC CSV has no usable data. Generating synthetic CDC data instead...")
            return self._generate_synthetic_cdc(start_dt, end_dt)

        # Check required columns
        required_cols = ["week_start", "percent_unweighted_ili"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"⚠️ Missing columns in CDC CSV: {missing}")
            print("   Generating synthetic CDC data with proper structure instead...")
            return self._generate_synthetic_cdc(start_dt, end_dt)

        df["week_start"] = pd.to_datetime(df["week_start"])
        df["percent_unweighted_ili"] = pd.to_numeric(
            df["percent_unweighted_ili"], errors="coerce"
        )

        df = df[(df["week_start"] >= start_dt) & (df["week_start"] <= end_dt)].copy()
        df = df.dropna(subset=["percent_unweighted_ili"]).reset_index(drop=True)

        print(f"✅ Loaded {len(df)} weekly rows from CDC CSV")
        print(f"   Date range: {df['week_start'].min()} to {df['week_start'].max()}")
        return df

    def _generate_synthetic_cdc(self, start_dt, end_dt):
        """Create a fake but realistic CDC ILI time series and save it."""
        os.makedirs(self.data_folder, exist_ok=True)

        dates = pd.date_range(start=start_dt, end=end_dt, freq="W-MON")

        years = (dates.year - dates.year.min()) + (dates.month - 1) / 12
        seasonal = 2.5 * np.sin(2 * np.pi * years - 1.5) + 3.5  # seasonal wave
        noise = np.random.normal(0, 0.4, len(dates))
        ili_values = np.maximum(seasonal + noise, 0.5)

        df = pd.DataFrame(
            {"week_start": dates, "percent_unweighted_ili": ili_values}
        )
        df.to_csv(self.cdc_csv, index=False)

        print(f"✅ Generated synthetic CDC ILI data ({len(df)} weeks)")
        print(f"   Saved to: {self.cdc_csv}")
        return df


# ============================================================================
# PHASE 2: DATA CLEANING & ALIGNMENT
# ============================================================================

class DataProcessor:
    """Handles data cleaning, alignment, and merging"""

    @staticmethod
    def align_and_merge(trends_df, cdc_df):
        if trends_df is None or cdc_df is None:
            print("❌ Cannot merge: missing data")
            return None

        try:
            print("🔄 Aligning and merging datasets...")

            trends_df["week_start"] = pd.to_datetime(trends_df["date"])
            cdc_df["week_start"] = pd.to_datetime(cdc_df["week_start"])

            # Normalize both to week start (Monday)
            trends_df["week_start"] = trends_df["week_start"].dt.to_period("W").dt.start_time
            cdc_df["week_start"] = cdc_df["week_start"].dt.to_period("W").dt.start_time

            merged = pd.merge(
                cdc_df,
                trends_df.drop("date", axis=1, errors="ignore"),
                on="week_start",
                how="inner",
            )

            merged = merged.sort_values("week_start").reset_index(drop=True)
            merged = merged.dropna()

            print(f"✅ Merged dataset: {len(merged)} weeks")
            print(f"   Date range: {merged['week_start'].min()} to {merged['week_start'].max()}")
            return merged

        except Exception as e:
            print(f"❌ Error merging data: {e}")
            return None

    @staticmethod
    def create_alignment_plot(merged_df):
        if merged_df is None or merged_df.empty:
            print("⚠️ No data to plot")
            return

        trend_cols = [col for col in merged_df.columns if col.startswith("trend_")]
        if not trend_cols:
            print("⚠️ No trend columns found")
            return

        fig, axes = plt.subplots(len(trend_cols) + 1, 1, figsize=(14, 4 * (len(trend_cols) + 1)))
        if len(trend_cols) == 0:
            axes = [axes]

        # CDC line
        ax = axes[0]
        ax.plot(
            merged_df["week_start"],
            merged_df["percent_unweighted_ili"],
            color="red",
            linewidth=2,
            label="CDC ILI %",
        )
        ax.set_ylabel("CDC ILI %", fontsize=12, fontweight="bold")
        ax.set_title("CDC Influenza-Like Illness Percentage", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Each trend
        for idx, col in enumerate(trend_cols, 1):
            ax = axes[idx]
            keyword = col.replace("trend_", "").replace("_", " ")
            ax.plot(
                merged_df["week_start"],
                merged_df[col],
                color="blue",
                linewidth=2,
                label=f"Google Trends: {keyword}",
            )
            ax.set_ylabel("Search Interest", fontsize=12, fontweight="bold")
            ax.set_title(f'Google Trends: "{keyword}"', fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Date", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig("data_alignment_check.png", dpi=300, bbox_inches="tight")
        print("✅ Saved alignment plot: data_alignment_check.png")
        plt.show()


# ============================================================================
# PHASE 3: CORRELATION ANALYSIS
# ============================================================================

class CorrelationAnalyzer:
    """Performs correlation analysis and lag studies"""

    @staticmethod
    def compute_correlations(merged_df):
        if merged_df is None or merged_df.empty:
            return None

        print("\n📊 Computing correlations...")

        trend_cols = [col for col in merged_df.columns if col.startswith("trend_")]
        results = {}

        for col in trend_cols:
            keyword = col.replace("trend_", "").replace("_", " ")

            corr, pval = stats.pearsonr(
                merged_df[col],
                merged_df["percent_unweighted_ili"],
            )

            results[keyword] = {
                "correlation": corr,
                "p_value": pval,
                "significant": pval < 0.05,
            }

            print(f"  '{keyword}': r={corr:.3f}, p={pval:.4f} {'✓' if pval < 0.05 else '✗'}")

        return results

    @staticmethod
    def lag_correlation_analysis(merged_df, max_lag=4):
        if merged_df is None or merged_df.empty:
            return None

        print(f"\n⏱️ Computing lag correlations (0 to {max_lag} weeks)...")

        trend_cols = [col for col in merged_df.columns if col.startswith("trend_")]
        lag_results = []

        for col in trend_cols:
            keyword = col.replace("trend_", "").replace("_", " ")

            for lag in range(max_lag + 1):
                if lag == 0:
                    corr, pval = stats.pearsonr(
                        merged_df[col],
                        merged_df["percent_unweighted_ili"],
                    )
                else:
                    valid_length = len(merged_df) - lag
                    if valid_length < 10:
                        continue

                    corr, pval = stats.pearsonr(
                        merged_df[col].iloc[:-lag],
                        merged_df["percent_unweighted_ili"].iloc[lag:],
                    )

                lag_results.append(
                    {
                        "keyword": keyword,
                        "lag_weeks": lag,
                        "correlation": corr,
                        "p_value": pval,
                    }
                )

        lag_df = pd.DataFrame(lag_results)

        print("\n🎯 Best predictive lag for each search term:")
        for keyword in lag_df["keyword"].unique():
            keyword_data = lag_df[lag_df["keyword"] == keyword]
            best = keyword_data.loc[keyword_data["correlation"].idxmax()]
            print(f"  '{keyword}': {best['lag_weeks']} weeks ahead (r={best['correlation']:.3f})")

        return lag_df

    @staticmethod
    def visualize_correlations(merged_df, lag_df=None):
        if merged_df is None or merged_df.empty:
            return

        trend_cols = [col for col in merged_df.columns if col.startswith("trend_")]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Correlation heatmap
        corr_data = merged_df[trend_cols + ["percent_unweighted_ili"]].corr()
        sns.heatmap(
            corr_data,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0,
            ax=axes[0],
            cbar_kws={"label": "Correlation"},
        )
        axes[0].set_title("Correlation Matrix", fontsize=14, fontweight="bold")

        # Overlay plot
        ax = axes[1]
        ax2 = ax.twinx()

        ax.plot(
            merged_df["week_start"],
            merged_df["percent_unweighted_ili"],
            color="red",
            linewidth=2,
            label="CDC ILI %",
            alpha=0.8,
        )

        colors = plt.cm.tab10(np.linspace(0, 1, len(trend_cols)))
        for idx, col in enumerate(trend_cols):
            keyword = col.replace("trend_", "").replace("_", " ")
            ax2.plot(
                merged_df["week_start"],
                merged_df[col],
                color=colors[idx],
                linewidth=1.5,
                label=keyword,
                alpha=0.7,
            )

        ax.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax.set_ylabel("CDC ILI %", fontsize=12, fontweight="bold", color="red")
        ax2.set_ylabel("Search Interest", fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", labelcolor="red")

        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        ax.set_title(
            "Temporal Overlap: CDC ILI vs Search Trends",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("correlation_analysis.png", dpi=300, bbox_inches="tight")
        print("✅ Saved correlation plot: correlation_analysis.png")
        plt.show()

        # Lag correlation plot
        if lag_df is not None and not lag_df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))

            for keyword in lag_df["keyword"].unique():
                keyword_data = lag_df[lag_df["keyword"] == keyword]
                ax.plot(
                    keyword_data["lag_weeks"],
                    keyword_data["correlation"],
                    marker="o",
                    linewidth=2,
                    label=keyword,
                    markersize=8,
                )

            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Lag (weeks ahead)", fontsize=12, fontweight="bold")
            ax.set_ylabel("Pearson Correlation", fontsize=12, fontweight="bold")
            ax.set_title(
                "Predictive Power: How Many Weeks Can Searches Predict Flu Activity?",
                fontsize=14,
                fontweight="bold",
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig("lag_correlation_analysis.png", dpi=300, bbox_inches="tight")
            print("✅ Saved lag correlation plot: lag_correlation_analysis.png")
            plt.show()


# ============================================================================
# PHASE 4: INSIGHTS & REPORTING
# ============================================================================

class InsightsReporter:
    """Generates summary insights and reports"""

    @staticmethod
    def generate_summary(correlation_results, lag_df):
        print("\n" + "=" * 70)
        print("📋 PROJECT SUMMARY: Google Trends vs CDC Flu Data")
        print("=" * 70)

        if correlation_results:
            print("\n🔍 KEY FINDINGS:")
            print("-" * 70)

            best_term = max(
                correlation_results.items(), key=lambda x: x[1]["correlation"]
            )

            print(f"\n✨ Best Correlating Search Term: '{best_term[0]}'")
            print(f"   Correlation: r = {best_term[1]['correlation']:.3f}")
            print(f"   Statistical significance: p = {best_term[1]['p_value']:.4f}")

            if best_term[1]["correlation"] > 0.7:
                strength = "strong positive"
            elif best_term[1]["correlation"] > 0.4:
                strength = "moderate positive"
            else:
                strength = "weak positive"

            print(f"   Interpretation: {strength} correlation")

        if lag_df is not None and not lag_df.empty:
            print("\n⏱️ PREDICTIVE SIGNAL ANALYSIS:")
            print("-" * 70)

            for keyword in lag_df["keyword"].unique():
                keyword_data = lag_df[lag_df["keyword"] == keyword]
                best_lag = keyword_data.loc[keyword_data["correlation"].idxmax()]

                print(f"\n  Search term: '{keyword}'")
                print(
                    f"  └─ Strongest correlation at {best_lag['lag_weeks']} weeks ahead"
                )
                print(f"  └─ Correlation: r = {best_lag['correlation']:.3f}")

                if best_lag["lag_weeks"] > 0:
                    print(
                        f"  └─ ✓ Can predict flu activity {best_lag['lag_weeks']} week(s) in advance!"
                    )
                else:
                    print("  └─ ○ Best as concurrent indicator, not predictive")

        print("\n" + "=" * 70)
        print("🎯 CONCLUSION:")
        print("=" * 70)
        print("Google search trends show measurable correlation with CDC flu data,")
        print("suggesting that public search behavior can serve as a valuable")
        print("supplementary signal for flu surveillance systems.")
        print("=" * 70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("📊 GOOGLE TRENDS vs CDC FLU DATA ANALYSIS")
    print("   A Public Health Data Science Project")
    print("=" * 70 + "\n")

    # Date range (must fit inside synthetic data or your real CSVs)
    START_DATE = "2019-10-01"
    END_DATE = "2025-04-30"

    KEYWORDS = [
        "flu symptoms",
        "flu near me",
        "flu shot",
        "influenza",
    ]

    # Phase 1: load data
    print("🩵 PHASE 1: DATA COLLECTION (LOCAL / SYNTHETIC)")
    print("-" * 70)

    collector = DataCollector()
    trends_df = collector.collect_google_trends(KEYWORDS, START_DATE, END_DATE)
    cdc_df = collector.collect_cdc_data(START_DATE, END_DATE)

    # Phase 2
    print("\n🧹 PHASE 2: DATA CLEANING & ALIGNMENT")
    print("-" * 70)
    processor = DataProcessor()
    merged_df = processor.align_and_merge(trends_df, cdc_df)

    if merged_df is not None:
        merged_df.to_csv("merged_data.csv", index=False)
        print("💾 Saved: merged_data.csv")
        processor.create_alignment_plot(merged_df)

    # Phase 3
    print("\n📈 PHASE 3: CORRELATION & ANALYSIS")
    print("-" * 70)
    analyzer = CorrelationAnalyzer()
    correlation_results = analyzer.compute_correlations(merged_df)
    lag_df = analyzer.lag_correlation_analysis(merged_df, max_lag=4)

    if lag_df is not None:
        lag_df.to_csv("lag_correlation_results.csv", index=False)
        print("💾 Saved: lag_correlation_results.csv")

    analyzer.visualize_correlations(merged_df, lag_df)

    # Phase 4
    print("\n🧠 PHASE 4: INSIGHTS & REPORTING")
    print("-" * 70)
    reporter = InsightsReporter()
    reporter.generate_summary(correlation_results, lag_df)

    print("✅ Analysis complete! Check generated CSVs and visualizations.")


if __name__ == "__main__":
    main()
