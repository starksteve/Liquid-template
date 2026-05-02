import pandas as pd
import numpy as np
from datetime import datetime


class DataProfiler:
    """
    Deep pandas-based data profiler.
    Produces per-column statistics and an overall data quality score.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(self, df: pd.DataFrame) -> dict:
        """
        Profile a DataFrame and return a structured report.

        Returns:
            dict with keys:
                - 'summary': dict (row_count, col_count, total_nulls, quality_score, generated_at)
                - 'columns': dict  col_name -> column_profile
                - 'recommendations': list[str]
        """
        report = {
            "summary": self._summary(df),
            "columns": {col: self._profile_column(df[col]) for col in df.columns},
            "recommendations": [],
        }
        report["recommendations"] = self._recommendations(df, report["columns"])
        return report

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _summary(self, df: pd.DataFrame) -> dict:
        total_cells = df.shape[0] * df.shape[1]
        total_nulls = int(df.isnull().sum().sum())
        completeness = round((1 - total_nulls / total_cells) * 100, 1) if total_cells > 0 else 100.0

        # Duplicate row detection
        duplicate_rows = int(df.duplicated().sum())

        # Quality score: weighted avg of completeness (60%) + uniqueness proxy (20%) + type consistency (20%)
        dup_penalty = min(30, round((duplicate_rows / len(df)) * 30, 1)) if len(df) > 0 else 0
        quality_score = max(0, round(completeness * 0.7 - dup_penalty * 0.3, 1))

        return {
            "row_count": len(df),
            "col_count": len(df.columns),
            "total_cells": total_cells,
            "total_nulls": total_nulls,
            "completeness_pct": completeness,
            "duplicate_rows": duplicate_rows,
            "duplicate_pct": round(duplicate_rows / len(df) * 100, 2) if len(df) > 0 else 0,
            "quality_score": quality_score,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    # ------------------------------------------------------------------
    # Per-column profile
    # ------------------------------------------------------------------

    def _profile_column(self, series: pd.Series) -> dict:
        base = {
            "dtype": str(series.dtype),
            "null_count": int(series.isnull().sum()),
            "null_pct": round(series.isnull().mean() * 100, 2),
            "unique_count": int(series.nunique(dropna=True)),
            "unique_pct": round(series.nunique(dropna=True) / len(series) * 100, 2) if len(series) > 0 else 0,
            "fill_rate_pct": round(series.notnull().mean() * 100, 2),
            "sample_values": [str(v) for v in series.dropna().head(5).tolist()],
            "inferred_type": self._infer_type(series),
        }

        if pd.api.types.is_numeric_dtype(series):
            base.update(self._numeric_stats(series))
        elif self._is_date_column(series):
            base.update(self._date_stats(series))
        else:
            base.update(self._string_stats(series))

        return base

    # ------------------------------------------------------------------
    # Type-specific stats
    # ------------------------------------------------------------------

    def _numeric_stats(self, series: pd.Series) -> dict:
        clean = series.dropna()
        if len(clean) == 0:
            return {"col_category": "numeric"}
        return {
            "col_category": "numeric",
            "min": float(clean.min()),
            "max": float(clean.max()),
            "mean": round(float(clean.mean()), 4),
            "median": float(clean.median()),
            "std": round(float(clean.std()), 4),
            "q25": float(clean.quantile(0.25)),
            "q75": float(clean.quantile(0.75)),
            "zero_count": int((clean == 0).sum()),
            "negative_count": int((clean < 0).sum()),
            "outlier_count": self._count_outliers(clean),
        }

    def _string_stats(self, series: pd.Series) -> dict:
        clean = series.dropna().astype(str)
        if len(clean) == 0:
            return {"col_category": "string"}
        lengths = clean.str.len()
        top_vals = series.value_counts().head(10).to_dict()
        return {
            "col_category": "string",
            "min_length": int(lengths.min()),
            "max_length": int(lengths.max()),
            "avg_length": round(float(lengths.mean()), 1),
            "blank_count": int((clean.str.strip() == "").sum()),
            "top_values": {str(k): int(v) for k, v in top_vals.items()},
            "cardinality": "high" if series.nunique() / max(len(series), 1) > 0.5 else "low",
            "looks_like_id": series.nunique() == len(series.dropna()),
            "pattern_sample": self._detect_pattern(clean),
        }

    def _date_stats(self, series: pd.Series) -> dict:
        try:
            parsed = pd.to_datetime(series, errors="coerce")
            clean = parsed.dropna()
            if len(clean) == 0:
                return {"col_category": "date"}
            return {
                "col_category": "date",
                "min_date": str(clean.min().date()),
                "max_date": str(clean.max().date()),
                "date_range_days": (clean.max() - clean.min()).days,
                "parse_fail_count": int(parsed.isnull().sum() - series.isnull().sum()),
            }
        except Exception:
            return {"col_category": "date"}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_type(self, series: pd.Series) -> str:
        if pd.api.types.is_integer_dtype(series):
            return "integer"
        if pd.api.types.is_float_dtype(series):
            return "float"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if self._is_date_column(series):
            return "date"
        # Check if strings look numeric
        clean = series.dropna().astype(str).str.strip()
        if len(clean) > 0:
            numeric_ratio = clean.str.match(r"^-?\d+(\.\d+)?$").mean()
            if numeric_ratio > 0.9:
                return "numeric_string"
        return "string"

    def _is_date_column(self, series: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if series.dtype == object:
            sample = series.dropna().astype(str).head(20)
            if len(sample) == 0:
                return False
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                return parsed.notnull().mean() > 0.8
            except Exception:
                return False
        return False

    def _count_outliers(self, series: pd.Series) -> int:
        """IQR-based outlier count."""
        try:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            return int(((series < lower) | (series > upper)).sum())
        except Exception:
            return 0

    def _detect_pattern(self, series: pd.Series) -> str:
        """Return a human-readable pattern hint based on sample values."""
        sample = series.head(5).tolist()
        # Check common patterns
        checks = {
            "email": r"^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$",
            "phone": r"^\+?[\d\s\-().]{7,15}$",
            "date_yyyymmdd": r"^\d{4}-\d{2}-\d{2}$",
            "zip_code": r"^\d{5}(-\d{4})?$",
            "id_numeric": r"^\d+$",
            "yes_no": r"^(yes|no|y|n|true|false|1|0)$",
        }
        import re
        for pattern_name, regex in checks.items():
            match_count = sum(1 for v in sample if re.match(regex, str(v), re.IGNORECASE))
            if match_count >= min(3, len(sample)):
                return pattern_name
        return "freeform_text"

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _recommendations(self, df: pd.DataFrame, col_profiles: dict) -> list[str]:
        recs = []
        for col, p in col_profiles.items():
            null_pct = p.get("null_pct", 0)
            if null_pct > 50:
                recs.append(f"⚠️ **{col}**: {null_pct}% nulls — consider dropping or imputing this column.")
            elif null_pct > 20:
                recs.append(f"🔶 **{col}**: {null_pct}% nulls — investigate data completeness.")

            if p.get("inferred_type") == "numeric_string":
                recs.append(f"🔵 **{col}**: Stored as string but looks numeric — consider `CAST({col} AS DOUBLE)`.")

            if p.get("unique_pct", 0) == 100 and len(df) > 5:
                recs.append(f"🔑 **{col}**: All values are unique — likely a primary key or ID column.")

            if p.get("cardinality") == "low" and p.get("col_category") == "string":
                top = p.get("top_values", {})
                if len(top) <= 10:
                    recs.append(
                        f"📋 **{col}**: Low cardinality ({p.get('unique_count')} distinct values) — "
                        f"good candidate for `CASE WHEN` mapping or crosswalk join."
                    )

            if p.get("outlier_count", 0) > 0 and p.get("col_category") == "numeric":
                recs.append(
                    f"📊 **{col}**: {p['outlier_count']} statistical outliers detected (IQR method) — "
                    f"review before aggregation."
                )

            if p.get("looks_like_id") and p.get("col_category") == "string":
                recs.append(f"🔑 **{col}**: Every value is unique — likely an identifier column.")

        if df.duplicated().sum() > 0:
            recs.append(
                f"🔁 **Duplicate rows**: {df.duplicated().sum()} duplicate rows detected — "
                f"consider deduplication with `dropDuplicates()` in Spark."
            )

        if not recs:
            recs.append("✅ No major data quality issues detected.")

        return recs
