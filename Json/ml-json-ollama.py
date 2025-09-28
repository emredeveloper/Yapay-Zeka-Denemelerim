import csv
import json
import re

import ollama
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import joblib
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
except Exception:
    Console = None
    Table = None
    box = None


MODEL_NAME = "llama3.2:3b"
CSV_PATH = "apple_quality.csv"
OUTPUT_PATH = "apple_quality_structured.jsonl"
ROW_LIMIT = 10
MODEL_OUT_PATH = "apple_quality_model.pkl"


def _strip_fences(text: str) -> str:
    if not text:
        return text
    m = re.match(r"^```(?:json)?\n([\s\S]*?)\n```\s*$", text.strip(), re.IGNORECASE)
    return m.group(1) if m else text


def _safe_parse(text: str) -> dict:
    raw = _strip_fences(text or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {"raw": raw}


def _normalize(obj: dict) -> dict:
    if not isinstance(obj, dict):
        return {"raw": str(obj)}
    record_summary = str(obj.get("record_summary") or "").strip()
    target = obj.get("target")
    if isinstance(target, str) and target.strip().lower() in {"", "null", "none"}:
        target = None
    def _kw(v):
        items = v if isinstance(v, list) else [v] if isinstance(v, str) else []
        cleaned = []
        seen = set()
        for x in items:
            if not isinstance(x, str):
                continue
            y = re.sub(r"[^a-z0-9\s]", "", x.lower()).strip()
            if y and y not in seen:
                seen.add(y)
                cleaned.append(y)
        return cleaned[:5]
    return {
        "record_summary": record_summary or None,
        "target": target,
        "features": _kw(obj.get("features")),
        "anomalies": _kw(obj.get("anomalies")),
    }


def _build_record_summary(row: dict) -> str:
    """Deterministically build a short summary directly from the CSV row values."""
    if not isinstance(row, dict):
        return ""
    label_like = [k for k in row.keys() if str(k).lower() in {"quality", "target", "label", "class"}]
    numeric_keys = []
    for k, v in row.items():
        try:
            float(v)
            numeric_keys.append(k)
        except (TypeError, ValueError):
            continue
    parts = []
    # Prefer label first
    for k in label_like[:1]:
        v = row.get(k)
        parts.append(f"{k}={v}")
    # Then top 2 numeric fields
    for k in [x for x in numeric_keys if x not in label_like][:2]:
        v = row.get(k)
        try:
            v_num = float(v)
            v_fmt = f"{v_num:.3f}"
        except (TypeError, ValueError):
            v_fmt = str(v)
        parts.append(f"{k}={v_fmt}")
    # If still short, add one short text field
    if len(parts) < 3:
        for k, v in row.items():
            if k in label_like or k in numeric_keys:
                continue
            s = str(v).strip()
            if s:
                parts.append(f"{k}={s[:20]}")
                break
    return ", ".join(parts[:3])


def call_model(row: dict) -> dict:
    system = (
        "Only valid JSON. Language: English. Schema: "
        '{"record_summary":"string","target":"string|null","features":["string"],"anomalies":["string"]}. '
        "Guidelines: record_summary must reference 2-3 important field names with their values (e.g., 'size=7.2, weight=150g'); avoid generic phrases like 'This is a summary of the row'. "
        "Choose target from label-like columns if present (e.g., 'quality', 'is_apple', 'apple_type'); otherwise null. "
        "features should be concise, lowercase column names (max 5) most relevant to the row. "
        "anomalies lists any suspicious or out-of-range fields; empty if none."
    )
    user = (
        "Given this CSV row as a JSON object, produce the schema above. "
        "Infer target if a label-like field exists; otherwise null. Row: "
        + json.dumps(row, ensure_ascii=False)
    )
    res = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        format="json",
        options={"temperature": 0.2, "seed": 42},
    )
    result = _normalize(_safe_parse(res["message"]["content"]))
    # Override summary with deterministic one from raw CSV to avoid meaningless text
    result["record_summary"] = _build_record_summary(row)
    return result


def plan_model_with_llm(df: 'pd.DataFrame') -> dict:
    """Ask LLM to propose a training plan given dataframe dtypes and sample rows."""
    schema = {c: str(t) for c, t in df.dtypes.items()}
    sample = df.head(5).to_dict(orient="records")
    system = (
        "Only valid JSON. Propose an ML training plan for the given dataset. "
        "Schema: {\"task\":\"classification|regression\",\"target\":\"string\",\"features\":[\"string\"],\"models\":[\"RandomForest|LogisticRegression|LinearRegression\"],\"notes\":\"string\"}"
    )
    user = (
        "Given the dataframe schema and 5 sample rows, propose a plan. "
        + json.dumps({"schema": schema, "sample": sample}, ensure_ascii=False)
    )
    res = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        format="json",
        options={"temperature": 0.2, "seed": 7},
    )
    plan = _safe_parse(res["message"]["content"]) or {}
    # minimal defaults
    if plan.get("task") not in {"classification", "regression"}:
        # heuristic: numeric target -> regression else classification
        # guess a target: a column named 'quality' or last column
        target_guess = next((c for c in df.columns if c.lower() in {"quality", "target", "label"}), df.columns[-1])
        if str(df[target_guess].dtype).startswith("float") or str(df[target_guess].dtype).startswith("int"):
            task = "regression"
        else:
            task = "classification"
        plan = {
            "task": task,
            "target": target_guess,
            "features": [c for c in df.columns if c != target_guess][:10],
            "models": ["RandomForest"],
            "notes": "Default fallback plan",
        }
    return plan


def train_with_plan(df: 'pd.DataFrame', plan: dict) -> dict:
    target = plan.get("target")
    features = plan.get("features") or [c for c in df.columns if c != target]
    task = plan.get("task", "classification")
    models = plan.get("models", ["RandomForest"])  # list of names

    if target not in df.columns:
        raise ValueError("Target column not found in dataframe")

    X = df[features].copy()
    y = df[target].copy()

    # Replace inf with NaN, drop rows with NaN target
    X = X.replace([np.inf, -np.inf], np.nan)
    # y may be numeric or categorical
    try:
        y = y.replace([np.inf, -np.inf], np.nan)
    except Exception:
        pass
    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    # split numeric/categorical
    num_cols = [c for c in X.columns if str(df[c].dtype).startswith(("float", "int"))]
    cat_cols = [c for c in X.columns if c not in num_cols]

    transformers = []
    if num_cols:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))
    pre = ColumnTransformer(transformers)

    # If after filtering there are too few samples, abort gracefully
    if len(y) < 3:
        raise ValueError("Not enough non-null samples after filtering for training")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if task == "classification" else None
    )

    chosen = (models[0] if isinstance(models, list) and models else "RandomForest").lower()

    if task == "classification":
        if "logistic" in chosen:
            estimator = LogisticRegression(max_iter=1000)
        else:
            estimator = RandomForestClassifier(n_estimators=200, random_state=42)
        pipe = Pipeline([("pre", pre), ("clf", estimator)])
        # Ensure at least 2 classes exist
        if len(pd.Series(y_train).unique()) < 2:
            raise ValueError("Classification task requires at least two classes in training split")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        # Feature importances / coefficients
        fi = []
        top_features_serializable = []
        try:
            pre_fitted = pipe.named_steps["pre"]
            input_feats = list(X.columns)
            try:
                feat_names = list(pre_fitted.get_feature_names_out(input_feats))
            except Exception:
                feat_names = input_feats
            model = pipe.named_steps["clf"]
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                fi = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)[:15]
            elif hasattr(model, "coef_") and getattr(model, "coef_", None) is not None:
                coefs = model.coef_[0]
                fi = sorted(zip(feat_names, coefs), key=lambda x: abs(x[1]), reverse=True)[:15]
            top_features_serializable = [{"name": n, "weight": float(v)} for n, v in fi[:10]]
        except Exception:
            fi = []
            top_features_serializable = []

        # Decisions preview (sample)
        sample_decisions = []
        if Console and Table:
            console = Console()
            table_pred = Table(title="Model Decisions (sample)", box=box.SIMPLE_HEAVY)
            table_pred.add_column("#", justify="right")
            table_pred.add_column("Actual")
            table_pred.add_column("Pred")
            table_pred.add_column("Prob", justify="right")
            try:
                proba = pipe.predict_proba(X_test)
                for i in range(min(5, len(y_test))):
                    p = preds[i]
                    cls_index = list(pipe.named_steps["clf"].classes_).index(p)
                    pr = float(proba[i][cls_index])
                    sample_decisions.append({"actual": str(y_test.iloc[i]), "pred": str(p), "prob": pr})
                    table_pred.add_row(str(i), str(y_test.iloc[i]), str(p), f"{pr:.3f}")
            except Exception:
                for i in range(min(5, len(y_test))):
                    sample_decisions.append({"actual": str(y_test.iloc[i]), "pred": str(preds[i])})
                    table_pred.add_row(str(i), str(y_test.iloc[i]), str(preds[i]), "-")

            console.print("\n[bold green]Training Metrics[/bold green]")
            console.print(f"Accuracy: {acc:.4f}  |  F1: {f1:.4f}")
            if fi:
                table_imp = Table(title="Top Features", box=box.SIMPLE_HEAVY)
                table_imp.add_column("Feature")
                table_imp.add_column("Importance", justify="right")
                for name, imp in fi:
                    table_imp.add_row(str(name), f"{float(imp):.4f}")
                console.print(table_imp)
            console.print(table_pred)

        return {
            "task": task,
            "target": target,
            "features": features,
            "models": models,
            "accuracy": acc,
            "f1": f1,
            "top_features": top_features_serializable,
            "sample_decisions": sample_decisions,
            "pipeline": pipe,
        }

    else:  # regression
        if "linear" in chosen:
            estimator = LinearRegression()
        else:
            estimator = RandomForestRegressor(n_estimators=300, random_state=42)
        pipe = Pipeline([("pre", pre), ("reg", estimator)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        # Feature importances / coefficients
        fi = []
        top_features_serializable = []
        try:
            pre_fitted = pipe.named_steps["pre"]
            input_feats = list(X.columns)
            try:
                feat_names = list(pre_fitted.get_feature_names_out(input_feats))
            except Exception:
                feat_names = input_feats
            model = pipe.named_steps["reg"]
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                fi = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)[:15]
            elif hasattr(model, "coef_") and getattr(model, "coef_", None) is not None:
                coefs = model.coef_
                if hasattr(coefs, "tolist"):
                    coefs = coefs.tolist()
                if isinstance(coefs, list):
                    fi = sorted(zip(feat_names, coefs), key=lambda x: abs(x[1]), reverse=True)[:15]
            top_features_serializable = [{"name": n, "weight": float(v)} for n, v in fi[:10]]
        except Exception:
            fi = []
            top_features_serializable = []

        # Predictions preview
        sample_predictions = []
        if Console and Table:
            console = Console()
            table_pred = Table(title="Model Predictions (sample)", box=box.SIMPLE_HEAVY)
            table_pred.add_column("#", justify="right")
            table_pred.add_column("Actual", justify="right")
            table_pred.add_column("Pred", justify="right")
            for i in range(min(5, len(y_test))):
                sample_predictions.append({"actual": float(y_test.iloc[i]), "pred": float(preds[i])})
                table_pred.add_row(str(i), f"{float(y_test.iloc[i])}", f"{float(preds[i]):.3f}")
            console.print("\n[bold green]Training Metrics[/bold green]")
            console.print(f"R2: {r2:.4f}  |  MAE: {mae:.4f}")
            if fi:
                table_imp = Table(title="Top Features", box=box.SIMPLE_HEAVY)
                table_imp.add_column("Feature")
                table_imp.add_column("Importance", justify="right")
                for name, imp in fi:
                    table_imp.add_row(str(name), f"{float(imp):.4f}")
                console.print(table_imp)
            console.print(table_pred)

        return {
            "task": task,
            "target": target,
            "features": features,
            "models": models,
            "r2": r2,
            "mae": mae,
            "top_features": top_features_serializable,
            "sample_predictions": sample_predictions,
            "pipeline": pipe,
        }


def main() -> None:
    count = 0
    with open(CSV_PATH, "r", encoding="utf-8") as f, open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        reader = csv.DictReader(f)
        for row in reader:
            obj = call_model(row)
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
            if count >= ROW_LIMIT:
                break
    print(f"Wrote {count} JSON lines to {OUTPUT_PATH}")

    # Optional: simple preview in console as a compact table-like view (first 5 lines)
    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as preview:
            print("\nPreview (first 5 lines):")
            for i, line in enumerate(preview):
                if i >= 5:
                    break
                rec = json.loads(line)
                print(f"- summary: {rec.get('record_summary')} | target: {rec.get('target')} | features: {', '.join(rec.get('features') or [])}")
    except Exception:
        pass

    # Stage 2: Auto plan and train a model from CSV
    try:
        df = pd.read_csv(CSV_PATH)
        plan = plan_model_with_llm(df)
        report = train_with_plan(df, plan)
        joblib.dump(report["pipeline"], MODEL_OUT_PATH)
        print(f"\nModel trained and saved to {MODEL_OUT_PATH}")
        print(json.dumps({k: v for k, v in report.items() if k != "pipeline"}, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Training stage skipped due to error: {e}")


if __name__ == "__main__":
    main()


