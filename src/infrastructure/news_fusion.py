from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.domain.entities import FusionDiagnostics, NewsItem, SentimentAggregate
from src.domain.news import aggregate_sentiment, blend_probability
from src.infrastructure.modeling import LogisticBinaryModel, binary_metrics


NEWS_FEATURE_COLUMNS = [
    "sent_score",
    "sent_bullish",
    "sent_bearish",
    "sent_neutral",
    "sent_items_log",
    "sent_abs_score",
    "sent_signed_items",
]

FUSION_FEATURE_COLUMNS = [
    "q_logit",
    "n_logit",
    "q_minus_n",
]


@dataclass(frozen=True)
class LearnedFusionPrediction:
    final_prob: float
    news_prob: float
    sentiment: SentimentAggregate
    mode: str
    diagnostics: FusionDiagnostics


def _clip_prob(p: float) -> float:
    return float(np.clip(p, 1e-6, 1.0 - 1e-6))


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


def _sentiment_features(sent: SentimentAggregate) -> dict[str, float]:
    score = float(np.clip(sent.score, -1.0, 1.0))
    items_log = float(np.log1p(max(0, int(sent.items))))
    return {
        "sent_score": score,
        "sent_bullish": float(np.clip(sent.bullish, 0.0, 1.0)),
        "sent_bearish": float(np.clip(sent.bearish, 0.0, 1.0)),
        "sent_neutral": float(np.clip(sent.neutral, 0.0, 1.0)),
        "sent_items_log": items_log,
        "sent_abs_score": float(abs(score)),
        "sent_signed_items": float(score * items_log),
    }


def _fallback_news_prob(sent: SentimentAggregate) -> float:
    return _clip_prob(0.5 + 0.5 * float(np.clip(sent.score, -1.0, 1.0)))


def _rule_diagnostics(
    target: str,
    horizon: str,
    reason: str,
    samples: int = 0,
) -> FusionDiagnostics:
    return FusionDiagnostics(
        target=target,
        horizon=horizon,
        mode="rule",
        reason=reason,
        samples=int(samples),
        holdout_n=0,
        holdout_accuracy=np.nan,
        holdout_brier=np.nan,
        holdout_auc=np.nan,
        news_coef_score=np.nan,
        fusion_coef_quant=np.nan,
        fusion_coef_news=np.nan,
    )


def _walkforward_quant_probs(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    min_train_days: int,
    step_days: int,
    l2: float,
) -> pd.DataFrame:
    valid = frame.dropna(subset=["date"] + feature_cols + [target_col]).copy()
    valid = valid.sort_values("date").drop_duplicates(subset=["date"])
    if valid.empty:
        return pd.DataFrame(columns=["date", "p_quant", "y"])

    dates = valid["date"].drop_duplicates().sort_values().tolist()
    if len(dates) <= int(min_train_days):
        return pd.DataFrame(columns=["date", "p_quant", "y"])

    records: list[dict[str, float | pd.Timestamp]] = []
    step_days = max(1, int(step_days))
    for i in range(int(min_train_days), len(dates), step_days):
        train_dates = dates[:i]
        test_dates = dates[i : i + step_days]
        if not test_dates:
            break
        train = valid[valid["date"].isin(train_dates)]
        test = valid[valid["date"].isin(test_dates)]
        if train.empty or test.empty:
            continue
        try:
            model = LogisticBinaryModel(l2=l2).fit(train, feature_cols=feature_cols, target_col=target_col)
            prob = model.predict_proba(test, feature_cols=feature_cols)
        except Exception:
            continue

        for d, p, y in zip(test["date"].tolist(), prob.tolist(), test[target_col].astype(float).tolist()):
            records.append({"date": pd.Timestamp(d), "p_quant": _clip_prob(float(p)), "y": float(y)})

    out = pd.DataFrame(records)
    if out.empty:
        return pd.DataFrame(columns=["date", "p_quant", "y"])
    return out.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)


def _build_news_frame(
    dates: Sequence[pd.Timestamp],
    news_items: Sequence[NewsItem],
    target: str,
    horizon: str,
    half_life_days: float,
) -> tuple[pd.DataFrame, SentimentAggregate]:
    rows: list[dict[str, float | pd.Timestamp]] = []
    latest_sent = SentimentAggregate(bullish=0.0, bearish=0.0, neutral=1.0, score=0.0, items=0)
    for d in dates:
        sent = aggregate_sentiment(
            news_items=news_items,
            as_of_date=pd.Timestamp(d),
            target=target,
            horizon=horizon,
            half_life_days=half_life_days,
        )
        latest_sent = sent
        feat = _sentiment_features(sent)
        feat["date"] = pd.Timestamp(d)
        rows.append(feat)
    frame = pd.DataFrame(rows)
    return frame, latest_sent


def _fusion_feature_frame(p_quant: Sequence[float], p_news: Sequence[float]) -> pd.DataFrame:
    q = np.asarray([_clip_prob(float(v)) for v in p_quant], dtype=float)
    n = np.asarray([_clip_prob(float(v)) for v in p_news], dtype=float)
    return pd.DataFrame(
        {
            "q_logit": _logit(q),
            "n_logit": _logit(n),
            "q_minus_n": q - n,
        }
    )


def _holdout_metrics(
    dataset: pd.DataFrame,
    news_l2: float,
    fusion_l2: float,
    holdout_ratio: float,
) -> tuple[int, float, float, float]:
    if dataset.empty:
        return 0, np.nan, np.nan, np.nan

    ratio = float(np.clip(holdout_ratio, 0.05, 0.45))
    split = int(len(dataset) * (1.0 - ratio))
    split = max(split, 40)
    if split >= len(dataset) - 10:
        return 0, np.nan, np.nan, np.nan

    train = dataset.iloc[:split].copy()
    test = dataset.iloc[split:].copy()
    if train.empty or test.empty:
        return 0, np.nan, np.nan, np.nan

    try:
        news_model = LogisticBinaryModel(l2=news_l2).fit(train, NEWS_FEATURE_COLUMNS, "y")
        p_news_train = news_model.predict_proba(train, NEWS_FEATURE_COLUMNS)
        p_news_test = news_model.predict_proba(test, NEWS_FEATURE_COLUMNS)
        fusion_train = _fusion_feature_frame(train["p_quant"].to_numpy(), p_news_train)
        fusion_train["y"] = train["y"].astype(float).to_numpy()
        fusion_model = LogisticBinaryModel(l2=fusion_l2).fit(fusion_train, FUSION_FEATURE_COLUMNS, "y")
        fusion_test = _fusion_feature_frame(test["p_quant"].to_numpy(), p_news_test)
        p_final_test = fusion_model.predict_proba(fusion_test, FUSION_FEATURE_COLUMNS)
        metrics = binary_metrics(test["y"].astype(float).to_numpy(), p_final_test)
        return int(len(test)), float(metrics.accuracy), float(metrics.brier), float(metrics.auc)
    except Exception:
        return 0, np.nan, np.nan, np.nan


def predict_with_learned_fusion(
    *,
    enabled: bool,
    base_prob: float,
    target: str,
    horizon: str,
    feature_frame: pd.DataFrame | None,
    feature_cols: list[str],
    target_col: str,
    news_items_train: Sequence[NewsItem],
    news_items_live: Sequence[NewsItem] | None,
    as_of_date: pd.Timestamp,
    half_life_days: float,
    min_train_days: int,
    step_days: int,
    quant_l2: float,
    news_l2: float,
    fusion_l2: float,
    min_samples: int,
    holdout_ratio: float,
    fallback_strength: float,
) -> LearnedFusionPrediction:
    news_items_live = news_items_train if news_items_live is None else news_items_live
    sent_latest = aggregate_sentiment(
        news_items=news_items_live,
        as_of_date=as_of_date,
        target=target,
        horizon=horizon,
        half_life_days=half_life_days,
    )
    final_rule = blend_probability(base_prob, sent_latest.score, sentiment_strength=fallback_strength)
    news_rule = _fallback_news_prob(sent_latest)

    if not enabled:
        return LearnedFusionPrediction(
            final_prob=final_rule,
            news_prob=news_rule,
            sentiment=sent_latest,
            mode="rule",
            diagnostics=_rule_diagnostics(target, horizon, reason="learning_disabled"),
        )

    if feature_frame is None or feature_frame.empty:
        return LearnedFusionPrediction(
            final_prob=final_rule,
            news_prob=news_rule,
            sentiment=sent_latest,
            mode="rule",
            diagnostics=_rule_diagnostics(target, horizon, reason="feature_frame_empty"),
        )

    try:
        quant_ds = _walkforward_quant_probs(
            frame=feature_frame,
            feature_cols=feature_cols,
            target_col=target_col,
            min_train_days=min_train_days,
            step_days=step_days,
            l2=quant_l2,
        )
        if len(quant_ds) < int(min_samples):
            return LearnedFusionPrediction(
                final_prob=final_rule,
                news_prob=news_rule,
                sentiment=sent_latest,
                mode="rule",
                diagnostics=_rule_diagnostics(target, horizon, reason="insufficient_samples", samples=len(quant_ds)),
            )

        news_frame, _ = _build_news_frame(
            dates=quant_ds["date"].tolist(),
            news_items=news_items_train,
            target=target,
            horizon=horizon,
            half_life_days=half_life_days,
        )
        dataset = quant_ds.merge(news_frame, on="date", how="inner").sort_values("date").reset_index(drop=True)
        if len(dataset) < int(min_samples):
            return LearnedFusionPrediction(
                final_prob=final_rule,
                news_prob=news_rule,
                sentiment=sent_latest,
                mode="rule",
                diagnostics=_rule_diagnostics(target, horizon, reason="insufficient_after_merge", samples=len(dataset)),
            )

        news_model = LogisticBinaryModel(l2=news_l2).fit(dataset, NEWS_FEATURE_COLUMNS, "y")
        p_news_hist = news_model.predict_proba(dataset, NEWS_FEATURE_COLUMNS)

        fusion_train = _fusion_feature_frame(dataset["p_quant"].to_numpy(), p_news_hist)
        fusion_train["y"] = dataset["y"].astype(float).to_numpy()
        fusion_model = LogisticBinaryModel(l2=fusion_l2).fit(fusion_train, FUSION_FEATURE_COLUMNS, "y")

        latest_news_feat = pd.DataFrame([_sentiment_features(sent_latest)])
        p_news_latest = float(news_model.predict_proba(latest_news_feat, NEWS_FEATURE_COLUMNS)[0])
        latest_fusion_feat = _fusion_feature_frame([base_prob], [p_news_latest])
        p_final_latest = float(fusion_model.predict_proba(latest_fusion_feat, FUSION_FEATURE_COLUMNS)[0])

        holdout_n, holdout_acc, holdout_brier, holdout_auc = _holdout_metrics(
            dataset=dataset,
            news_l2=news_l2,
            fusion_l2=fusion_l2,
            holdout_ratio=holdout_ratio,
        )

        news_coef_score = np.nan
        if news_model.coef_ is not None and NEWS_FEATURE_COLUMNS:
            idx = NEWS_FEATURE_COLUMNS.index("sent_score")
            news_coef_score = float(news_model.coef_[idx])

        fusion_coef_quant = np.nan
        fusion_coef_news = np.nan
        if fusion_model.coef_ is not None and FUSION_FEATURE_COLUMNS:
            idx_q = FUSION_FEATURE_COLUMNS.index("q_logit")
            idx_n = FUSION_FEATURE_COLUMNS.index("n_logit")
            fusion_coef_quant = float(fusion_model.coef_[idx_q])
            fusion_coef_news = float(fusion_model.coef_[idx_n])

        diag = FusionDiagnostics(
            target=target,
            horizon=horizon,
            mode="learned",
            reason="ok",
            samples=int(len(dataset)),
            holdout_n=int(holdout_n),
            holdout_accuracy=float(holdout_acc),
            holdout_brier=float(holdout_brier),
            holdout_auc=float(holdout_auc),
            news_coef_score=float(news_coef_score),
            fusion_coef_quant=float(fusion_coef_quant),
            fusion_coef_news=float(fusion_coef_news),
        )
        return LearnedFusionPrediction(
            final_prob=_clip_prob(p_final_latest),
            news_prob=_clip_prob(p_news_latest),
            sentiment=sent_latest,
            mode="learned",
            diagnostics=diag,
        )
    except Exception:
        return LearnedFusionPrediction(
            final_prob=final_rule,
            news_prob=news_rule,
            sentiment=sent_latest,
            mode="rule",
            diagnostics=_rule_diagnostics(target, horizon, reason="learning_error"),
        )
