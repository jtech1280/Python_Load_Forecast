from __future__ import annotations

import math
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import date as date_type

from dash import Dash, Input, Output, State, html, dcc
from dash import callback_context
from .layout import make_layout

BRAND_BLUE = "#0057B8"
BRAND_RED = "#D62728"
BRAND_GOLD = "#F2A900"
BRAND_PURPLE = "#7B2CBF"
BRAND_GREEN = "#00843D"
BAND_FILL = "rgba(0, 87, 184, 0.13)"
BAND_EDGE = "rgba(0, 87, 184, 0.32)"
TEMP_SENSITIVITY_STEPS = [
    ("plus_7", "Plus 7", 7, "#f7b7b7"),
    ("plus_6", "Plus 6", 6, "#f49c9c"),
    ("plus_5", "Plus 5", 5, "#ef8585"),
    ("plus_4", "Plus 4", 4, "#ec6d6d"),
    ("plus_3", "Plus 3", 3, "#e85151"),
    ("plus_2", "Plus 2", 2, "#dc3838"),
    ("plus_1", "Plus 1", 1, "#c91f1f"),
    ("baseline", "Baseline", 0, BRAND_GREEN),
    ("minus_1", "Minus 1", -1, "#233be5"),
    ("minus_2", "Minus 2", -2, "#3e55ed"),
    ("minus_3", "Minus 3", -3, "#6074f3"),
    ("minus_4", "Minus 4", -4, "#7c8cf6"),
    ("minus_5", "Minus 5", -5, "#98a5fb"),
    ("minus_6", "Minus 6", -6, "#b0bafb"),
    ("minus_7", "Minus 7", -7, "#c9cffc"),
]

OPERATOR_TABLE_COLUMNS = [
    "DT", "Forecast", "Low", "Expected", "High", "Actual",
    "Risk Code", "Caution Reason", "Confidence", "Horizon", "Weather Risk",
    "Temp", "Daily Max", "BTM Solar", "BTM Solar Loss",
    "Scenario Spread", "Scenario Max Abs", "Scenario Cap",
    "Prior Run Fcst", "Fcst Change", "Prior Temp", "Temp Change",
]


def _fmt_num(v, decimals=1):
    try:
        if pd.isna(v):
            return ""
        return f"{float(v):,.{decimals}f}"
    except Exception:
        return ""


def _metric_card(title: str, value: str, subtitle: str = ""):
    return html.Div([
        html.Div(title, style={"fontSize": "11px", "fontWeight": "700", "color": BRAND_PURPLE, "textTransform": "uppercase"}),
        html.Div(value, style={"fontSize": "22px", "fontWeight": "800", "color": BRAND_BLUE, "lineHeight": "1.15"}),
        html.Div(subtitle, style={"fontSize": "11px", "color": BRAND_GREEN}) if subtitle else None,
    ], style={"background": "white", "border": "1px solid #B7D7F5", "borderRadius": "10px", "padding": "9px", "marginBottom": "8px", "boxShadow": "0 1px 3px rgba(0, 87, 184, 0.12)"})


def _slice_display(display_df: pd.DataFrame, horizon_days: int, history_hours: int) -> pd.DataFrame:
    latest_hist_dt = display_df.loc[display_df["Actual"].notna(), "DT"].max()
    start_dt = latest_hist_dt - pd.Timedelta(hours=int(history_hours or 72))
    end_dt = latest_hist_dt + pd.Timedelta(days=int(horizon_days or 16))
    return display_df[(display_df["DT"] >= start_dt) & (display_df["DT"] <= end_dt)].copy()


def _future_only(display_df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    latest_hist_dt = display_df.loc[display_df["Actual"].notna(), "DT"].max()
    end_dt = latest_hist_dt + pd.Timedelta(days=int(horizon_days or 16))
    return display_df[(display_df["DT"] > latest_hist_dt) & (display_df["DT"] <= end_dt)].copy()


def _display_confidence_band(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    base = pd.to_numeric(df.get("Forecast"), errors="coerce")
    raw_band = pd.to_numeric(df.get("Band"), errors="coerce") if "Band" in df.columns else pd.Series(np.nan, index=df.index)
    if raw_band.notna().sum() == 0 and {"Upper_Band", "Lower_Band"}.issubset(df.columns):
        upper = pd.to_numeric(df["Upper_Band"], errors="coerce")
        lower = pd.to_numeric(df["Lower_Band"], errors="coerce")
        raw_band = np.maximum((upper - base).abs(), (base - lower).abs())
    raw_band = pd.Series(raw_band, index=df.index, dtype=float).clip(lower=0.0)

    forecast_mask = base.notna() & raw_band.notna()
    display_band = pd.Series(np.nan, index=df.index, dtype=float)
    if not forecast_mask.any():
        return display_band, display_band, display_band

    smoothed = (
        raw_band.loc[forecast_mask]
        .rolling(window=5, center=True, min_periods=1)
        .median()
        .rolling(window=3, center=True, min_periods=1)
        .mean()
    )

    horizon = df.get("Operational_Horizon_Label", pd.Series("", index=df.index)).astype(str)
    horizon_scale = pd.Series(0.72, index=df.index, dtype=float)
    horizon_scale.loc[horizon.eq("Day1")] = 0.75
    horizon_scale.loc[horizon.eq("Days2to3")] = 0.72
    horizon_scale.loc[horizon.eq("Days4to7")] = 0.70
    horizon_scale.loc[horizon.str.contains("Days8to16", case=False, na=False)] = 0.60
    horizon_scale.loc[horizon.eq("Informational")] = 1.0

    risk_mult = pd.to_numeric(df.get("Weather_Input_Risk_Multiplier", pd.Series(1.0, index=df.index)), errors="coerce").fillna(1.0)
    high_risk = risk_mult.gt(1.0) | horizon.str.contains("low_confidence", case=False, na=False)
    global_cap = max(12.0, float(raw_band.loc[forecast_mask].quantile(0.90)) * 0.95)
    high_risk_cap = max(global_cap, float(raw_band.loc[forecast_mask].quantile(0.95)) * 0.90)
    load_cap = base.abs() * 0.18
    cap = np.minimum(load_cap.where(load_cap.notna(), global_cap), global_cap)
    cap.loc[high_risk] = np.minimum(load_cap.loc[high_risk].where(load_cap.loc[high_risk].notna(), high_risk_cap), high_risk_cap)
    floor = np.maximum(2.5, raw_band * 0.35)

    scaled = smoothed * horizon_scale.loc[forecast_mask]
    display_band.loc[forecast_mask] = np.minimum(
        np.maximum(scaled, floor.loc[forecast_mask]),
        cap.loc[forecast_mask],
    )
    upper = base + display_band
    lower = np.maximum(0.0, base - display_band)
    return upper, lower, display_band


def _make_forecast_graph(df: pd.DataFrame, weather_variable: str | None):
    fig = go.Figure()
    has_previous = "Previous_Forecast_MWH" in df.columns and pd.to_numeric(df["Previous_Forecast_MWH"], errors="coerce").notna().any()
    has_previous_miss = "Previous_Forecast_Miss_MWH" in df.columns and pd.to_numeric(df["Previous_Forecast_Miss_MWH"], errors="coerce").notna().any()
    has_prior_run = "Prior_Run_Forecast_MWH" in df.columns and pd.to_numeric(df["Prior_Run_Forecast_MWH"], errors="coerce").notna().any()
    has_forecast_change = "Forecast_Change_From_Prior_Run_MWH" in df.columns and pd.to_numeric(df["Forecast_Change_From_Prior_Run_MWH"], errors="coerce").notna().any()
    has_prior_temp = "Prior_Run_Temperature_F" in df.columns and pd.to_numeric(df["Prior_Run_Temperature_F"], errors="coerce").notna().any()
    has_weather = bool(weather_variable and weather_variable != "none" and weather_variable in df.columns)
    # Bands first, filled between upper/lower.
    if {"Upper_Band", "Lower_Band"}.issubset(df.columns):
        display_upper, display_lower, display_band = _display_confidence_band(df)
        fig.add_trace(go.Scatter(
            x=df["DT"],
            y=display_upper,
            mode="lines",
            name="Display Upper Band",
            line=dict(color=BAND_EDGE, width=0.7),
            showlegend=False,
            hovertemplate="Display Band<br>%{x|%Y-%m-%d %H:%M}<br>Upper: %{y:,.1f} MWh<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=df["DT"],
            y=display_lower,
            mode="lines",
            name="Smoothed Forecast Band",
            fill="tonexty",
            fillcolor=BAND_FILL,
            line=dict(color=BAND_EDGE, width=0.7),
            customdata=np.column_stack([
                pd.to_numeric(df.get("Band", pd.Series(np.nan, index=df.index)), errors="coerce"),
                display_band,
            ]),
            hovertemplate=(
                "Smoothed Forecast Band<br>%{x|%Y-%m-%d %H:%M}<br>"
                "Lower: %{y:,.1f} MWh<br>"
                "Display half-width: %{customdata[1]:,.1f} MWh<br>"
                "Operational half-width: %{customdata[0]:,.1f} MWh<extra></extra>"
            ),
        ))

    if "Actual" in df.columns:
        fig.add_trace(go.Scatter(x=df["DT"], y=df["Actual"], mode="lines", name="Actual", line=dict(color=BRAND_RED, width=2.1)))
    if has_previous:
        fig.add_trace(go.Scatter(
            x=df["DT"],
            y=pd.to_numeric(df["Previous_Forecast_MWH"], errors="coerce"),
            mode="lines",
            name="Previous Forecast",
            line=dict(color="#4D4D4D", width=2.0, dash="dash"),
            hovertemplate="Previous Forecast<br>%{x|%Y-%m-%d %H:%M}<br>%{y:,.1f} MWh<extra></extra>",
        ))
    if has_prior_run:
        fig.add_trace(go.Scatter(
            x=df["DT"],
            y=pd.to_numeric(df["Prior_Run_Forecast_MWH"], errors="coerce"),
            mode="lines",
            name="Prior Run Forecast",
            line=dict(color="#5A5A5A", width=2.0, dash="dashdot"),
            hovertemplate="Prior Run Forecast<br>%{x|%Y-%m-%d %H:%M}<br>%{y:,.1f} MWh<extra></extra>",
        ))
    if has_forecast_change:
        delta = pd.to_numeric(df["Forecast_Change_From_Prior_Run_MWH"], errors="coerce")
        delta_axis = "y3" if has_weather else "y2"
        delta_colors = np.where(delta.fillna(0.0) >= 0.0, BRAND_BLUE, BRAND_PURPLE)
        fig.add_trace(go.Bar(
            x=df["DT"],
            y=delta,
            name="Forecast Change",
            marker_color=delta_colors,
            opacity=0.24,
            yaxis=delta_axis,
            hovertemplate="Forecast Change<br>%{x|%Y-%m-%d %H:%M}<br>Current - Prior: %{y:+,.1f} MWh<extra></extra>",
        ))
    if has_previous_miss:
        miss = pd.to_numeric(df["Previous_Forecast_Miss_MWH"], errors="coerce")
        miss_axis = "y3" if has_weather else "y2"
        miss_colors = np.where(miss.fillna(0.0) >= 0.0, BRAND_GOLD, BRAND_PURPLE)
        fig.add_trace(go.Bar(
            x=df["DT"],
            y=miss,
            name="Previous Miss",
            marker_color=miss_colors,
            opacity=0.32,
            yaxis=miss_axis,
            hovertemplate="Previous Miss<br>%{x|%Y-%m-%d %H:%M}<br>Actual - Forecast: %{y:+,.1f} MWh<extra></extra>",
        ))
    if "Raw_Forecast_MWH" in df.columns:
        fig.add_trace(go.Scatter(x=df["DT"], y=df["Raw_Forecast_MWH"], mode="lines", name="Raw XGB+LGB", line=dict(color=BRAND_PURPLE, width=1.5, dash="dot")))
    if "Prophet_Pred_MWH" in df.columns:
        fig.add_trace(go.Scatter(x=df["DT"], y=df["Prophet_Pred_MWH"], mode="lines", name="Prophet", line=dict(color=BRAND_GREEN, width=1.3, dash="dash")))
    fig.add_trace(go.Scatter(x=df["DT"], y=df["Forecast"], mode="lines", name="Calibrated Forecast", line=dict(color=BRAND_BLUE, width=2.8)))

    if {"DT", "Actual", "Forecast"}.issubset(df.columns):
        actual_rows = df[pd.to_numeric(df["Actual"], errors="coerce").notna()].copy()
        forecast_rows = df[pd.to_numeric(df["Forecast"], errors="coerce").notna()].copy()
        if not actual_rows.empty and not forecast_rows.empty:
            latest_actual = actual_rows.sort_values("DT").iloc[-1]
            first_forecast = forecast_rows.sort_values("DT").iloc[0]
            if pd.to_datetime(first_forecast["DT"], errors="coerce") > pd.to_datetime(latest_actual["DT"], errors="coerce"):
                fig.add_trace(go.Scatter(
                    x=[latest_actual["DT"], first_forecast["DT"]],
                    y=[latest_actual["Actual"], first_forecast["Forecast"]],
                    mode="lines+markers",
                    name="Actual-Forecast Handoff",
                    line=dict(color="#222222", width=1.8, dash="dot"),
                    marker=dict(size=6, color="#222222"),
                    hovertemplate="Actual-Forecast Handoff<br>%{x|%Y-%m-%d %H:%M}<br>%{y:,.1f} MWh<extra></extra>",
                ))

    if has_weather:
        fig.add_trace(go.Scatter(
            x=df["DT"], y=df[weather_variable], mode="lines", name=weather_variable,
            yaxis="y2", line=dict(color=BRAND_GOLD, width=1.8, dash="dash")
        ))
        if weather_variable == "Temperature" and has_prior_temp:
            fig.add_trace(go.Scatter(
                x=df["DT"],
                y=pd.to_numeric(df["Prior_Run_Temperature_F"], errors="coerce"),
                mode="lines",
                name="Prior Run Temperature",
                yaxis="y2",
                line=dict(color="#8A6D1D", width=1.5, dash="dot"),
                hovertemplate="Prior Run Temperature<br>%{x|%Y-%m-%d %H:%M}<br>%{y:,.1f} F<extra></extra>",
            ))

    has_delta_axis = has_previous_miss or has_forecast_change
    yaxis2_title = weather_variable if has_weather else ("Delta MWh" if has_delta_axis else "")
    yaxis3 = None
    if has_weather and has_delta_axis:
        yaxis3 = {
            "title": "Delta MWh",
            "overlaying": "y",
            "side": "right",
            "anchor": "free",
            "position": 0.98,
            "showgrid": False,
        }
    fig.update_layout(
        title="Forecast vs Actual with Prior Run Comparison (band shown is display-smoothed; table carries operational Low/Expected/High)",
        xaxis={"title": "Date / Hour", "showgrid": True, "gridcolor": "rgba(0,87,184,0.10)"},
        yaxis={"title": "MWh", "showgrid": True, "gridcolor": "rgba(0,87,184,0.10)"},
        yaxis2={"title": yaxis2_title, "overlaying": "y", "side": "right", "showgrid": False},
        yaxis3=yaxis3,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 55, "r": 55, "t": 55, "b": 42},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _attach_previous_forecast_history(display_df: pd.DataFrame, backtest_df: pd.DataFrame) -> pd.DataFrame:
    if display_df is None or display_df.empty or backtest_df is None or backtest_df.empty:
        return display_df
    if "DT" not in display_df.columns or "DT" not in backtest_df.columns:
        return display_df

    forecast_candidates = [
        "Final_Backtest_Forecast_MWH",
        "Recent_Corrected_Forecast_MWH",
        "Stage_Selected_Forecast_MWH",
        "Calibrated_Forecast_MWH",
        "Raw_Forecast_MWH",
    ]
    forecast_col = next((c for c in forecast_candidates if c in backtest_df.columns), None)
    if forecast_col is None:
        return display_df

    keep = ["DT", forecast_col]
    for col in ["Final_Residual_MWH", "Final_AbsError_MWH", "Actual_MWH"]:
        if col in backtest_df.columns:
            keep.append(col)
    prev = backtest_df[keep].copy()
    prev.rename(columns={forecast_col: "Previous_Forecast_MWH"}, inplace=True)
    prev["Previous_Forecast_MWH"] = pd.to_numeric(prev["Previous_Forecast_MWH"], errors="coerce")

    if "Final_Residual_MWH" in prev.columns:
        prev["Previous_Forecast_Miss_MWH"] = pd.to_numeric(prev["Final_Residual_MWH"], errors="coerce")
    elif "Actual_MWH" in prev.columns:
        prev["Previous_Forecast_Miss_MWH"] = pd.to_numeric(prev["Actual_MWH"], errors="coerce") - prev["Previous_Forecast_MWH"]
    else:
        prev["Previous_Forecast_Miss_MWH"] = np.nan
    if "Final_AbsError_MWH" in prev.columns:
        prev["Previous_Forecast_AbsMiss_MWH"] = pd.to_numeric(prev["Final_AbsError_MWH"], errors="coerce")
    else:
        prev["Previous_Forecast_AbsMiss_MWH"] = prev["Previous_Forecast_Miss_MWH"].abs()

    prev = prev[["DT", "Previous_Forecast_MWH", "Previous_Forecast_Miss_MWH", "Previous_Forecast_AbsMiss_MWH"]]
    prev = prev.dropna(subset=["DT"]).drop_duplicates(subset=["DT"], keep="last")
    out = display_df.merge(prev, on="DT", how="left")
    return out


def _attach_prior_run_comparison(
    display_df: pd.DataFrame,
    previous_forecast_df: pd.DataFrame | None,
    previous_weather_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if display_df is None or display_df.empty or "DT" not in display_df.columns:
        return display_df
    out = display_df.copy()
    out["__DT_KEY"] = pd.to_datetime(out["DT"], errors="coerce", utc=True)

    if previous_forecast_df is not None and not previous_forecast_df.empty and "DT" in previous_forecast_df.columns:
        forecast_col = "Forecast" if "Forecast" in previous_forecast_df.columns else None
        if forecast_col is None and "Final_Forecast_MWH" in previous_forecast_df.columns:
            forecast_col = "Final_Forecast_MWH"
        if forecast_col is not None:
            prev = previous_forecast_df[["DT", forecast_col]].copy()
            prev["__DT_KEY"] = pd.to_datetime(prev["DT"], errors="coerce", utc=True)
            prev.rename(columns={forecast_col: "Prior_Run_Forecast_MWH"}, inplace=True)
            prev["Prior_Run_Forecast_MWH"] = pd.to_numeric(prev["Prior_Run_Forecast_MWH"], errors="coerce")
            prev = prev.dropna(subset=["__DT_KEY"]).drop_duplicates(subset=["__DT_KEY"], keep="last")
            out = out.merge(prev[["__DT_KEY", "Prior_Run_Forecast_MWH"]], on="__DT_KEY", how="left")
            out["Forecast_Change_From_Prior_Run_MWH"] = (
                pd.to_numeric(out.get("Forecast"), errors="coerce")
                - pd.to_numeric(out.get("Prior_Run_Forecast_MWH"), errors="coerce")
            )

    if previous_weather_df is not None and not previous_weather_df.empty and "DT" in previous_weather_df.columns:
        temp_col = "TempF" if "TempF" in previous_weather_df.columns else None
        if temp_col is None and "Temperature" in previous_weather_df.columns:
            temp_col = "Temperature"
        if temp_col is not None:
            prev_wx = previous_weather_df[["DT", temp_col]].copy()
            prev_wx["__DT_KEY"] = pd.to_datetime(prev_wx["DT"], errors="coerce", utc=True)
            prev_wx.rename(columns={temp_col: "Prior_Run_Temperature_F"}, inplace=True)
            prev_wx["Prior_Run_Temperature_F"] = pd.to_numeric(prev_wx["Prior_Run_Temperature_F"], errors="coerce")
            prev_wx = prev_wx.dropna(subset=["__DT_KEY"]).drop_duplicates(subset=["__DT_KEY"], keep="last")
            out = out.merge(prev_wx[["__DT_KEY", "Prior_Run_Temperature_F"]], on="__DT_KEY", how="left")
            out["Temperature_Change_From_Prior_Run_F"] = (
                pd.to_numeric(out.get("Temperature"), errors="coerce")
                - pd.to_numeric(out.get("Prior_Run_Temperature_F"), errors="coerce")
            )

    out.drop(columns=["__DT_KEY"], inplace=True, errors="ignore")
    return out


def _make_backtest_graph(backtest_df: pd.DataFrame):
    fig = go.Figure()
    if backtest_df is None or backtest_df.empty:
        fig.update_layout(title="Backtest unavailable")
        return fig
    fig.add_trace(go.Scatter(x=backtest_df["DT"], y=backtest_df["Actual_MWH"], mode="lines", name="Actual", line=dict(color=BRAND_RED, width=2.0)))
    fig.add_trace(go.Scatter(x=backtest_df["DT"], y=backtest_df["Raw_Forecast_MWH"], mode="lines", name="Holdout Raw XGB+LGB", line=dict(color=BRAND_PURPLE, width=1.5, dash="dot")))
    if "Recent_Corrected_Forecast_MWH" in backtest_df.columns:
        fig.add_trace(go.Scatter(x=backtest_df["DT"], y=backtest_df["Recent_Corrected_Forecast_MWH"], mode="lines", name="Recent Corrected", line=dict(color=BRAND_BLUE, width=2.2)))
    if "Prophet_Pred_MWH" in backtest_df.columns:
        fig.add_trace(go.Scatter(x=backtest_df["DT"], y=backtest_df["Prophet_Pred_MWH"], mode="lines", name="Prophet", line=dict(color=BRAND_GREEN, width=1.2, dash="dash")))
    fig.add_trace(go.Bar(x=backtest_df["DT"], y=backtest_df["Residual_MWH"], name="Residual", marker_color=BRAND_PURPLE, yaxis="y2", opacity=0.35))
    fig.update_layout(
        title="Leakage-Safe Recent Backtest",
        xaxis={"title": "Date / Hour"},
        yaxis={"title": "MWh"},
        yaxis2={"title": "Residual MWh", "overlaying": "y", "side": "right", "showgrid": False},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 55, "r": 55, "t": 55, "b": 42},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _make_weather_graph(df: pd.DataFrame):
    fig = go.Figure()
    for col, name, color in [
        ("Temperature", "Temp", BRAND_RED),
        ("Temperature_DailyMax", "Daily Max Temp", BRAND_PURPLE),
        ("BTM_Solar_Proxy_MW", "BTM Solar Proxy", BRAND_GOLD),
        ("Solar_Irradiance", "Solar Irradiance", BRAND_GREEN),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["DT"], y=df[col], mode="lines", name=name, line=dict(color=color, width=1.8)))
    fig.update_layout(
        title="Weather and BTM Solar Drivers",
        xaxis={"title": "Date / Hour"},
        yaxis={"title": "Driver Value"},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 55, "r": 30, "t": 55, "b": 42},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _make_error_by_hour_graph(backtest_df: pd.DataFrame):
    fig = go.Figure()
    if backtest_df is None or backtest_df.empty:
        return fig
    work = backtest_df.copy()
    if "Recent_Corrected_Residual_MWH" in work.columns:
        work["DisplayResidual"] = pd.to_numeric(work["Recent_Corrected_Residual_MWH"], errors="coerce")
        title = "Final Corrected Backtest Error by Hour"
    else:
        work["DisplayResidual"] = pd.to_numeric(work["Residual_MWH"], errors="coerce")
        title = "Raw Backtest Error by Hour"
    work["DisplayAbsError"] = work["DisplayResidual"].abs()
    grp = work.groupby("Hour", as_index=False).agg(MAE=("DisplayAbsError", "mean"), Bias=("DisplayResidual", "mean"))
    fig.add_trace(go.Bar(x=grp["Hour"], y=grp["MAE"], name="MAE", marker_color=BRAND_BLUE))
    fig.add_trace(go.Scatter(x=grp["Hour"], y=grp["Bias"], mode="lines+markers", name="Bias", line=dict(color=BRAND_PURPLE, width=2.0)))
    fig.update_layout(title=title, xaxis={"title": "Hour"}, yaxis={"title": "MWh"}, plot_bgcolor="white", paper_bgcolor="white", margin={"l": 50, "r": 25, "t": 55, "b": 38})
    return fig


def _residual_for_validation(backtest_df: pd.DataFrame) -> pd.Series:
    if backtest_df is None or backtest_df.empty:
        return pd.Series(dtype=float)
    for col in ["Final_Residual_MWH", "Recent_Corrected_Residual_MWH", "Residual_MWH"]:
        if col in backtest_df.columns:
            return pd.to_numeric(backtest_df[col], errors="coerce").dropna()
    return pd.Series(dtype=float)


def _acf_at_lag(values: pd.Series, lag: int) -> float:
    if values is None or len(values) <= lag:
        return np.nan
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) <= lag:
        return np.nan
    a = x[:-lag]
    b = x[lag:]
    if np.nanstd(a) < 1e-9 or np.nanstd(b) < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _durbin_watson(values: pd.Series) -> float:
    if values is None or len(values) < 3:
        return np.nan
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    denom = float(np.dot(x, x))
    if denom <= 1e-9:
        return np.nan
    diff = np.diff(x)
    return float(np.dot(diff, diff) / denom)


def _make_validation_detail_graph(backtest_df: pd.DataFrame):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.18,
        subplot_titles=("Final Error by Hour", "Residual Serial Dependence Check"),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )
    if backtest_df is None or backtest_df.empty:
        fig.update_layout(title="Validation detail unavailable", plot_bgcolor="white", paper_bgcolor="white")
        return fig

    work = backtest_df.copy()
    if "Final_Residual_MWH" in work.columns:
        work["DisplayResidual"] = pd.to_numeric(work["Final_Residual_MWH"], errors="coerce")
    elif "Recent_Corrected_Residual_MWH" in work.columns:
        work["DisplayResidual"] = pd.to_numeric(work["Recent_Corrected_Residual_MWH"], errors="coerce")
    else:
        work["DisplayResidual"] = pd.to_numeric(work.get("Residual_MWH"), errors="coerce")
    if "Hour" not in work.columns and "DT" in work.columns:
        work["Hour"] = pd.to_datetime(work["DT"], errors="coerce").dt.hour
    work["DisplayAbsError"] = work["DisplayResidual"].abs()
    if "Hour" in work.columns:
        grp = work.groupby("Hour", as_index=False).agg(MAE=("DisplayAbsError", "mean"), Bias=("DisplayResidual", "mean"))
        fig.add_trace(go.Bar(x=grp["Hour"], y=grp["MAE"], name="MAE by Hour", marker_color=BRAND_BLUE), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=grp["Hour"], y=grp["Bias"], mode="lines+markers", name="Bias by Hour", line=dict(color=BRAND_PURPLE, width=2.0)), row=1, col=1, secondary_y=True)

    residual = _residual_for_validation(backtest_df)
    lags = [1, 2, 3, 6, 12, 24, 48, 168]
    acf = [_acf_at_lag(residual, lag) for lag in lags]
    colors = [BRAND_RED if np.isfinite(v) and abs(v) >= 0.30 else BRAND_GREEN for v in acf]
    fig.add_trace(go.Bar(
        x=[str(lag) for lag in lags],
        y=acf,
        name="Residual ACF",
        marker_color=colors,
        hovertemplate="Lag %{x}<br>ACF %{y:.3f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0.30, line_dash="dot", line_color=BRAND_GOLD, row=2, col=1)
    fig.add_hline(y=-0.30, line_dash="dot", line_color=BRAND_GOLD, row=2, col=1)

    fig.update_yaxes(title_text="MWh", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Bias", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="Hour", row=1, col=1)
    fig.update_yaxes(title_text="ACF", range=[-1, 1], row=2, col=1)
    fig.update_xaxes(title_text="Lag Hours", row=2, col=1)
    fig.update_layout(
        title="Validation Detail: Hourly Error and Residual Dependence",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 55, "r": 55, "t": 75, "b": 42},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _make_production_scorecard_graph(diagnostics_results: dict):
    scorecard = (diagnostics_results or {}).get("production_readiness_scorecard")
    fig = go.Figure()
    if scorecard is None or not isinstance(scorecard, pd.DataFrame) or scorecard.empty:
        fig.update_layout(title="Production Readiness Scorecard unavailable", plot_bgcolor="white", paper_bgcolor="white")
        return fig
    stale = bool((diagnostics_results or {}).get("_stale_scorecards", {}).get("production_readiness_scorecard", False))
    work = scorecard.copy()
    work["MAE_MWH"] = pd.to_numeric(work.get("MAE_MWH"), errors="coerce")
    work["MAPE_PCT"] = pd.to_numeric(work.get("MAPE_PCT"), errors="coerce")
    gate_text = work.get("Gate", pd.Series("", index=work.index)).astype(str)
    work["MAE_Gate_MWH"] = gate_text.map(
        lambda gate: float(gate.split("mae<=", 1)[1].split(";")[0])
        if "mae<=" in gate else np.nan
    )
    work["MAE_Delta_To_Gate_MWH"] = work["MAE_MWH"] - work["MAE_Gate_MWH"]
    passed = work.get("Pass", pd.Series(False, index=work.index)).astype(str).str.lower().eq("true")
    colors = np.where(passed, BRAND_GREEN, BRAND_GOLD)
    fig.add_trace(go.Bar(
        x=work["Test"],
        y=work["MAE_MWH"],
        name="MAE",
        marker_color=colors,
        customdata=np.stack([
            work.get("Purpose", pd.Series("", index=work.index)).astype(str),
            work.get("Target", pd.Series("", index=work.index)).astype(str),
            np.where(passed, "Pass", "Caution"),
            work["MAE_Delta_To_Gate_MWH"],
        ], axis=-1),
        hovertemplate="<b>%{x}</b><br>MAE %{y:.2f} MWh<br>Delta to MAE gate: %{customdata[3]:+.2f} MWh<br>%{customdata[0]}<br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>",
    ))
    if work["MAE_Gate_MWH"].notna().any():
        fig.add_trace(go.Scatter(
            x=work["Test"],
            y=work["MAE_Gate_MWH"],
            mode="lines+markers",
            name="MAE Gate",
            line=dict(color=BRAND_RED, width=1.8, dash="dot"),
            hovertemplate="<b>%{x}</b><br>MAE gate %{y:.2f} MWh<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=work["Test"],
        y=work["MAPE_PCT"],
        mode="lines+markers",
        name="MAPE %",
        yaxis="y2",
        line=dict(color=BRAND_BLUE, width=2.0),
    ))
    fig.update_layout(
        title="Official Production Readiness Scorecard - Replay First" + (" (stale vs current forecast)" if stale else ""),
        xaxis={"title": "", "tickangle": -25},
        yaxis={"title": "MAE (MWh)"},
        yaxis2={"title": "MAPE (%)", "overlaying": "y", "side": "right", "showgrid": False},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 55, "r": 55, "t": 60, "b": 95},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _table_data(df: pd.DataFrame):
    fut = df[df["Forecast"].notna()].copy()
    keep = [
        "DT", "Forecast", "Actual", "Raw_Forecast_MWH", "XGB_Pred_MWH", "LGB_Pred_MWH", "CatBoost_Pred_MWH", "Prophet_Pred_MWH",
        "Stage_Selected_Forecast_MWH", "Stage_Selector_Reason", "Residual_Calibrated_Forecast_MWH", "Heat_Adjusted_Forecast_MWH",
        "Warm_Ramp_Adjusted_Forecast_MWH", "Peak_Risk_Adjusted_Forecast_MWH", "Recent_Corrected_Forecast_MWH",
        "Prior_Run_Forecast_MWH", "Forecast_Change_From_Prior_Run_MWH",
        "P10_Forecast_MWH", "P50_Forecast_MWH", "P90_Forecast_MWH",
        "Forecast_Low_MWH", "Forecast_Expected_MWH", "Forecast_High_MWH",
        "Upper_Band", "Lower_Band", "Band",
        "Operational_Horizon_Label", "Production_Confidence_Label", "Production_Risk_Code",
        "Production_Caution_Reason", "Weather_Input_Risk_Class",
        "WeatherScenario_Spread_MWH", "WeatherScenario_MaxAbsDelta_MWH", "WeatherScenario_Cap_Applied",
        "Load_Source", "FiveMin_Interval_Count", "FiveMin_Hourly_Last_MW", "FiveMin_Hourly_Range_MW",
        "Temperature", "Prior_Run_Temperature_F", "Temperature_Change_From_Prior_Run_F", "Temperature_DailyMax", "BTM_Solar_Proxy_MW", "BTM_Solar_Loss_From_ClearSky_MW",
        "FiveMin_Load_Available", "FiveMin_Data_Age_Hours", "FiveMin_PrevHour_Avg_MW", "FiveMin_PrevHour_Last_MW",
        "FiveMin_PrevHour_Ramp_MW", "FiveMin_Ramp_15Min_MW", "FiveMin_Ramp_60Min_MW",
        "Residual_Cal_MWH", "Heat_Peak_Cal_MWH", "Warm_Ramp_Cal_MWH", "Recent_Level_Correction_MWH", "Calibration_Level",
    ]
    keep = [c for c in keep if c in fut.columns]
    t = fut[keep].copy()
    t["DT"] = pd.to_datetime(t["DT"]).dt.strftime("%m/%d/%Y %H:%M")
    rename = {
        "Raw_Forecast_MWH": "Raw XGB+LGB",
        "XGB_Pred_MWH": "XGB",
        "LGB_Pred_MWH": "LGB",
        "CatBoost_Pred_MWH": "CatBoost",
        "Prophet_Pred_MWH": "Prophet",
        "Stage_Selected_Forecast_MWH": "Stage Selected",
        "Stage_Selector_Reason": "Stage Reason",
        "Residual_Calibrated_Forecast_MWH": "Residual Cal Fcst",
        "Heat_Adjusted_Forecast_MWH": "Heat Fcst",
        "Warm_Ramp_Adjusted_Forecast_MWH": "Warm Ramp Fcst",
        "Peak_Risk_Adjusted_Forecast_MWH": "Peak Risk Fcst",
        "Recent_Corrected_Forecast_MWH": "Recent Fcst",
        "Prior_Run_Forecast_MWH": "Prior Run Fcst",
        "Forecast_Change_From_Prior_Run_MWH": "Fcst Change",
        "P10_Forecast_MWH": "P10",
        "P50_Forecast_MWH": "P50",
        "P90_Forecast_MWH": "P90",
        "Forecast_Low_MWH": "Low",
        "Forecast_Expected_MWH": "Expected",
        "Forecast_High_MWH": "High",
        "Upper_Band": "Upper",
        "Lower_Band": "Lower",
        "Operational_Horizon_Label": "Horizon",
        "Production_Confidence_Label": "Confidence",
        "Production_Risk_Code": "Risk Code",
        "Production_Caution_Reason": "Caution Reason",
        "Weather_Input_Risk_Class": "Weather Risk",
        "WeatherScenario_Spread_MWH": "Scenario Spread",
        "WeatherScenario_MaxAbsDelta_MWH": "Scenario Max Abs",
        "WeatherScenario_Cap_Applied": "Scenario Cap",
        "Load_Source": "Load Source",
        "FiveMin_Interval_Count": "5m Hour Count",
        "FiveMin_Hourly_Last_MW": "5m Hour Last",
        "FiveMin_Hourly_Range_MW": "5m Hour Range",
        "Temperature": "Temp",
        "Prior_Run_Temperature_F": "Prior Temp",
        "Temperature_Change_From_Prior_Run_F": "Temp Change",
        "Temperature_DailyMax": "Daily Max",
        "FiveMin_Load_Available": "5m Avail",
        "FiveMin_Data_Age_Hours": "5m Age Hr",
        "FiveMin_PrevHour_Avg_MW": "5m Prev Hr Avg",
        "FiveMin_PrevHour_Last_MW": "5m Prev Hr Last",
        "FiveMin_PrevHour_Ramp_MW": "5m Prev Hr Ramp",
        "FiveMin_Ramp_15Min_MW": "5m Ramp 15",
        "FiveMin_Ramp_60Min_MW": "5m Ramp 60",
        "BTM_Solar_Proxy_MW": "BTM Solar",
        "BTM_Solar_Loss_From_ClearSky_MW": "BTM Solar Loss",
        "Residual_Cal_MWH": "Residual Cal",
        "Heat_Peak_Cal_MWH": "Heat Cal",
        "Warm_Ramp_Cal_MWH": "Warm Cal",
        "Recent_Level_Correction_MWH": "Recent Cal",
        "Calibration_Level": "Cal Level",
    }
    t.rename(columns=rename, inplace=True)
    if "Scenario Cap" in t.columns:
        cap_flag = t["Scenario Cap"].astype(str).str.lower().isin({"true", "1", "1.0", "yes"})
        t["Scenario Cap"] = np.where(cap_flag, "Yes", "")
    text_cols = {
        "DT", "Cal Level", "Horizon", "Confidence", "Risk Code", "Caution Reason",
        "Weather Risk", "Stage Reason", "Load Source", "Scenario Cap",
    }
    for c in t.columns:
        if c not in text_cols:
            t[c] = t[c].map(lambda x: _fmt_num(x, 1))
    columns = [{"name": c, "id": c} for c in t.columns]
    return t.to_dict("records"), columns


def _temp_sensitivity_table_data(frame: pd.DataFrame, selected_steps: list[str] | None):
    if frame is None or frame.empty:
        return [], []

    selected = set(selected_steps or [])
    ordered = [step for step in TEMP_SENSITIVITY_STEPS if step[0] in selected]
    if not ordered:
        ordered = [step for step in TEMP_SENSITIVITY_STEPS if step[0] == "baseline"]

    table = pd.DataFrame()
    timestamps = pd.to_datetime(frame["DT"], errors="coerce")
    table["Date"] = timestamps.dt.strftime("%Y-%m-%d")
    table["Time"] = timestamps.dt.strftime("%H:%M")
    for key, label, _delta, _color in ordered:
        if key in frame.columns:
            table[label] = pd.to_numeric(frame[key], errors="coerce").map(lambda x: _fmt_num(x, 1))
    columns = [{"name": c, "id": c} for c in table.columns]
    return table.to_dict("records"), columns


def _scorecard_metric(scorecard: pd.DataFrame, test_name: str, value_col: str = "MAE_MWH") -> tuple[str, str] | None:
    if scorecard is None or scorecard.empty or "Test" not in scorecard.columns:
        return None
    row = scorecard[scorecard["Test"].astype(str).eq(test_name)]
    if row.empty:
        return None
    item = row.iloc[0]
    value = pd.to_numeric(pd.Series([item.get(value_col)]), errors="coerce").iloc[0]
    passed = str(item.get("Pass", "")).lower() == "true"
    status = "Pass" if passed else "Caution"
    if pd.isna(value):
        return status, "N/A"
    unit = "%" if value_col == "MAPE_PCT" else " MWh"
    return status, f"{value:,.2f}{unit}"


def _cards(display_df: pd.DataFrame, backtest_df: pd.DataFrame, horizon_days: int, diagnostics_results: dict | None = None):
    fut = _future_only(display_df, horizon_days)
    forecast = pd.to_numeric(fut.get("Forecast"), errors="coerce")
    raw = pd.to_numeric(fut.get("Raw_Forecast_MWH"), errors="coerce") if "Raw_Forecast_MWH" in fut else pd.Series(dtype=float)
    peak_dt = fut.loc[forecast.idxmax(), "DT"] if forecast.notna().any() else None
    peak_delta = (forecast.max() - raw.max()) if forecast.notna().any() and raw.notna().any() else np.nan

    cards = [
        _metric_card("Forecast Energy", f"{forecast.sum():,.0f} MWh" if forecast.notna().any() else "N/A", f"Next {horizon_days or 16} days"),
        _metric_card("Forecast Peak", f"{forecast.max():,.1f} MW" if forecast.notna().any() else "N/A", pd.to_datetime(peak_dt).strftime("%m/%d %H:%M") if peak_dt is not None else ""),
        _metric_card("Calibration Impact", f"{peak_delta:+,.1f} MW" if not pd.isna(peak_delta) else "N/A", "Peak vs raw model"),
    ]
    if display_df is not None and not display_df.empty and "Actual" in display_df.columns:
        actual_rows = display_df[pd.to_numeric(display_df["Actual"], errors="coerce").notna()].copy()
        if not actual_rows.empty and "DT" in actual_rows.columns:
            latest_idx = actual_rows["DT"].idxmax()
            latest_actual_dt = actual_rows.loc[latest_idx, "DT"]
            latest_source = str(actual_rows.loc[latest_idx, "Load_Source"]) if "Load_Source" in actual_rows.columns else "history"
            cards.append(_metric_card(
                "Latest Actual",
                pd.to_datetime(latest_actual_dt).strftime("%m/%d %H:%M"),
                latest_source,
            ))
            if "Load_Source" in actual_rows.columns:
                five_min_hours = int(actual_rows["Load_Source"].astype(str).eq("five_min_completed_hour").sum())
                cards.append(_metric_card("5-Min Anchored Hours", f"{five_min_hours}", "Completed hours appended to load history"))
    if not fut.empty and "DT" in fut.columns:
        first_forecast_dt = fut["DT"].min()
        cards.append(_metric_card("First Forecast Hour", pd.to_datetime(first_forecast_dt).strftime("%m/%d %H:%M"), "Starts after latest actual"))
    if "Forecast_Change_From_Prior_Run_MWH" in fut.columns:
        delta = pd.to_numeric(fut["Forecast_Change_From_Prior_Run_MWH"], errors="coerce")
        if delta.notna().any():
            cards.append(_metric_card("Prior Run Delta", f"{delta.abs().max():,.1f} MWh", f"Mean {delta.mean():+,.1f} MWh"))
    if "Temperature_Change_From_Prior_Run_F" in fut.columns:
        temp_delta = pd.to_numeric(fut["Temperature_Change_From_Prior_Run_F"], errors="coerce")
        if temp_delta.notna().any():
            cards.append(_metric_card("Weather Delta", f"{temp_delta.abs().max():,.1f} F", f"Mean {temp_delta.mean():+,.1f} F"))
    if backtest_df is not None and not backtest_df.empty:
        if "Recent_Corrected_Residual_MWH" in backtest_df.columns:
            residual = pd.to_numeric(backtest_df["Recent_Corrected_Residual_MWH"], errors="coerce")
            label = "Final corrected backtest"
        else:
            residual = pd.to_numeric(backtest_df["Residual_MWH"], errors="coerce")
            label = "Raw recent backtest"
        mae = residual.abs().mean()
        rmse = math.sqrt(np.nanmean(np.square(residual)))
        actual = pd.to_numeric(backtest_df.get("Actual_MWH"), errors="coerce")
        mape = np.nanmean(np.where(actual.abs() > 1e-9, residual.abs() / actual.abs() * 100.0, np.nan))
        cards.extend([
            _metric_card("Holdout MAE", f"{mae:,.1f} MWh", label),
            _metric_card("Holdout RMSE", f"{rmse:,.1f} MWh", f"MAPE {mape:,.2f}%"),
        ])
        validation_residual = _residual_for_validation(backtest_df)
        acf1 = _acf_at_lag(validation_residual, 1)
        dw = _durbin_watson(validation_residual)
        if np.isfinite(acf1):
            status = "Caution: serial dependence" if abs(acf1) >= 0.30 else "Low serial dependence"
            cards.append(_metric_card("Residual ACF Lag 1", f"{acf1:+.2f}", status))
        if np.isfinite(dw):
            cards.append(_metric_card("Durbin-Watson", f"{dw:.2f}", "2.0 is roughly uncorrelated"))
    if "Production_Caution_Flag" in fut.columns:
        caution = pd.to_numeric(fut["Production_Caution_Flag"], errors="coerce").fillna(0).gt(0)
        reasons = fut.get("Production_Caution_Reason", pd.Series("", index=fut.index)).astype(str)
        cards.append(_metric_card("Caution Hours", f"{int(caution.sum())}", "Peak, weather, solar, or long-horizon"))
        cards.append(_metric_card("Hot-Peak Caution", f"{int(reasons.str.contains('hot_peak', case=False, na=False).sum())}", "Data-limited point accuracy"))
    scorecard = (diagnostics_results or {}).get("production_readiness_scorecard")
    if bool((diagnostics_results or {}).get("_stale_scorecards", {}).get("production_readiness_scorecard", False)):
        cards.append(_metric_card("Scorecard Freshness", "Stale", "Rerun rolling-origin replay for current forecast"))
    if isinstance(scorecard, pd.DataFrame) and not scorecard.empty:
        seasonal = _scorecard_metric(scorecard, "Seasonal rolling origins")
        day1 = _scorecard_metric(scorecard, "Day 1 only")
        hot = _scorecard_metric(scorecard, "Hot peak days")
        peak = _scorecard_metric(scorecard, "Peak window hours 14-18")
        if seasonal:
            cards.append(_metric_card("Official Replay MAE", seasonal[1], f"Seasonal rolling origins: {seasonal[0]}"))
        if day1:
            cards.append(_metric_card("Day 1 MAE", day1[1], f"Replay gate: {day1[0]}"))
        if hot:
            cards.append(_metric_card("Hot Peak Gate", hot[1], f"Data-limited: {hot[0]}"))
        if peak:
            cards.append(_metric_card("Peak Window Gate", peak[1], f"Data-limited: {peak[0]}"))
    return cards



def _make_daily_peak_miss_graph(diagnostics_results: dict):
    fig = go.Figure()
    daily = (diagnostics_results or {}).get("daily_peak_miss_by_stage")
    if isinstance(daily, pd.DataFrame) and not daily.empty and "Stage" in daily.columns:
        preferred = daily[daily["Stage"].isin(["final_corrected_production", "recent_corrected_simulation"])]
        if not preferred.empty:
            daily = preferred
    if daily is None or not isinstance(daily, pd.DataFrame) or daily.empty:
        daily = (diagnostics_results or {}).get("daily_peak_miss_table")
    if daily is None or not isinstance(daily, pd.DataFrame) or daily.empty:
        fig.update_layout(title="Daily Peak Miss Diagnostics unavailable", plot_bgcolor="white", paper_bgcolor="white")
        return fig
    work = daily.copy()
    work["Actual_Peak_DT"] = pd.to_datetime(work["Actual_Peak_DT"], errors="coerce")
    fig.add_trace(go.Bar(
        x=work["Actual_Peak_DT"],
        y=work["Underforecast_At_Actual_Peak_MWH"],
        name="Underforecast at Actual Daily Peak",
        marker_color=BRAND_PURPLE,
    ))
    fig.add_trace(go.Scatter(
        x=work["Actual_Peak_DT"],
        y=work["Daily_MAE_MWH"],
        mode="lines+markers",
        name="Daily MAE",
        line=dict(color=BRAND_BLUE, width=2.0),
        yaxis="y2",
    ))
    fig.update_layout(
        title="Daily Peak Miss Diagnostics",
        xaxis={"title": "Actual Daily Peak Date/Hour"},
        yaxis={"title": "Underforecast at Actual Peak (MWh)"},
        yaxis2={"title": "Daily MAE (MWh)", "overlaying": "y", "side": "right", "showgrid": False},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 55, "r": 55, "t": 55, "b": 42},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _make_segment_error_graph(diagnostics_results: dict):
    fig = go.Figure()
    seg = (diagnostics_results or {}).get("backtest_metrics_by_segment_by_stage")
    if isinstance(seg, pd.DataFrame) and not seg.empty and "Stage" in seg.columns:
        preferred = seg[seg["Stage"].isin(["final_corrected_production", "recent_corrected_simulation"])]
        if not preferred.empty:
            seg = preferred
    if seg is None or not isinstance(seg, pd.DataFrame) or seg.empty:
        seg = (diagnostics_results or {}).get("backtest_metrics_by_segment")
    if seg is None or not isinstance(seg, pd.DataFrame) or seg.empty:
        fig.update_layout(title="Segment Error Diagnostics unavailable", plot_bgcolor="white", paper_bgcolor="white")
        return fig
    work = seg.copy().sort_values("MAE_MWH", ascending=False).head(20)
    labels = []
    for _, row in work.iterrows():
        parts = [str(row.get("Segment", "Segment"))]
        for key in ["Season", "Month", "Hour", "HourGroup", "DailyMaxTempBucket", "CloudCoverBucket", "BTMSolarBucket", "IsWeekend", "IsHoliday"]:
            if key in row.index and pd.notna(row[key]):
                parts.append(f"{key}={row[key]}")
        labels.append(" | ".join(parts))
    fig.add_trace(go.Bar(
        x=work["MAE_MWH"],
        y=labels,
        orientation="h",
        name="MAE",
        marker_color=BRAND_BLUE,
    ))
    fig.add_trace(go.Scatter(
        x=work["Bias_MWH"],
        y=labels,
        mode="markers",
        name="Bias",
        marker=dict(color=BRAND_RED, size=8),
    ))
    fig.update_layout(
        title="Worst Backtest Segments by MAE",
        xaxis={"title": "MWh"},
        yaxis={"title": "", "autorange": "reversed"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 230, "r": 35, "t": 55, "b": 38},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _diagnostics_table_data(diagnostics_results: dict):
    top = (diagnostics_results or {}).get("top_100_underforecast_hours_by_stage")
    if isinstance(top, pd.DataFrame) and not top.empty and "Stage" in top.columns:
        preferred = top[top["Stage"].isin(["final_corrected_production", "recent_corrected_simulation"])]
        if not preferred.empty:
            top = preferred
    if top is None or not isinstance(top, pd.DataFrame) or top.empty:
        top = (diagnostics_results or {}).get("top_100_underforecast_hours")
    if top is None or not isinstance(top, pd.DataFrame) or top.empty:
        top = (diagnostics_results or {}).get("daily_peak_miss_table")
    if top is None or not isinstance(top, pd.DataFrame) or top.empty:
        return [], []
    keep = [
        "Stage", "DT", "Actual_Peak_DT", "Date", "Forecast_Lead_Hour", "Forecast_Day", "Season", "Month", "Hour", "HourGroup", "Actual_MWH", "Stage_Forecast_MWH", "Raw_Forecast_MWH",
        "XGB_Pred_MWH", "LGB_Pred_MWH", "Prophet_Pred_MWH", "Stage_Residual_MWH", "Stage_AbsError_MWH", "Stage_APE", "Residual_MWH", "AbsError_MWH", "APE", "Underforecast_At_Actual_Peak_MWH", "Daily_MAE_MWH",
        "Temperature", "Temperature_DailyMax", "DailyMaxTempBucket", "CloudCoverBucket", "BTMSolarBucket",
    ]
    keep = [c for c in keep if c in top.columns]
    t = top[keep].head(100).copy()
    for c in ["DT", "Actual_Peak_DT"]:
        if c in t.columns:
            t[c] = pd.to_datetime(t[c], errors="coerce").dt.strftime("%m/%d/%Y %H:%M")
    for c in t.columns:
        if c not in {"DT", "Actual_Peak_DT", "Date", "Season", "HourGroup", "DailyMaxTempBucket", "CloudCoverBucket", "BTMSolarBucket"}:
            t[c] = t[c].map(lambda x: _fmt_num(x, 2))
    columns = [{"name": c, "id": c} for c in t.columns]
    return t.to_dict("records"), columns

def create_dashboard_app(historical_fit_df: pd.DataFrame, future_results: dict, backtest_results: pd.DataFrame, config: dict, diagnostics_results: dict | None = None):
    app = Dash(__name__)
    app.config.suppress_callback_exceptions = True

    project_tz = str((config or {}).get("project", {}).get("timezone") or "")

    def _coerce_dt(s: pd.Series) -> pd.Series:
        # Handles DST offset changes by normalizing to UTC first, then converting back.
        out = pd.to_datetime(s, errors="coerce", utc=True)
        if project_tz:
            try:
                out = out.dt.tz_convert(project_tz)
            except Exception:
                pass
        return out

    display_df = future_results["display"].copy()
    if "DT" in display_df.columns:
        display_df["DT"] = _coerce_dt(display_df["DT"])
    previous_forecast_snapshot = (future_results or {}).get("previous_forecast_snapshot", pd.DataFrame())
    previous_weather_snapshot = (future_results or {}).get("previous_weather_snapshot", pd.DataFrame())
    hist_df = historical_fit_df.copy() if historical_fit_df is not None else pd.DataFrame()
    if not hist_df.empty and "DT" in hist_df.columns:
        hist_df["DT"] = _coerce_dt(hist_df["DT"])
    backtest_df = backtest_results.copy() if backtest_results is not None else pd.DataFrame()
    if not backtest_df.empty:
        if "DT" in backtest_df.columns:
            backtest_df["DT"] = _coerce_dt(backtest_df["DT"])
    display_df = _attach_previous_forecast_history(display_df, backtest_df)
    display_df = _attach_prior_run_comparison(display_df, previous_forecast_snapshot, previous_weather_snapshot)

    # Dates for the v11.6-style date picker.
    min_date = pd.to_datetime(display_df["DT"].min(), errors="coerce").date() if not display_df.empty else None
    max_date = pd.to_datetime(display_df["DT"].max(), errors="coerce").date() if not display_df.empty else None
    if not display_df.empty and display_df.get("Forecast").notna().any():
        default_start_date = pd.to_datetime(display_df.loc[display_df["Forecast"].notna(), "DT"].min(), errors="coerce").date()
    else:
        default_start_date = pd.to_datetime(display_df["DT"].max(), errors="coerce").date() if not display_df.empty else None
    available_days = sorted(pd.to_datetime(display_df["DT"], errors="coerce").dt.date.dropna().unique().tolist()) if not display_df.empty else []

    app.layout = make_layout(
        min_date=min_date,
        max_date=max_date,
        default_start_date=default_start_date,
        available_days=available_days,
    )

    def _active_tab_styles(active: str):
        base = {
            "border": "none",
            "background": "transparent",
            "color": BRAND_BLUE,
            "padding": "10px 14px",
            "cursor": "pointer",
            "fontSize": "13px",
        }
        active_style = {
            "border": "1px solid #D2D2D2",
            "borderBottom": "1px solid #FFFFFF",
            "background": "#FFFFFF",
            "color": "#222222",
            "padding": "10px 16px",
            "cursor": "pointer",
            "fontWeight": "700",
            "fontSize": "13px",
            "height": "38px",
        }
        styles = {
            "spreadsheet": base.copy(),
            "graph": base.copy(),
            "dual": base.copy(),
            "statistics": base.copy(),
        }
        if active in styles:
            styles[active] = active_style
        return styles["spreadsheet"], styles["graph"], styles["dual"], styles["statistics"]

    def _parse_date(d) -> date_type | None:
        if not d:
            return None
        try:
            return pd.to_datetime(d).date()
        except Exception:
            return None

    def _clamp_to_available(target: date_type, avail: list[date_type]) -> date_type:
        if not avail:
            return target
        if target in avail:
            return target
        # pick closest available date
        best = min(avail, key=lambda x: abs((x - target).days))
        return best

    def _window(df: pd.DataFrame, start_d: date_type | None, horizon_days: int, history_hours: int) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if not start_d:
            return _slice_display(df, horizon_days, history_hours)
        start_ts = pd.Timestamp(start_d)
        if project_tz:
            try:
                start_ts = start_ts.tz_localize(project_tz)
            except Exception:
                # If already tz-aware or localization fails, keep as-is.
                pass
        # keep history before start date as well (history-hours), but anchor end on start + horizon
        end_ts = start_ts + pd.Timedelta(days=int(horizon_days or 16))
        # show some history prior to start date so the user can see the seam
        start_hist = start_ts - pd.Timedelta(hours=int(history_hours or 72))
        return df[(df["DT"] >= start_hist) & (df["DT"] < end_ts)].copy()

    def _temp_sensitivity_history(hist: pd.DataFrame, display: pd.DataFrame) -> pd.DataFrame:
        frames = []
        if hist is not None and not hist.empty and {"DT", "MWH", "Temperature"}.issubset(hist.columns):
            frames.append(hist[["DT", "MWH", "Temperature"]].copy())
        if display is not None and not display.empty and {"DT", "Actual", "Temperature"}.issubset(display.columns):
            display_hist = display.loc[display["Actual"].notna(), ["DT", "Actual", "Temperature"]].copy()
            display_hist.rename(columns={"Actual": "MWH"}, inplace=True)
            frames.append(display_hist)
        if not frames:
            return pd.DataFrame()

        work = pd.concat(frames, ignore_index=True)
        work["DT"] = pd.to_datetime(work["DT"], errors="coerce")
        work["MWH"] = pd.to_numeric(work["MWH"], errors="coerce")
        work["Temperature"] = pd.to_numeric(work["Temperature"], errors="coerce")
        work = work.dropna(subset=["DT", "MWH", "Temperature"]).drop_duplicates(subset=["DT"], keep="first")
        work["Hour"] = work["DT"].dt.hour
        return work

    def _fit_positive_temp_slope(work: pd.DataFrame) -> float:
        if work is None or len(work) < 24:
            return np.nan
        x = pd.to_numeric(work["Temperature"], errors="coerce").to_numpy()
        y = pd.to_numeric(work["MWH"], errors="coerce").to_numpy()
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 24 or np.nanstd(x[finite]) < 0.1:
            return np.nan
        try:
            slope, _intercept = np.polyfit(x[finite], y[finite], deg=1)
        except Exception:
            return np.nan
        return float(slope) if np.isfinite(slope) and slope > 0 else np.nan

    def _temp_sensitivity_slopes(history: pd.DataFrame) -> dict[int, float]:
        if history is None or history.empty:
            return {hour: 0.0 for hour in range(24)}

        warm_cutoff = max(65.0, float(history["Temperature"].quantile(0.60)))
        warm = history[history["Temperature"] >= warm_cutoff].copy()
        fallback = _fit_positive_temp_slope(warm)
        if not np.isfinite(fallback):
            fallback = _fit_positive_temp_slope(history)
        if not np.isfinite(fallback):
            fallback = 0.0

        slopes = {}
        for hour in range(24):
            slope = _fit_positive_temp_slope(warm[warm["Hour"] == hour])
            slopes[hour] = float(slope if np.isfinite(slope) else fallback)
        return slopes

    def _make_temp_sensitivity_frame(display: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
        if display is None or display.empty or not {"DT", "Forecast"}.issubset(display.columns):
            return pd.DataFrame()

        frame = display.loc[display["Forecast"].notna()].copy()
        if frame.empty:
            return frame
        frame["DT"] = pd.to_datetime(frame["DT"], errors="coerce")
        frame["Forecast"] = pd.to_numeric(frame["Forecast"], errors="coerce")
        frame = frame.dropna(subset=["DT", "Forecast"]).sort_values("DT")
        if frame.empty:
            return frame

        history = _temp_sensitivity_history(hist, display_df)
        slopes = _temp_sensitivity_slopes(history)
        frame["Temp_Sensitivity_MWH_Per_F"] = frame["DT"].dt.hour.map(slopes).fillna(0.0)
        for key, _label, delta, _color in TEMP_SENSITIVITY_STEPS:
            frame[key] = frame["Forecast"] + (frame["Temp_Sensitivity_MWH_Per_F"] * delta)
        return frame

    def _make_temp_sensitivity_graph(
        frame: pd.DataFrame,
        selected_steps: list[str] | None,
        show_weather: bool,
        highlight_max: bool,
    ) -> go.Figure:
        fig = go.Figure()
        if frame is None or frame.empty:
            fig.update_layout(title="Temperature Sensitivity unavailable", plot_bgcolor="white", paper_bgcolor="white")
            return fig

        selected = set(selected_steps or [])
        ordered = [step for step in TEMP_SENSITIVITY_STEPS if step[0] in selected]
        if not ordered:
            ordered = [step for step in TEMP_SENSITIVITY_STEPS if step[0] == "baseline"]

        for key, label, _delta, color in ordered:
            values = pd.to_numeric(frame.get(key), errors="coerce")
            if values.notna().sum() == 0:
                continue
            max_value = values.max()
            fig.add_trace(go.Scatter(
                x=frame["DT"],
                y=values,
                mode="lines",
                name=f"{label} (Max: {max_value:,.1f})",
                line=dict(color=color, width=3.1 if key == "baseline" else 1.9),
            ))
            if highlight_max and values.notna().any():
                max_idx = values.idxmax()
                fig.add_trace(go.Scatter(
                    x=[frame.loc[max_idx, "DT"]],
                    y=[values.loc[max_idx]],
                    mode="markers",
                    marker=dict(color=color, size=10 if key == "baseline" else 8),
                    hovertemplate=f"{label}<br>%{{x|%Y-%m-%d %H:%M}}<br>%{{y:,.1f}} MWh<extra></extra>",
                    showlegend=False,
                ))

        if show_weather and "Temperature" in frame.columns:
            temperature = pd.to_numeric(frame["Temperature"], errors="coerce")
            if temperature.notna().any():
                fig.add_trace(go.Scatter(
                    x=frame["DT"],
                    y=temperature,
                    mode="lines",
                    name="Temperature",
                    line=dict(color="#9A6A3B", width=2.0),
                    yaxis="y2",
                ))
        fig.update_layout(
            title="Temperature Sensitivity",
            xaxis={"title": "Date / Hour", "showgrid": True, "gridcolor": "rgba(0,87,184,0.10)"},
            yaxis={"title": "MWh", "showgrid": True, "gridcolor": "rgba(0,87,184,0.10)"},
            yaxis2={"title": "Temperature (F)", "overlaying": "y", "side": "right", "showgrid": False},
            hovermode="x unified",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            margin={"l": 55, "r": 55, "t": 55, "b": 42},
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig

    def _make_comparable_days_graph(display: pd.DataFrame, hist: pd.DataFrame, start_d: date_type | None) -> go.Figure:
        # Simple comparable-days overlay: pick recent days with similar max temp and same DOW.
        fig = go.Figure()
        if display is None or display.empty or hist is None or hist.empty or not start_d:
            fig.update_layout(title="Comparable Days unavailable", plot_bgcolor="white", paper_bgcolor="white")
            return fig
        day_start = pd.Timestamp(start_d)
        if project_tz:
            try:
                day_start = day_start.tz_localize(project_tz)
            except Exception:
                pass
        day_end = day_start + pd.Timedelta(days=1)
        fut_day = display[(display["DT"] >= day_start) & (display["DT"] < day_end)].copy()
        fut_day["Temperature"] = pd.to_numeric(fut_day.get("Temperature"), errors="coerce")
        target_max = float(fut_day["Temperature"].max()) if "Temperature" in fut_day else np.nan
        target_dow = day_start.dayofweek

        hist_work = hist.copy()
        hist_work["DT"] = pd.to_datetime(hist_work["DT"], errors="coerce")
        hist_work["Date"] = hist_work["DT"].dt.normalize()
        hist_work["DOW"] = hist_work["DT"].dt.dayofweek
        hist_work["Temperature"] = pd.to_numeric(hist_work.get("Temperature"), errors="coerce")
        hist_work["MWH"] = pd.to_numeric(hist_work.get("MWH"), errors="coerce")
        daily = (
            hist_work.dropna(subset=["Date"])
            .groupby(["Date", "DOW"], as_index=False)
            .agg(MaxTemp=("Temperature", "max"))
        )
        daily = daily[daily["DOW"] == target_dow].dropna(subset=["MaxTemp"])
        if not np.isfinite(target_max) or daily.empty:
            fig.update_layout(title="Comparable Days (No target temperature)", plot_bgcolor="white", paper_bgcolor="white")
            return fig
        daily["TempDiff"] = (daily["MaxTemp"] - target_max).abs()
        candidates = daily.sort_values(["TempDiff", "Date"], ascending=[True, False]).head(3)["Date"].tolist()

        # Plot target forecast day (forecast + weather if present)
        if "Forecast" in fut_day:
            fig.add_trace(go.Scatter(x=fut_day["DT"], y=fut_day["Forecast"], mode="lines", name="Forecast", line=dict(color=BRAND_BLUE, width=2.8)))
        if "Actual" in fut_day:
            fig.add_trace(go.Scatter(x=fut_day["DT"], y=fut_day["Actual"], mode="lines", name="Actual", line=dict(color=BRAND_RED, width=2.1)))

        for d0 in candidates:
            d1 = pd.Timestamp(d0)
            d2 = d1 + pd.Timedelta(days=1)
            day = hist_work[(hist_work["DT"] >= d1) & (hist_work["DT"] < d2)].copy()
            if day.empty:
                continue
            fig.add_trace(go.Scatter(x=day["DT"], y=day["MWH"], mode="lines", name=str(d0), line=dict(width=1.5, dash="dot")))

        fig.update_layout(
            title=f"Comparable Days (MaxTemp ~ {target_max:.1f}F, DOW={target_dow})",
            xaxis={"title": "Date / Hour", "showgrid": True, "gridcolor": "rgba(0,87,184,0.10)"},
            yaxis={"title": "MWh", "showgrid": True, "gridcolor": "rgba(0,87,184,0.10)"},
            hovermode="x unified",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            margin={"l": 55, "r": 35, "t": 55, "b": 42},
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig

    @app.callback(
        Output("display-mode", "data"),
        Output("tab-spreadsheet", "style"),
        Output("tab-graph", "style"),
        Output("tab-dual", "style"),
        Output("tab-statistics", "style"),
        Input("tab-spreadsheet", "n_clicks"),
        Input("tab-graph", "n_clicks"),
        Input("tab-dual", "n_clicks"),
        Input("tab-statistics", "n_clicks"),
        State("display-mode", "data"),
    )
    def _set_tab(_a, _b, _c, _d, current):
        trig = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        mapping = {
            "tab-spreadsheet": "spreadsheet",
            "tab-graph": "graph",
            "tab-dual": "dual",
            "tab-statistics": "statistics",
        }
        active = mapping.get(trig, current or "dual")
        ss, gg, dd, st = _active_tab_styles(active)
        return active, ss, gg, dd, st

    @app.callback(
        Output("start-date", "date"),
        Input("nav-prev-day", "n_clicks"),
        Input("nav-next-day", "n_clicks"),
        Input("nav-prev-horizon", "n_clicks"),
        Input("nav-next-horizon", "n_clicks"),
        State("start-date", "date"),
        State("horizon-days", "value"),
        State("available-days", "data"),
    )
    def _nav_dates(_p, _n, _ph, _nh, start_date, horizon_days, available_days_store):
        trig = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        d0 = _parse_date(start_date) or default_start_date
        if not d0:
            return start_date
        step = 1
        if trig in {"nav-prev-horizon", "nav-next-horizon"}:
            step = int(horizon_days or 16)
        if trig in {"nav-prev-day", "nav-prev-horizon"}:
            d0 = d0 - pd.Timedelta(days=step)
        if trig in {"nav-next-day", "nav-next-horizon"}:
            d0 = d0 + pd.Timedelta(days=step)
        avail = [pd.to_datetime(x).date() for x in (available_days_store or [])]
        d0 = _clamp_to_available(d0, avail)
        return d0.isoformat()

    @app.callback(
        Output("dual-container", "style"),
        Output("stats-container", "style"),
        Output("aux-graph-container", "style"),
        Output("main-graph", "figure"),
        Output("stats-graph", "figure"),
        Output("stats-detail-graph", "figure"),
        Output("aux-graph", "figure"),
        Output("forecast-table", "data"),
        Output("forecast-table", "columns"),
        Output("metric-cards", "children"),
        Input("display-mode", "data"),
        Input("model-type", "value"),
        Input("weather-variable", "value"),
        Input("graph-options", "value"),
        Input("temperature-steps", "value"),
        Input("historical-options", "value"),
        Input("start-date", "date"),
        Input("horizon-days", "value"),
        Input("spreadsheet-mode", "value"),
        Input("custom-columns-store", "data"),
    )
    def update_dashboard(
        display_mode: str,
        model_type: str,
        weather_variable: str,
        graph_options: list[str] | None,
        temperature_steps: list[str] | None,
        historical_options: list[str] | None,
        start_date,
        horizon_days: int,
        spreadsheet_mode: str,
        custom_cols,
    ):
        # View-mode styles
        dual_style = {
            "display": "grid",
            "gridTemplateColumns": "minmax(520px, 3fr) minmax(360px, 2fr)",
            "gap": "12px",
            "height": "100%",
        }
        stats_style = {"display": "none", "height": "100%"}
        aux_style = {"display": "none", "height": "100%"}

        history_hours = 72  # v11 uses a start-date anchor; keep a small history context.
        start_d = _parse_date(start_date)
        df = _window(display_df, start_d, int(horizon_days or 16), history_hours)
        cards = _cards(display_df, backtest_df, int(horizon_days or 16), diagnostics_results)

        # Table always reflects the selected window, filtered by spreadsheet-mode.
        data, columns = _table_data(df)

        diagnostics_mode = custom_cols == "__ALL_DIAGNOSTICS__"

        # Spreadsheet-mode column shaping. Full-weather defaults to the operator view;
        # diagnostics mode exposes all generated stage/weather/debug columns.
        if not diagnostics_mode and columns:
            # Column ids are from the renamed table output (strings).
            keep = set([c["id"] for c in columns])
            base_keep = {"DT", "Forecast", "Actual", "Raw XGB+LGB", "XGB", "LGB", "Prophet", "Low", "Expected", "High", "Cal Level"}
            if spreadsheet_mode == "load_only":
                keep = {"DT", "Forecast", "Actual", "Low", "Expected", "High"}
            elif spreadsheet_mode == "no_weather":
                keep = base_keep
            elif spreadsheet_mode == "temperature_only":
                keep = base_keep | {"Temp", "Daily Max"}
            else:
                keep = set(OPERATOR_TABLE_COLUMNS)
            columns = [c for c in columns if c["id"] in keep]
            data = [{k: row.get(k, "") for k in [c["id"] for c in columns]} for row in data]

        # Custom spreadsheet column selection overrides the above.
        if isinstance(custom_cols, list) and columns:
            keep = set([str(x) for x in custom_cols])
            columns = [c for c in columns if c["id"] in keep]
            data = [{k: row.get(k, "") for k in [c["id"] for c in columns]} for row in data]

        # Decide which graph to render based on model-type.
        main_fig = go.Figure()
        aux_fig = go.Figure()
        if model_type == "temp_sens":
            sensitivity = _make_temp_sensitivity_frame(df, hist_df)
            show_weather = True if (graph_options and "show_weather" in graph_options) else False
            highlight_max = True if (graph_options and "highlight_max" in graph_options) else False
            aux_fig = _make_temp_sensitivity_graph(sensitivity, temperature_steps, show_weather, highlight_max)
            data, columns = _temp_sensitivity_table_data(sensitivity, temperature_steps)
        elif model_type == "comparable":
            aux_fig = _make_comparable_days_graph(display_df, hist_df, start_d)
        else:
            # Baseline.
            show_weather = True if (graph_options and "show_weather" in graph_options) else False
            wvar = weather_variable if show_weather else "none"
            main_fig = _make_forecast_graph(df, wvar)

        if isinstance((diagnostics_results or {}).get("production_readiness_scorecard"), pd.DataFrame):
            stats_fig = _make_production_scorecard_graph(diagnostics_results)
            stats_detail = _make_validation_detail_graph(backtest_df)
        else:
            stats_fig = _make_backtest_graph(backtest_df)
            stats_detail = _make_validation_detail_graph(backtest_df)

        # Apply tab mode visibility.
        is_aux = model_type in {"comparable", "temp_sens"}

        if display_mode == "statistics":
            dual_style = {"display": "none"}
            stats_style = {"display": "block", "height": "100%"}
            aux_style = {"display": "none"}
        elif display_mode == "graph":
            if is_aux:
                dual_style = {"display": "none"}
                aux_style = {"display": "block", "height": "100%"}
            else:
                dual_style = {"display": "grid", "gridTemplateColumns": "100% 0%", "gap": "0px", "height": "100%"}
                aux_style = {"display": "none"}
            stats_style = {"display": "none"}
        elif display_mode == "spreadsheet":
            dual_style = {"display": "grid", "gridTemplateColumns": "0% 100%", "gap": "0px", "height": "100%"}
            stats_style = {"display": "none"}
            aux_style = {"display": "none"}
        else:
            # dual
            if is_aux:
                dual_style = {
                    "display": "grid",
                    "gridTemplateColumns": "minmax(520px, 3fr) minmax(360px, 2fr)",
                    "gap": "12px",
                    "height": "100%",
                }
                aux_style = {"display": "none"}
                # In dual mode, reuse the main graph slot for aux views.
                main_fig = aux_fig
                aux_fig = go.Figure()
            else:
                dual_style = {
                    "display": "grid",
                    "gridTemplateColumns": "minmax(520px, 3fr) minmax(360px, 2fr)",
                    "gap": "12px",
                    "height": "100%",
                }
                aux_style = {"display": "none"}
            stats_style = {"display": "none"}
        # Spreadsheet mode keeps the table visible; main_fig is still computed but hidden.

        return dual_style, stats_style, aux_style, main_fig, stats_fig, stats_detail, aux_fig, data, columns, cards

    @app.callback(
        Output("download-spreadsheet", "data"),
        Input("export-spreadsheet", "n_clicks"),
        State("forecast-table", "data"),
        State("forecast-table", "columns"),
        prevent_initial_call=True,
    )
    def _export_spreadsheet(n, rows, cols):
        if not n or not rows or not cols:
            return None
        df = pd.DataFrame(rows)
        return dcc.send_data_frame(df.to_csv, "forecast_spreadsheet.csv", index=False)

    @app.callback(
        Output("custom-columns-store", "data"),
        Input("custom-spreadsheet", "n_clicks"),
        State("forecast-table", "columns"),
        State("custom-columns-store", "data"),
        prevent_initial_call=True,
    )
    def _custom_spreadsheet(n, cols, current):
        # Toggle the table between the operator default and the full diagnostics column set.
        if not n:
            return None
        if current:
            return None
        return "__ALL_DIAGNOSTICS__"

    return app


from forecasting.dashboard.dashboard_app import create_dashboard_app as create_dashboard_app
