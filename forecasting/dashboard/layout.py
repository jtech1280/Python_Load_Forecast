from __future__ import annotations

from dash import html, dcc, dash_table


BRAND_BLUE = "#0057B8"
INK = "#18212F"
FONT_FAMILY = "'Aptos', 'Segoe UI', sans-serif"
PANEL_BG = "#F2F2F2"
PANEL_BORDER = "#C8C8C8"
TAB_BG = "#FAFAFA"
TAB_BORDER = "#D2D2D2"


def _section_header(text: str):
    return html.Div(
        text,
        style={
            "padding": "8px 7px",
            "fontSize": "14px",
            "fontWeight": "700",
            "borderBottom": "1px solid #D0D0D0",
        },
    )


def _sidebar_header(text: str):
    return html.Div(text, style={"fontWeight": "700"})


def _sidebar_label(text: str):
    return html.Div(text, style={"fontSize": "12px", "margin": "6px 0 4px 0", "color": "#333"})


def _small_button(text: str, btn_id: str):
    return html.Button(
        text,
        id=btn_id,
        n_clicks=0,
        style={
            "height": "24px",
            "width": "34px",
            "border": "1px solid #BDBDBD",
            "backgroundColor": "#FFFFFF",
            "margin": "0 3px",
            "cursor": "pointer",
        },
    )


def _tab_button(text: str, btn_id: str):
    # Style is applied dynamically in a callback.
    return html.Button(
        text,
        id=btn_id,
        n_clicks=0,
        style={
            "border": "none",
            "background": "transparent",
            "color": BRAND_BLUE,
            "padding": "10px 14px",
            "cursor": "pointer",
            "fontSize": "13px",
        },
    )


def _series_group_controls(series_options=None, series_default=None):
    options = list(series_options or [])
    if not options:
        return html.Div(
            "No forecast series available.",
            style={"fontSize": "12px", "color": "#555", "marginBottom": "6px"},
        )

    default_values = set(series_default or [])
    groups = []
    for option in options:
        group = option.get("group") or "Other"
        if group not in groups:
            groups.append(group)

    children = []
    for group in groups:
        group_options = [
            {"label": option["label"], "value": option["value"]}
            for option in options
            if (option.get("group") or "Other") == group
        ]
        group_values = [option["value"] for option in group_options if option["value"] in default_values]
        children.append(
            html.Div(
                style={"marginBottom": "8px"},
                children=[
                    html.Div(group, style={"fontSize": "11px", "fontWeight": "700", "color": "#333", "margin": "6px 0 3px 0"}),
                    dcc.Checklist(
                        id={"type": "series-group", "group": group},
                        options=group_options,
                        value=group_values,
                        labelStyle={"display": "block", "fontSize": "12px", "margin": "2px 0"},
                        inputStyle={"marginRight": "5px"},
                    ),
                ],
            )
        )
    return html.Div(children=children)


def make_layout(
    min_date=None,
    max_date=None,
    default_start_date=None,
    available_days=None,
    series_options=None,
    series_default=None,
):
    # Use dcc.Store for the right-side view mode (mirrors v11.6 tab experience).
    series_controls = _series_group_controls(series_options, series_default) if series_options is not None else None
    graph_option_children = []
    if series_controls is not None:
        graph_option_children.extend([_sidebar_label("Forecast Series"), series_controls])
    graph_option_children.extend(
        [
            dcc.Checklist(
                id="graph-options",
                options=[
                    {"label": "Highlight Max Values", "value": "highlight_max"},
                    {"label": "Show Weather", "value": "show_weather"},
                ],
                value=["highlight_max", "show_weather"],
                labelStyle={"display": "block", "fontSize": "12px", "margin": "4px 0"},
                inputStyle={"marginRight": "5px"},
            ),
            _sidebar_label("Weather Variable to Plot"),
            dcc.Dropdown(
                id="weather-variable",
                options=[
                    {"label": "tempf", "value": "Temperature"},
                    {"label": "daily max", "value": "Temperature_DailyMax"},
                    {"label": "humid", "value": "Humidity_Norm"},
                    {"label": "cloud", "value": "CloudCover_Norm"},
                    {"label": "wind", "value": "WindSpeed_Mph"},
                    {"label": "rain", "value": "PrecipIn"},
                    {"label": "solar", "value": "Solar_Irradiance"},
                    {"label": "btm solar", "value": "BTM_Solar_Proxy_MW"},
                ],
                value="Temperature",
                clearable=False,
                style={"fontSize": "12px"},
            ),
            _sidebar_label("Temperature Steps"),
            dcc.Checklist(
                id="temperature-steps",
                options=[
                    {"label": f"Plus {v}", "value": f"plus_{v}"}
                    for v in range(7, 0, -1)
                ]
                + [{"label": "Baseline", "value": "baseline"}]
                + [
                    {"label": f"Minus {v}", "value": f"minus_{v}"}
                    for v in range(1, 8)
                ],
                value=[f"plus_{v}" for v in range(7, 0, -1)]
                + ["baseline"]
                + [f"minus_{v}" for v in range(1, 8)],
                labelStyle={"display": "block", "fontSize": "12px", "margin": "3px 0"},
                inputStyle={"marginRight": "5px"},
            ),
        ]
    )
    return html.Div(
        style={
            "fontFamily": FONT_FAMILY,
            "fontSize": "13px",
            "backgroundColor": "#FFFFFF",
            "minHeight": "100vh",
            "overflow": "hidden",
            "display": "flex",
            "margin": "0",
            "color": INK,
        },
        children=[
            dcc.Store(id="display-mode", data="dual"),
            dcc.Store(id="available-days", data=[str(d) for d in (available_days or [])]),
            dcc.Store(id="custom-columns-store", data=None),
            dcc.Download(id="download-spreadsheet"),
            # LEFT CONTROL PANEL
            html.Div(
                style={
                    "width": "225px",
                    "minWidth": "225px",
                    "backgroundColor": PANEL_BG,
                    "borderRight": f"1px solid {PANEL_BORDER}",
                    "height": "100vh",
                    "overflowY": "auto",
                },
                children=[
                    _section_header("Model"),
                    html.Div(
                        style={"padding": "8px 6px"},
                        children=[
                            dcc.Checklist(
                                id="allow-multiple",
                                options=[{"label": "Allow Multiple Model Selection", "value": "allow"}],
                                value=[],
                                labelStyle={"display": "block", "fontSize": "12px"},
                                inputStyle={"marginRight": "4px"},
                            ),
                            dcc.Dropdown(
                                id="model-dropdown",
                                options=[{"label": "Roseville", "value": "Roseville"}],
                                value="Roseville",
                                clearable=False,
                                style={"fontSize": "12px", "marginTop": "8px"},
                            ),
                            dcc.RadioItems(
                                id="model-type",
                                options=[
                                    {"label": "Baseline Forecast", "value": "baseline"},
                                    {"label": "Comparable Days", "value": "comparable"},
                                    {"label": "Temperature Sensitivity", "value": "temp_sens"},
                                ],
                                value="baseline",
                                labelStyle={"display": "block", "fontSize": "12px", "margin": "4px 0"},
                                inputStyle={"marginRight": "5px"},
                                style={"marginTop": "8px"},
                            ),
                            dcc.Checklist(
                                id="stage-checklist",
                                options=[{"label": "Stage 2", "value": "stage2"}],
                                value=["stage2"],
                                labelStyle={"display": "block", "fontSize": "12px"},
                                inputStyle={"marginRight": "5px"},
                            ),
                            html.Button(
                                "Refresh",
                                id="refresh-button",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "#2D9CDB",
                                    "color": "#FFFFFF",
                                    "border": "1px solid #1E88C8",
                                    "borderRadius": "2px",
                                    "padding": "6px 18px",
                                    "fontSize": "12px",
                                    "marginTop": "6px",
                                    "cursor": "pointer",
                                },
                            ),
                            html.Div(
                                id="metric-cards",
                                style={"marginTop": "10px"},
                            ),
                        ],
                    ),
                    _section_header("Date Selection"),
                    html.Div(
                        style={"padding": "7px 6px"},
                        children=[
                            _sidebar_label("Starting Date"),
                            dcc.DatePickerSingle(
                                id="start-date",
                                min_date_allowed=min_date,
                                max_date_allowed=max_date,
                                date=default_start_date,
                                display_format="YYYY-MM-DD",
                                style={"fontSize": "12px", "width": "100%"},
                            ),
                            _sidebar_label("Days to Display (Horizon)"),
                            dcc.Dropdown(
                                id="horizon-days",
                                options=[{"label": str(v), "value": v} for v in [1, 2, 3, 7, 10, 14, 16]],
                                value=16,
                                clearable=False,
                                searchable=False,
                                style={"fontSize": "12px"},
                            ),
                            html.Div(
                                style={"textAlign": "center", "marginTop": "8px"},
                                children=[
                                    _small_button("<<", "nav-prev-horizon"),
                                    _small_button("<", "nav-prev-day"),
                                    _small_button(">", "nav-next-day"),
                                    _small_button(">>", "nav-next-horizon"),
                                ],
                            ),
                        ],
                    ),
                    _section_header("Spreadsheet Options"),
                    html.Div(
                        style={"padding": "7px 6px"},
                        children=[
                            dcc.RadioItems(
                                id="spreadsheet-mode",
                                options=[
                                    {"label": "Load Only Display", "value": "load_only"},
                                    {"label": "No Weather", "value": "no_weather"},
                                    {"label": "Temperature Only", "value": "temperature_only"},
                                    {"label": "Full Weather", "value": "full_weather"},
                                ],
                                value="full_weather",
                                labelStyle={"display": "block", "fontSize": "12px", "margin": "4px 0"},
                                inputStyle={"marginRight": "5px"},
                            ),
                            html.Button(
                                "Export Spreadsheet",
                                id="export-spreadsheet",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "#2D9CDB",
                                    "color": "#FFFFFF",
                                    "border": "1px solid #1E88C8",
                                    "borderRadius": "2px",
                                    "padding": "6px 13px",
                                    "fontSize": "12px",
                                    "margin": "6px 0 4px 30px",
                                    "cursor": "pointer",
                                },
                            ),
                            html.Button(
                                "Toggle Diagnostics",
                                id="custom-spreadsheet",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "#2D9CDB",
                                    "color": "#FFFFFF",
                                    "border": "1px solid #1E88C8",
                                    "borderRadius": "2px",
                                    "padding": "6px 10px",
                                    "fontSize": "12px",
                                    "marginLeft": "30px",
                                    "cursor": "pointer",
                                },
                            ),
                        ],
                    ),
                    _section_header("Graph Options"),
                    html.Div(
                        style={"padding": "7px 6px"},
                        children=graph_option_children,
                    ),
                    _section_header("Historical Data"),
                    html.Div(
                        style={"padding": "7px 6px"},
                        children=[
                            dcc.Checklist(
                                id="historical-options",
                                options=[
                                    {"label": "Show Historic Load Fit", "value": "historic_load"},
                                    {"label": "Show Historic Weather", "value": "historic_weather"},
                                ],
                                value=["historic_load"],
                                labelStyle={"display": "block", "fontSize": "12px", "margin": "4px 0"},
                                inputStyle={"marginRight": "5px"},
                            )
                        ],
                    ),
                ],
            ),
            # RIGHT CONTENT AREA
            html.Div(
                style={
                    "flex": "1",
                    "height": "100vh",
                    "overflow": "hidden",
                    "display": "flex",
                    "flexDirection": "column",
                    "backgroundColor": "#FFFFFF",
                },
                children=[
                    # TOP TAB BAR
                    html.Div(
                        style={
                            "height": "38px",
                            "borderBottom": f"1px solid {TAB_BORDER}",
                            "display": "flex",
                            "alignItems": "flex-end",
                            "paddingLeft": "12px",
                            "backgroundColor": TAB_BG,
                        },
                        children=[
                            _tab_button("Spreadsheet", "tab-spreadsheet"),
                            _tab_button("Graph", "tab-graph"),
                            _tab_button("Dual Display", "tab-dual"),
                            _tab_button("Scorecard", "tab-statistics"),
                        ],
                    ),
                    html.Div(
                        id="main-content",
                        style={
                            "flex": "1",
                            "overflow": "hidden",
                            "padding": "8px 10px 10px 10px",
                        },
                        children=[
                            # Forecast: dual/grid view. We toggle column widths and visibility via styles.
                            html.Div(
                                id="dual-container",
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "minmax(520px, 3fr) minmax(360px, 2fr)",
                                    "gap": "12px",
                                    "height": "100%",
                                },
                                children=[
                                    html.Div(
                                        id="graph-pane",
                                        style={"minWidth": 0},
                                        children=[
                                            dcc.Graph(id="main-graph", config={"displayModeBar": True}, style={"height": "calc(100vh - 170px)"}),
                                        ],
                                    ),
                                    html.Div(
                                        id="table-pane",
                                        style={"minWidth": 0},
                                        children=[
                                            html.Div(
                                                "Operator Forecast Table",
                                                style={"margin": "0 0 8px 0", "color": BRAND_BLUE, "fontWeight": "700"},
                                            ),
                                            make_table(),
                                        ],
                                    ),
                                ],
                            ),
                            # Statistics view.
                            html.Div(
                                id="stats-container",
                                style={"display": "none", "height": "100%"},
                                children=[
                                    dcc.Graph(id="stats-graph", config={"displayModeBar": True}, style={"height": "calc(100vh - 170px)"}),
                                    dcc.Graph(id="stats-detail-graph", config={"displayModeBar": True}, style={"height": "320px"}),
                                ],
                            ),
                            # Sensitivity / comparable view (graph-only).
                            html.Div(
                                id="aux-graph-container",
                                style={"display": "none", "height": "100%"},
                                children=[
                                    dcc.Graph(id="aux-graph", config={"displayModeBar": True}, style={"height": "calc(100vh - 170px)"}),
                                ],
                            ),
                        ],
                    ),
                    # Hidden modal container for custom spreadsheet column selection.
                    html.Div(
                        id="custom-spreadsheet-modal",
                        style={"display": "none"},
                    ),
                ],
            ),
        ],
    )


def make_table():
    # This DataTable is re-used across display modes.
    return dash_table.DataTable(
        id="forecast-table",
        page_size=24,
        sort_action="native",
        filter_action="native",
        fixed_rows={"headers": True},
        style_table={"height": "calc(100vh - 185px)", "overflowY": "auto", "overflowX": "auto", "border": "1px solid #B7D7F5"},
        style_header={"backgroundColor": BRAND_BLUE, "color": "white", "fontWeight": "bold", "fontSize": "12px"},
        style_cell={"fontFamily": FONT_FAMILY, "fontSize": "12px", "padding": "6px", "textAlign": "right", "whiteSpace": "normal"},
        style_cell_conditional=[{"if": {"column_id": "DT"}, "textAlign": "left", "minWidth": "130px"}],
        style_data_conditional=[
            {"if": {"column_id": "Forecast"}, "color": BRAND_BLUE, "fontWeight": "bold"},
            {"if": {"column_id": "Actual"}, "color": "#D62728", "fontWeight": "bold"},
            {"if": {"filter_query": "{Risk Code} != \"NORMAL\"", "column_id": "Risk Code"}, "color": "#D62728", "fontWeight": "bold"},
            {"if": {"filter_query": "{Scenario Cap} = \"Yes\"", "column_id": "Scenario Cap"}, "color": "#D62728", "fontWeight": "bold"},
            {"if": {"filter_query": "{Residual Cal} > 0", "column_id": "Residual Cal"}, "color": "#7B2CBF"},
            {"if": {"filter_query": "{Residual Cal} < 0", "column_id": "Residual Cal"}, "color": "#F2A900"},
        ],
    )
