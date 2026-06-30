from __future__ import annotations
import pandas as pd

def load_solar_forecast(config: dict) -> pd.DataFrame:
    """Load the solar forecast from the CSV file."""
    output_dir = config.get("project", {}).get("output_dir", "forecast_outputs")
    solar_forecast_path = f"{output_dir}/roseville_solar_forecast_hourly.csv"
    
    try:
        solar_df = pd.read_csv(solar_forecast_path)
        solar_df["DT"] = pd.to_datetime(solar_df["IntervalStartDT"])
        solar_df.rename(columns={"Forecast_MW": "Solar_Forecast_MW"}, inplace=True)
        return solar_df
    except FileNotFoundError:
        print(f"Solar forecast file not found at {solar_forecast_path}. Returning empty DataFrame.")
        return pd.DataFrame()