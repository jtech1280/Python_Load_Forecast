/*
    Creates the SQL Server tables used by forecasting.data.output_sql_store.

    Run this in the database targeted by the local ODBC DSN named Forecast_DB.
    The Python config uses:
        output_sql.dsn_name: Forecast_DB
        output_sql.schema: Forecasting
        output_sql.enabled: true

    After this script has been run, normal forecasting.main runs will insert
    run metadata, forecast output rows, backtest rows, weather rows, and rolling
    replay diagnostic rows by default.

    Rolling replay diagnostic tables are created with RunID metadata columns here.
    The Python writer adds the current diagnostic columns automatically on insert.
*/

SET NOCOUNT ON;

IF SCHEMA_ID(N'Forecasting') IS NULL
    EXEC(N'CREATE SCHEMA [Forecasting]');
GO

IF OBJECT_ID(N'[Forecasting].[LoadForecastRun]', N'U') IS NULL
BEGIN
    CREATE TABLE [Forecasting].[LoadForecastRun] (
        [RunID] UNIQUEIDENTIFIER NOT NULL CONSTRAINT [PK_LoadForecastRun_RunID] PRIMARY KEY,
        [RunStartedAtUTC] DATETIME2(7) NOT NULL,
        [InsertedAtUTC] DATETIME2(7) NOT NULL CONSTRAINT [DF_LoadForecastRun_InsertedAtUTC] DEFAULT SYSUTCDATETIME(),
        [ProjectName] NVARCHAR(256) NULL,
        [Source] NVARCHAR(128) NULL,
        [ForecastRows] INT NULL,
        [BacktestRows] INT NULL,
        [WeatherRows] INT NULL,
        [FirstOutputDT] DATETIMEOFFSET(7) NULL,
        [LastOutputDT] DATETIMEOFFSET(7) NULL,
        [ContentHash] NVARCHAR(64) NULL,
        [MetadataJson] NVARCHAR(MAX) NULL
    );
END;
GO

IF OBJECT_ID(N'[Forecasting].[LoadForecastOutput]', N'U') IS NULL
BEGIN
    CREATE TABLE [Forecasting].[LoadForecastOutput] (
        [OutputRowID] BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT [PK_LoadForecastOutput_OutputRowID] PRIMARY KEY,
        [RunID] UNIQUEIDENTIFIER NOT NULL,
        [InsertedAtUTC] DATETIME2(7) NOT NULL CONSTRAINT [DF_LoadForecastOutput_InsertedAtUTC] DEFAULT SYSUTCDATETIME(),
        [DT] DATETIMEOFFSET(7) NULL,
        [Actual] FLOAT NULL,
        [Load_Source] NVARCHAR(MAX) NULL,
        [FiveMin_Interval_Count] FLOAT NULL,
        [FiveMin_Hourly_Last_MW] FLOAT NULL,
        [FiveMin_Hourly_Range_MW] FLOAT NULL,
        [Temperature] FLOAT NULL,
        [Temperature_DailyMax] FLOAT NULL,
        [BTM_Solar_Proxy_MW] FLOAT NULL,
        [CloudCover_Norm] FLOAT NULL,
        [Humidity_Norm] FLOAT NULL,
        [WindSpeed_Mph] FLOAT NULL,
        [PrecipIn] FLOAT NULL,
        [Forecast] FLOAT NULL,
        [Raw_Forecast_MWH] FLOAT NULL,
        [Upper_Band] FLOAT NULL,
        [Lower_Band] FLOAT NULL,
        [Band] FLOAT NULL,
        [P10_Forecast_MWH] FLOAT NULL,
        [P50_Forecast_MWH] FLOAT NULL,
        [P90_Forecast_MWH] FLOAT NULL,
        [Forecast_Low_MWH] FLOAT NULL,
        [Forecast_Expected_MWH] FLOAT NULL,
        [Forecast_High_MWH] FLOAT NULL,
        [Weather_Input_Risk_Multiplier] FLOAT NULL,
        [Weather_Input_Risk_Reason] NVARCHAR(MAX) NULL,
        [Weather_Input_Risk_Class] NVARCHAR(MAX) NULL,
        [Production_Caution_Flag] FLOAT NULL,
        [Production_Caution_Reason] NVARCHAR(MAX) NULL,
        [Production_Confidence_Label] NVARCHAR(MAX) NULL,
        [Production_Risk_Code] NVARCHAR(MAX) NULL,
        [Calibration_Level] NVARCHAR(MAX) NULL,
        [Calibrated_Forecast_MWH] FLOAT NULL,
        [Targeted_Meta_Adjusted_Forecast_MWH] FLOAT NULL,
        [Residual_Calibrated_Forecast_MWH] FLOAT NULL,
        [Warm_Ramp_Adjusted_Forecast_MWH] FLOAT NULL,
        [Cloud_Solar_Adjusted_Forecast_MWH] FLOAT NULL,
        [Peak_Risk_Adjusted_Forecast_MWH] FLOAT NULL,
        [Recent_Corrected_Forecast_MWH] FLOAT NULL,
        [XGB_Pred_MWH] FLOAT NULL,
        [LGB_Pred_MWH] FLOAT NULL,
        [CatBoost_Pred_MWH] FLOAT NULL,
        [Prophet_Pred_MWH] FLOAT NULL,
        [Prophet_Lower_MWH] FLOAT NULL,
        [Prophet_Upper_MWH] FLOAT NULL,
        [Targeted_Meta_Bias_Cal_MWH] FLOAT NULL,
        [Targeted_Meta_SolarCloud_Cal_MWH] FLOAT NULL,
        [Targeted_Meta_Cal_MWH] FLOAT NULL,
        [Residual_Cal_MWH] FLOAT NULL,
        [Warm_Ramp_Cal_MWH] FLOAT NULL,
        [Cloud_Solar_Shape_Cal_MWH] FLOAT NULL,
        [Cloud_Solar_Shape_Raw_Cal_MWH] FLOAT NULL,
        [Peak_Risk_Cal_MWH] FLOAT NULL,
        [Recent_Level_Correction_MWH] FLOAT NULL,
        [Band_Method] NVARCHAR(MAX) NULL,
        [Quantile_Method] NVARCHAR(MAX) NULL,
        [Operational_Horizon_Label] NVARCHAR(MAX) NULL,
        [Pre_Conformal_Band_MWH] FLOAT NULL,
        [Conformal_Weather_Band_MWH] FLOAT NULL,
        [Conformal_Weather_Source] NVARCHAR(MAX) NULL,
        [WeatherScenario_Min_P50_MWH] FLOAT NULL,
        [WeatherScenario_Max_P50_MWH] FLOAT NULL,
        [WeatherScenario_Spread_MWH] FLOAT NULL,
        [WeatherScenario_HalfSpread_MWH] FLOAT NULL,
        [WeatherScenario_MaxAbsDelta_MWH] FLOAT NULL,
        [WeatherScenario_Cap_Applied] FLOAT NULL,
        [Weather_Robustness_Hedge_MWH] FLOAT NULL,
        [Weather_Robustness_Hedge_Source] NVARCHAR(MAX) NULL,
        [Weather_Robustness_Jensen_MWH] FLOAT NULL,
        [Weather_Robustness_Upper_MWH] FLOAT NULL,
        [Weather_Robustness_Warmer_Delta_MWH] FLOAT NULL,
        [Weather_Robustness_Temp_Sigma_F] FLOAT NULL,
        [Weather_Robustness_Temp_Bias_Damping] FLOAT NULL,
        [Weather_Robustness_Gate] FLOAT NULL,
        [Focused_Scorecard_Guard_MWH] FLOAT NULL,
        [Focused_Scorecard_Guard_Source] NVARCHAR(MAX) NULL,
        [Calibration_Matched_Levels] NVARCHAR(MAX) NULL,
        [Targeted_Meta_Source] NVARCHAR(MAX) NULL,
        [Warm_Ramp_Correction_Source] NVARCHAR(MAX) NULL,
        [Cloud_Solar_Correction_Source] NVARCHAR(MAX) NULL,
        [Peak_Risk_Source] NVARCHAR(MAX) NULL,
        [Recent_Correction_Source] NVARCHAR(MAX) NULL,
        [Long_Horizon_Peak_Month_Correction_MWH] FLOAT NULL,
        [Long_Horizon_Hot_Month_Correction_MWH] FLOAT NULL,
        [Stage_Selected_Forecast_MWH] FLOAT NULL,
        [Stage_Selector_Source] NVARCHAR(MAX) NULL,
        [Stage_Selector_Reason] NVARCHAR(MAX) NULL,
        [FiveMin_Load_Available] FLOAT NULL,
        [FiveMin_Data_Age_Hours] FLOAT NULL,
        [FiveMin_PrevHour_Avg_MW] FLOAT NULL,
        [FiveMin_PrevHour_Max_MW] FLOAT NULL,
        [FiveMin_PrevHour_Min_MW] FLOAT NULL,
        [FiveMin_PrevHour_Last_MW] FLOAT NULL,
        [FiveMin_PrevHour_Range_MW] FLOAT NULL,
        [FiveMin_PrevHour_Ramp_MW] FLOAT NULL,
        [FiveMin_PrevHour_Count] FLOAT NULL,
        [FiveMin_Ramp_15Min_MW] FLOAT NULL,
        [FiveMin_Ramp_30Min_MW] FLOAT NULL,
        [FiveMin_Ramp_60Min_MW] FLOAT NULL,
        [BTM_Solar_Loss_From_ClearSky_MW] FLOAT NULL,
        [Midday_Overcast_Solar_Loss_MW] FLOAT NULL,
        [CloudSolarEventClass] NVARCHAR(MAX) NULL,
        [CloudSolarEventMultiplier] FLOAT NULL,
        [CloudSolarBaseBucket] NVARCHAR(MAX) NULL,
        [Daily_BTM_Solar_Proxy_Max_MW] FLOAT NULL,
        [Daily_BTM_Solar_Loss_Max_MW] FLOAT NULL,
        [Solar_Irradiance] FLOAT NULL,
        [ClearSky_Index] FLOAT NULL,
        [WeatherScenario_warmer_P50_MWH] FLOAT NULL,
        [WeatherScenario_hot_stress_5f_P50_MWH] FLOAT NULL,
        [WeatherScenario_cooler_P50_MWH] FLOAT NULL,
        [WeatherScenario_cloudier_solar_loss_P50_MWH] FLOAT NULL,
        [WeatherScenario_severe_cloud_solar_loss_P50_MWH] FLOAT NULL,
        [WeatherScenario_clearer_high_solar_P50_MWH] FLOAT NULL,
        CONSTRAINT [FK_LoadForecastOutput_Run] FOREIGN KEY ([RunID])
            REFERENCES [Forecasting].[LoadForecastRun] ([RunID])
    );
END;
GO

IF OBJECT_ID(N'[Forecasting].[LoadForecastBacktest]', N'U') IS NULL
BEGIN
    CREATE TABLE [Forecasting].[LoadForecastBacktest] (
        [OutputRowID] BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT [PK_LoadForecastBacktest_OutputRowID] PRIMARY KEY,
        [RunID] UNIQUEIDENTIFIER NOT NULL,
        [InsertedAtUTC] DATETIME2(7) NOT NULL CONSTRAINT [DF_LoadForecastBacktest_InsertedAtUTC] DEFAULT SYSUTCDATETIME(),
        [DT] DATETIMEOFFSET(7) NULL,
        [Actual_MWH] FLOAT NULL,
        [Season] NVARCHAR(MAX) NULL,
        [Month] BIGINT NULL,
        [Hour] BIGINT NULL,
        [HourGroup] NVARCHAR(MAX) NULL,
        [DOW] BIGINT NULL,
        [IsWeekend] BIGINT NULL,
        [IsHoliday] BIGINT NULL,
        [IsLikelySystemPeakHour] BIGINT NULL,
        [Temperature] FLOAT NULL,
        [Temperature_DailyMax] FLOAT NULL,
        [DailyMaxTempBin] FLOAT NULL,
        [BTM_Solar_Proxy_MW] FLOAT NULL,
        [BTM_Solar_Loss_From_ClearSky_MW] FLOAT NULL,
        [Midday_Overcast_Solar_Loss_MW] FLOAT NULL,
        [ClearSky_Index] FLOAT NULL,
        [CloudCover_Norm] FLOAT NULL,
        [Humidity_Norm] FLOAT NULL,
        [WindSpeed_Mph] FLOAT NULL,
        [PrecipIn] FLOAT NULL,
        [Raw_Forecast_MWH] FLOAT NULL,
        [XGB_Pred_MWH] FLOAT NULL,
        [LGB_Pred_MWH] FLOAT NULL,
        [CatBoost_Pred_MWH] FLOAT NULL,
        [Prophet_Pred_MWH] FLOAT NULL,
        [Prophet_Lower_MWH] FLOAT NULL,
        [Prophet_Upper_MWH] FLOAT NULL,
        [Residual_MWH] FLOAT NULL,
        [AbsError_MWH] FLOAT NULL,
        [APE] FLOAT NULL,
        [Targeted_Meta_Bias_Cal_MWH] FLOAT NULL,
        [Targeted_Meta_SolarCloud_Cal_MWH] FLOAT NULL,
        [Targeted_Meta_Cal_MWH] FLOAT NULL,
        [Targeted_Meta_Source] NVARCHAR(MAX) NULL,
        [Targeted_Meta_Adjusted_Forecast_MWH] FLOAT NULL,
        [CloudCoverBin] FLOAT NULL,
        [CloudCoverBucket] NVARCHAR(MAX) NULL,
        [BTMSolarProxyBin] FLOAT NULL,
        [SolarLossBucket] NVARCHAR(MAX) NULL,
        [Residual_Cal_MWH] FLOAT NULL,
        [Calibration_Level] NVARCHAR(MAX) NULL,
        [Calibration_Matched_Levels] NVARCHAR(MAX) NULL,
        [Residual_Calibrated_Forecast_MWH] FLOAT NULL,
        [Calibrated_Forecast_MWH] FLOAT NULL,
        [Warm_Ramp_Cal_MWH] FLOAT NULL,
        [Warm_Ramp_Adjusted_Forecast_MWH] FLOAT NULL,
        [Warm_Ramp_Correction_Source] NVARCHAR(MAX) NULL,
        [CloudSolarBaseBucket] NVARCHAR(MAX) NULL,
        [CloudSolarEventClass] NVARCHAR(MAX) NULL,
        [CloudSolarEventMultiplier] FLOAT NULL,
        [Cloud_Solar_Shape_Cal_MWH] FLOAT NULL,
        [Cloud_Solar_Shape_Raw_Cal_MWH] FLOAT NULL,
        [Cloud_Solar_Adjusted_Forecast_MWH] FLOAT NULL,
        [Cloud_Solar_Correction_Source] NVARCHAR(MAX) NULL,
        [Peak_Risk_Cal_MWH] FLOAT NULL,
        [Peak_Risk_Source] NVARCHAR(MAX) NULL,
        [Peak_Risk_Adjusted_Forecast_MWH] FLOAT NULL,
        [DailyMaxTempBucket] FLOAT NULL,
        [BTMSolarBucket] NVARCHAR(MAX) NULL,
        [Recent_Level_Correction_MWH] FLOAT NULL,
        [Recent_Correction_Source] NVARCHAR(MAX) NULL,
        [Pre_Recent_Forecast_MWH] FLOAT NULL,
        [Recent_Corrected_Forecast_MWH] FLOAT NULL,
        [Final_Backtest_Forecast_MWH] FLOAT NULL,
        [Final_Forecast_MWH] FLOAT NULL,
        [Recent_Corrected_Residual_MWH] FLOAT NULL,
        [Recent_Corrected_AbsError_MWH] FLOAT NULL,
        [Recent_Corrected_APE] FLOAT NULL,
        [Long_Horizon_Peak_Month_Correction_MWH] FLOAT NULL,
        [Long_Horizon_Hot_Month_Correction_MWH] FLOAT NULL,
        [Stage_Selected_Forecast_MWH] FLOAT NULL,
        [Stage_Selector_Source] NVARCHAR(MAX) NULL,
        [Stage_Selector_Reason] NVARCHAR(MAX) NULL,
        [Focused_Scorecard_Guard_MWH] FLOAT NULL,
        [Focused_Scorecard_Guard_Source] NVARCHAR(MAX) NULL,
        [Final_Residual_MWH] FLOAT NULL,
        [Final_AbsError_MWH] FLOAT NULL,
        [Final_APE] FLOAT NULL,
        CONSTRAINT [FK_LoadForecastBacktest_Run] FOREIGN KEY ([RunID])
            REFERENCES [Forecasting].[LoadForecastRun] ([RunID])
    );
END;
GO

IF OBJECT_ID(N'[Forecasting].[LoadForecastWeather]', N'U') IS NULL
BEGIN
    CREATE TABLE [Forecasting].[LoadForecastWeather] (
        [OutputRowID] BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT [PK_LoadForecastWeather_OutputRowID] PRIMARY KEY,
        [RunID] UNIQUEIDENTIFIER NOT NULL,
        [InsertedAtUTC] DATETIME2(7) NOT NULL CONSTRAINT [DF_LoadForecastWeather_InsertedAtUTC] DEFAULT SYSUTCDATETIME(),
        [DT] DATETIMEOFFSET(7) NULL,
        [TempF] FLOAT NULL,
        [HumidityPct] FLOAT NULL,
        [CloudCoverPct] FLOAT NULL,
        [WindSpeedMph] FLOAT NULL,
        [PrecipIn] FLOAT NULL,
        [GHI_Wm2] FLOAT NULL,
        [IsDay] BIGINT NULL,
        [Dynamic_Weather_Correction_F] FLOAT NULL,
        CONSTRAINT [FK_LoadForecastWeather_Run] FOREIGN KEY ([RunID])
            REFERENCES [Forecasting].[LoadForecastRun] ([RunID])
    );
END;
GO

DECLARE @ReplayTables TABLE (
    [TableName] SYSNAME NOT NULL PRIMARY KEY
);

INSERT INTO @ReplayTables ([TableName])
VALUES
    (N'LoadForecastReplaySummary'),
    (N'LoadForecastReplayResult'),
    (N'LoadForecastReplayOriginCoverage'),
    (N'LoadForecastReplayScorecard'),
    (N'LoadForecastReplayWeatherRealismScorecard'),
    (N'LoadForecastReplayWeatherInputErrorByLead'),
    (N'LoadForecastReplayWeatherInputSensitivityScorecard'),
    (N'LoadForecastReplayWeatherInputSensitivityDetail'),
    (N'LoadForecastReplayStageMetric'),
    (N'LoadForecastReplayOriginMetricByStage'),
    (N'LoadForecastReplayScoredSeasonMetricByStage'),
    (N'LoadForecastReplayOriginSeasonMetricByStage'),
    (N'LoadForecastReplayHorizonMetricByStage'),
    (N'LoadForecastReplayPeakWindowMetricByStage'),
    (N'LoadForecastReplayHotPeakMetricByStage'),
    (N'LoadForecastReplayShoulderHeatMetricByStage'),
    (N'LoadForecastReplayCloudSolarMiddayMetricByStage'),
    (N'LoadForecastReplayWeekendMetricByStage'),
    (N'LoadForecastReplayHolidayMetricByStage'),
    (N'LoadForecastReplayLongHorizonMetricByStage'),
    (N'LoadForecastReplayDailyPeakMissByStage'),
    (N'LoadForecastReplayTiming'),
    (N'LoadForecastProductionReadinessScorecard');

DECLARE @ReplayTable SYSNAME;
DECLARE @ReplayFullName NVARCHAR(300);
DECLARE @ReplayObjectName NVARCHAR(300);
DECLARE @ReplaySql NVARCHAR(MAX);
DECLARE @ReplayIndexName SYSNAME;

DECLARE ReplayTableCursor CURSOR LOCAL FAST_FORWARD FOR
    SELECT [TableName]
    FROM @ReplayTables
    ORDER BY [TableName];

OPEN ReplayTableCursor;
FETCH NEXT FROM ReplayTableCursor INTO @ReplayTable;

WHILE @@FETCH_STATUS = 0
BEGIN
    SET @ReplayFullName = QUOTENAME(N'Forecasting') + N'.' + QUOTENAME(@ReplayTable);
    SET @ReplayObjectName = N'Forecasting.' + @ReplayTable;

    IF OBJECT_ID(@ReplayObjectName, N'U') IS NULL
    BEGIN
        SET @ReplaySql = N'
            CREATE TABLE ' + @ReplayFullName + N' (
                [OutputRowID] BIGINT IDENTITY(1,1) NOT NULL CONSTRAINT ' + QUOTENAME(N'PK_' + @ReplayTable + N'_OutputRowID') + N' PRIMARY KEY,
                [RunID] UNIQUEIDENTIFIER NOT NULL,
                [InsertedAtUTC] DATETIME2(7) NOT NULL CONSTRAINT ' + QUOTENAME(N'DF_' + @ReplayTable + N'_InsertedAtUTC') + N' DEFAULT SYSUTCDATETIME(),
                CONSTRAINT ' + QUOTENAME(N'FK_' + @ReplayTable + N'_Run') + N' FOREIGN KEY ([RunID])
                    REFERENCES [Forecasting].[LoadForecastRun] ([RunID])
            );';
        EXEC sp_executesql @ReplaySql;
    END;

    SET @ReplayIndexName = LEFT(N'IX_' + @ReplayTable + N'_RunID', 128);
    IF NOT EXISTS (
        SELECT 1 FROM sys.indexes
        WHERE name = @ReplayIndexName
          AND object_id = OBJECT_ID(@ReplayObjectName)
    )
    BEGIN
        SET @ReplaySql = N'CREATE INDEX ' + QUOTENAME(@ReplayIndexName) + N' ON ' + @ReplayFullName + N' ([RunID]);';
        EXEC sp_executesql @ReplaySql;
    END;

    FETCH NEXT FROM ReplayTableCursor INTO @ReplayTable;
END;

CLOSE ReplayTableCursor;
DEALLOCATE ReplayTableCursor;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_LoadForecastRun_RunStartedAtUTC'
      AND object_id = OBJECT_ID(N'[Forecasting].[LoadForecastRun]')
)
    CREATE INDEX [IX_LoadForecastRun_RunStartedAtUTC]
        ON [Forecasting].[LoadForecastRun] ([RunStartedAtUTC] DESC, [InsertedAtUTC] DESC);
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_LoadForecastOutput_RunID'
      AND object_id = OBJECT_ID(N'[Forecasting].[LoadForecastOutput]')
)
    CREATE INDEX [IX_LoadForecastOutput_RunID]
        ON [Forecasting].[LoadForecastOutput] ([RunID]);
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_LoadForecastOutput_RunID_DT'
      AND object_id = OBJECT_ID(N'[Forecasting].[LoadForecastOutput]')
)
    CREATE INDEX [IX_LoadForecastOutput_RunID_DT]
        ON [Forecasting].[LoadForecastOutput] ([RunID], [DT]);
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_LoadForecastBacktest_RunID'
      AND object_id = OBJECT_ID(N'[Forecasting].[LoadForecastBacktest]')
)
    CREATE INDEX [IX_LoadForecastBacktest_RunID]
        ON [Forecasting].[LoadForecastBacktest] ([RunID]);
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_LoadForecastBacktest_RunID_DT'
      AND object_id = OBJECT_ID(N'[Forecasting].[LoadForecastBacktest]')
)
    CREATE INDEX [IX_LoadForecastBacktest_RunID_DT]
        ON [Forecasting].[LoadForecastBacktest] ([RunID], [DT]);
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_LoadForecastWeather_RunID'
      AND object_id = OBJECT_ID(N'[Forecasting].[LoadForecastWeather]')
)
    CREATE INDEX [IX_LoadForecastWeather_RunID]
        ON [Forecasting].[LoadForecastWeather] ([RunID]);
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = N'IX_LoadForecastWeather_RunID_DT'
      AND object_id = OBJECT_ID(N'[Forecasting].[LoadForecastWeather]')
)
    CREATE INDEX [IX_LoadForecastWeather_RunID_DT]
        ON [Forecasting].[LoadForecastWeather] ([RunID], [DT]);
GO
