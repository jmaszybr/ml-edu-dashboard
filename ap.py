import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import pydeck as pdk

st.set_page_config(page_title="Weather & Air Quality Dashboard", layout="wide")

# -----------------------------
# Open-Meteo endpoints (no key)
# -----------------------------
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

DEFAULT_LOCATIONS = {
    "New York City": (40.7128, -74.0060),
    "Warsaw": (52.2297, 21.0122),
    "Berlin": (52.5200, 13.4050),
    "London": (51.5072, -0.1276),
    "Paris": (48.8566, 2.3522),
}

def safe_get(url: str, params: dict, timeout: int = 20) -> dict:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False, ttl=30 * 60)  # 30 min
def fetch_weather(lat: float, lon: float, tz: str, days: int) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": tz,
        "forecast_days": days,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "windspeed_10m",
            "windgusts_10m",
            "cloudcover"
        ]),
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max"
        ])
    }
    return safe_get(WEATHER_URL, params)

@st.cache_data(show_spinner=False, ttl=60 * 60)  # 60 min
def fetch_air(lat: float, lon: float, tz: str) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": tz,
        "hourly": ",".join(["pm10", "pm2_5", "nitrogen_dioxide", "ozone"])
    }
    return safe_get(AIR_URL, params)

def to_hourly_df(payload: dict, prefix: str) -> pd.DataFrame:
    h = payload.get("hourly", {})
    t = pd.to_datetime(h.get("time", []))
    df = pd.DataFrame({"time": t})
    for k, v in h.items():
        if k == "time":
            continue
        df[f"{prefix}{k}"] = v
    return df

def to_daily_df(payload: dict) -> pd.DataFrame:
    d = payload.get("daily", {})
    t = pd.to_datetime(d.get("time", []))
    df = pd.DataFrame({"date": t})
    for k, v in d.items():
        if k == "time":
            continue
        df[k] = v
    return df

def aq_label(pm25: float) -> str:
    # Proste progi poglądowe (nie udajemy oficjalnej klasyfikacji dla każdego kraju)
    if pm25 is None or np.isnan(pm25):
        return "n/a"
    if pm25 <= 12:
        return "good"
    if pm25 <= 35:
        return "moderate"
    return "poor"

# -----------------------------
# UI: Sidebar
# -----------------------------
st.sidebar.title("⚙️ Ustawienia")

tz = st.sidebar.selectbox("Timezone", ["auto", "Europe/Warsaw", "America/New_York", "UTC"], index=1)
if tz == "auto":
    tz = "UTC"

days = st.sidebar.slider("Forecast days", 1, 16, 7)

mode = st.sidebar.radio("Tryb", ["Single location", "Compare locations"], index=0)

if mode == "Single location":
    loc_name = st.sidebar.selectbox("Location", list(DEFAULT_LOCATIONS.keys()), index=0)
    lat, lon = DEFAULT_LOCATIONS[loc_name]
else:
    selected = st.sidebar.multiselect(
        "Locations (2–6)",
        list(DEFAULT_LOCATIONS.keys()),
        default=["New York City", "Warsaw", "Berlin"]
    )
    if len(selected) < 1:
        selected = ["New York City"]

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Force refresh (clear cache)"):
    st.cache_data.clear()
    st.rerun()

# -----------------------------
# Header
# -----------------------------
st.title("🌦️ Weather & Air Quality Dashboard (API-based)")
st.caption("Źródła: Open-Meteo Weather Forecast API i Open-Meteo Air Quality API (bez klucza API).")

tabs = st.tabs(["Overview", "Map", "Air Quality", "Data Quality", "Table & Export"])

# -----------------------------
# Single location view
# -----------------------------
def render_single(loc_name: str, lat: float, lon: float):
    colA, colB = st.columns([1.2, 1])

    with st.spinner("Pobieram dane z API..."):
        w = fetch_weather(lat, lon, tz, days)
        a = fetch_air(lat, lon, tz)

    w_hour = to_hourly_df(w, prefix="")
    w_day = to_daily_df(w)
    a_hour = to_hourly_df(a, prefix="aq_")

    df = pd.merge(w_hour, a_hour, on="time", how="left")
    df = df.sort_values("time")

    # KPI (use latest hour)
    latest = df.dropna(subset=["temperature_2m"]).tail(1)
    if len(latest) == 1:
        t_now = float(latest["temperature_2m"].iloc[0])
        wind_now = float(latest["windspeed_10m"].iloc[0])
        rain_now = float(latest["precipitation"].iloc[0])
        pm25_now = latest["aq_pm2_5"].iloc[0] if "aq_pm2_5" in latest.columns else np.nan
        pm10_now = latest["aq_pm10"].iloc[0] if "aq_pm10" in latest.columns else np.nan
    else:
        t_now = wind_now = rain_now = np.nan
        pm25_now = pm10_now = np.nan

    with colA:
        st.subheader(f"📍 {loc_name}  ({lat:.4f}, {lon:.4f})")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Temp (latest)", f"{t_now:.1f}°C" if np.isfinite(t_now) else "n/a")
        m2.metric("Wind (latest)", f"{wind_now:.1f} km/h" if np.isfinite(wind_now) else "n/a")
        m3.metric("Precip (latest)", f"{rain_now:.1f} mm" if np.isfinite(rain_now) else "n/a")
        m4.metric("PM2.5 (latest)", f"{pm25_now:.1f} µg/m³" if np.isfinite(pm25_now) else "n/a")
        m5.metric("AQ status", aq_label(pm25_now) if np.isfinite(pm25_now) else "n/a")

        # Temperature chart
        fig_t = px.line(df, x="time", y="temperature_2m", title="Temperature (hourly)")
        st.plotly_chart(fig_t, use_container_width=True)

        # Precip chart
        fig_p = px.bar(df, x="time", y="precipitation", title="Precipitation (hourly)")
        st.plotly_chart(fig_p, use_container_width=True)

    with colB:
        # Daily summary
        st.subheader("📅 Daily summary")
        if len(w_day) > 0:
            fig_d = px.line(
                w_day,
                x="date",
                y=["temperature_2m_max", "temperature_2m_min"],
                title="Daily max/min temperature"
            )
            st.plotly_chart(fig_d, use_container_width=True)

            fig_r = px.bar(w_day, x="date", y="precipitation_sum", title="Daily precipitation sum")
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("Brak danych daily w odpowiedzi API.")

    return df, w, a

# -----------------------------
# Compare view
# -----------------------------
def render_compare(locations: list[str]):
    rows = []
    map_rows = []
    meta = []

    with st.spinner("Pobieram dane dla wielu lokalizacji..."):
        for name in locations:
            lat, lon = DEFAULT_LOCATIONS[name]

            w = fetch_weather(lat, lon, tz, days)
            a = fetch_air(lat, lon, tz)

            w_hour = to_hourly_df(w, prefix="")
            a_hour = to_hourly_df(a, prefix="aq_")
            df = pd.merge(w_hour, a_hour, on="time", how="left").sort_values("time")

            # latest snapshot
            latest = df.dropna(subset=["temperature_2m"]).tail(1)
            if len(latest) == 1:
                t_now = float(latest["temperature_2m"].iloc[0])
                pm25_now = latest["aq_pm2_5"].iloc[0] if "aq_pm2_5" in latest.columns else np.nan
            else:
                t_now = np.nan
                pm25_now = np.nan

            rows.append({
                "location": name,
                "temp_latest_c": t_now,
                "pm25_latest": pm25_now,
                "aq_status": aq_label(pm25_now) if np.isfinite(pm25_now) else "n/a",
            })

            map_rows.append({
                "location": name,
                "lat": lat,
                "lon": lon,
                "temp": t_now,
                "pm25": pm25_now
            })

            meta.append({
                "location": name,
                "weather_hours": int(len(w_hour)),
                "aq_hours": int(len(a_hour)),
                "missing_temp": int(w_hour["temperature_2m"].isna().sum()) if "temperature_2m" in w_hour else None,
                "missing_pm25": int(a_hour["aq_pm2_5"].isna().sum()) if "aq_pm2_5" in a_hour else None,
            })

    kpi = pd.DataFrame(rows).sort_values("temp_latest_c", ascending=False)
    map_df = pd.DataFrame(map_rows)
    meta_df = pd.DataFrame(meta)

    return kpi, map_df, meta_df

# -----------------------------
# Tabs content
# -----------------------------
if mode == "Single location":
    df_all, w_payload, a_payload = render_single(loc_name, lat, lon)

    with tabs[0]:
        st.write("Wybierz zakładki **Map / Air Quality / Data Quality / Table & Export** dla dodatkowych widoków.")

    with tabs[1]:
        st.subheader("🗺️ Map (points + hexbin)")
        # Sample for speed
        d = df_all.dropna(subset=["temperature_2m"]).copy()
        if len(d) > 0:
            d = d.tail(1)  # single point (latest) for this location
        map_df = pd.DataFrame([{"lat": lat, "lon": lon, "temp": d["temperature_2m"].iloc[0] if len(d) else np.nan}])

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_radius=5000,
            get_fill_color="[200, 30, 0, 160]",
            pickable=True,
        )
        view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=8)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "Temp: {temp}°C"}))

    with tabs[2]:
        st.subheader("🫁 Air Quality (hourly)")
        cols = [c for c in df_all.columns if c.startswith("aq_")]
        if cols:
            show = ["aq_pm2_5", "aq_pm10", "aq_nitrogen_dioxide", "aq_ozone"]
            show = [c for c in show if c in df_all.columns]
            fig = px.line(df_all, x="time", y=show, title="Air quality variables (hourly)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Brak danych jakości powietrza w odpowiedzi API.")

    with tabs[3]:
        st.subheader("🧪 Data Quality & Engineering")
        st.write("**Caching:** dane pogodowe TTL 30 min, dane AQ TTL 60 min.")
        st.write("**Walidacja:** odfiltrowane wartości puste; zakresy opadów i czasu pozostają zgodne z API.")
        st.json({
            "weather_keys": list(w_payload.keys()),
            "air_quality_keys": list(a_payload.keys())
        })

    with tabs[4]:
        st.subheader("📄 Table & Export")
        st.dataframe(df_all, use_container_width=True)
        csv = df_all.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, file_name=f"weather_aq_{loc_name}.csv", mime="text/csv")

else:
    with tabs[0]:
        map_df = render_compare(selected)

    with tabs[1]:
        st.subheader("🗺️ Map: compare locations (color by temperature, tooltip includes PM2.5)")
        map_df = map_df.dropna(subset=["lat", "lon"]).copy()

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_radius=80000,
            get_fill_color="[200, 30, 0, 160]",
            pickable=True,
        )
        view_state = pdk.ViewState(latitude=float(map_df["lat"].mean()), longitude=float(map_df["lon"].mean()), zoom=2.2)
        st.pydeck_chart(
            pdk.Deck(layers=[layer], initial_view_state=view_state,
                     tooltip={"text": "{location}\nTemp: {temp}°C\nPM2.5: {pm25} µg/m³"})
        )

    with tabs[2]:
        st.info("W trybie porównawczym AQ jest pokazane w tooltipie mapy + w tabeli snapshot. Możesz rozbudować o wykresy AQ per miasto.")

    with tabs[3]:
        st.subheader("🧪 Data Quality & Engineering")
        st.write("W trybie porównawczym pobieramy dane per lokalizacja, cachowane według współrzędnych i parametrów (TTL).")

    with tabs[4]:
        st.info("Export dotyczy widoku single-location. Jeśli chcesz, dopiszę export dla compare-mode (łączenie danych do jednego CSV).")
