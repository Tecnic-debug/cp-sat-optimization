import os
import math
import json
import pandas as pd
from ortools.sat.python import cp_model

# Define where the source data is and where results should be saved
UPLOADED_CSV_PATH = r"C:\Users\Gaumtes\Downloads\Assignments\EDA\New_optimization\new_or_tools_program_1_uploaded_file.csv"
OUT_CSV_PATH = r"C:\Users\Gaumtes\Downloads\Assignments\EDA\New_optimization\new_or_tools_program_1_output.csv"
OUT_SUMMARY_PATH = r"C:\Users\Gaumtes\Downloads\Assignments\EDA\New_optimization\new_or_tools_program_1_summary.json"

TIME_LIMIT_SEC = 900 # Allow solver 15 minutes to find the best route
NUM_WORKERS = 8 # Utilize multi-core processing for faster solving
MAX_NEAR_DISTANCE_KM = 25  

MIN_LOAD_KG = 120   # Minimum gas weight required to justify a trip
BIG_SLACK = 2_000_000  # Penalty weight for failing to meet demand
DELIVERY_WEIGHT = 2000    
EXTRA_ROTATION_LCVS = {965, 967, 5833, 3682, 3727}  # Specific trucks allowed extra trips

compressor_config = {
    "Hafeezpet": [
        {"id": "Hafeezpet_Comp1", "capacity": 9840,  "running_hours": 24},
        {"id": "Hafeezpet_Comp2", "capacity": 8440,  "running_hours": 24},
        {"id": "Hafeezpet_Comp3", "capacity": 18480, "running_hours": 24},
        {"id": "Hafeezpet_Comp4", "capacity": 18480, "running_hours": 24},
        {"id": "Hafeezpet_Comp5", "capacity": 18480, "running_hours": 24}
    ],
    "Shamirpet": [
        {"id": "Shamirpet_Comp1", "capacity": 25760, "running_hours": 24},
        {"id": "Shamirpet_Comp2", "capacity": 18480, "running_hours": 24},
        {"id": "Shamirpet_Comp3", "capacity": 18480, "running_hours": 24},
        {"id": "Shamirpet_Comp4", "capacity": 6400,  "running_hours": 24}
    ],
    "Torrent": [
        {"id": "Torrent_Comp1", "capacity": 20000, "running_hours": 24}
    ]
}

ORIGINS = ["Hafeezpet", "Shamirpet", "Torrent"]
MAX_COST_PAISE = 81000000  
TARGET_DISPATCH_KG = 160000 


def effective_capacity_of_compressor(comp, downtime_minutes=0.0): 
    #Calculates actual output based on machine downtime.
    rated_capacity = float(comp.get("capacity", 0.0))
    running_hours = float(comp.get("running_hours", 24.0))
    downtime_hours = max(0.0, downtime_minutes / 60.0)
    effective_hours = max(0.0, running_hours - downtime_hours)
    return rated_capacity * (effective_hours / 24.0)

def compute_origin_daily_capacity(cfg, broken_map=None):
    #Aggregates total available gas at each loading station.
    broken_map = broken_map or {}
    origin_daily_capacity = {}
    for origin, comps in cfg.items():
        total = 0.0
        for idx, comp in enumerate(comps, start=1):
            downtime = 0.0
            if (origin, idx) in broken_map:
                downtime = broken_map[(origin, idx)]
            if (origin, comp.get("id")) in broken_map:
                downtime = broken_map[(origin, comp.get("id"))]
            total += effective_capacity_of_compressor(comp, downtime_minutes=downtime)
        origin_daily_capacity[origin] = int(round(total))
    return origin_daily_capacity

def get_rate(contract, transporter, km, assigned_qty_kg=None, total_km_per_trip=None):
    # Logic for transportation billing based on contract types and distance slabs
    if contract == 'D':
        if total_km_per_trip is None:
            return 5.5
        return 5.5 if total_km_per_trip <= 4000 else 6.0
    
    if contract == 'A':
        if transporter == 'CTC':
            if assigned_qty_kg and assigned_qty_kg > 360000: return 4.00
            if 0 <= km <= 30: return 4.37
            if 31 <= km <= 60: return 5.30
            if 61 <= km <= 90: return 5.40
            return 5.42
        if transporter in ('SV Infra', 'Chakra'):
            if assigned_qty_kg and assigned_qty_kg > 240000: return 4.00
            if 0 <= km <= 30: return 3.94
            return 5.42
        if transporter == 'Frozen':
            if assigned_qty_kg and assigned_qty_kg > 100000: return 3.00
            if 0 <= km <= 25: return 6.90
            if 26 <= km <= 50: return 7.65
            if 51 <= km <= 75: return 8.25
            if 76 <= km <= 100: return 8.75
            return 9.50
    if contract == 'B':
        threshold = 600000 if transporter == 'Frozen' else 400000
        if assigned_qty_kg and assigned_qty_kg > threshold: return 3.25
        if 0 <= km <= 30: return 4.42
        if 31 <= km <= 60: return 4.92
        if 61 <= km <= 90: return 5.37
        return 5.99
    if contract == 'C':
        threshold = 600000 if transporter == 'Frozen' else 900000
        if assigned_qty_kg and assigned_qty_kg > threshold: return 3.25
        if 0 <= km <= 30: return 4.49
        if 31 <= km <= 60: return 5.77
        if 61 <= km <= 90: return 5.88
        return 6.40
    return float('inf')


def load_data(uploaded_csv_path=UPLOADED_CSV_PATH):
    try:
        import station_file_copy as sf_mod
        import lcv_file_Copy as lcv_mod
        stations = pd.DataFrame(sf_mod.station_file)
        lcv_list = lcv_mod.lcv_list
        print("Loaded station_file & lcv_file modules.")
    except Exception:
        stations = None
        lcv_list = None

    if stations is None and os.path.exists(uploaded_csv_path):
        try:
            df = pd.read_csv(uploaded_csv_path)
            if 'Station (Unloading)' in df.columns:
                stations = []
                for name, group in df.groupby('Station (Unloading)'):
                    demand = int(group['Station Capacity'].iloc[0]) if 'Station Capacity' in group.columns else int(group['Quantity (kg)'].sum())
                    dist = {}
                    for o in ORIGINS:
                        col = f"Distance from {o}"
                        if col in group.columns:
                            try:
                                dist[o] = float(group[col].iloc[0])
                            except Exception:
                                dist[o] = float('inf')
                        else:
                            if '(Loading→Unloading)' in group.columns:
                                try:
                                    dist[o] = float(group['(Loading→Unloading)'].mean())
                                except Exception:
                                    dist[o] = float('inf')
                            else:
                                dist[o] = float('inf')
                    stations.append({
                        'Station Name': name,
                        'Capacity': demand,
                        'Distance from Hafeezpet': dist['Hafeezpet'],
                        'Distance from Shamirpet': dist['Shamirpet'],
                        'Distance from Torrent': dist['Torrent']
                    })
                stations = pd.DataFrame(stations)
                lcv_list = []
                if 'LCV No' in df.columns:
                    for lno, g in df.groupby('LCV No'):
                        sample = g.iloc[0].to_dict()
                        lcv_list.append({
                            'lcv_no': int(lno),
                            'capacity': int(sample.get('LCV Capacity', 400)),
                            'contract': sample.get('Contract', 'D'),
                            'transporter': sample.get('Transporter', 'Frozen'),
                            'status': sample.get('status', 'active')    
                        })
                if len(lcv_list) == 0:
                    raise ValueError("Uploaded CSV parsed but no LCV entries found.")
            else:
                if 'Type' in df.columns and set(df['Type'].unique()).issuperset({'Station', 'LCV'}):
                    stations = df[df['Type'] == 'Station'].copy()
                    lcv_list = df[df['Type'] == 'LCV'].to_dict('records')
                else:
                    raise ValueError("Uploaded CSV format not recognized.")
            print(f"Parsed {len(stations)} stations and {len(lcv_list)} LCVs from uploaded CSV.")
        except Exception as e:
            print("Error parsing uploaded CSV:", e)
            stations = None
            lcv_list = None

    stations = stations.reset_index(drop=True)
    return stations, lcv_list


def build_stage1_model(stations_df, lcv_list):
    model = cp_model.CpModel()
    capacities = {}
    contract_map = {}
    transporter_map = {}
    trip_limit_map = {}
    for lcv in lcv_list:
        l = int(lcv['lcv_no'])
        cap = int(lcv.get('capacity', 400))
        capacities[l] = cap
        contract_map[l] = (lcv.get('contract') or 'D')
        transporter_map[l] = lcv.get('transporter')
        if contract_map[l] == 'D' or l in EXTRA_ROTATION_LCVS:
            trip_limit_map[l] = 3
        else:
            trip_limit_map[l] = 2

    origin_capacity = compute_origin_daily_capacity(compressor_config)
    x = {}
    y = {}
    for lcv in lcv_list:
        l = int(lcv['lcv_no'])
        cap = capacities[l]
        for sid, srow in stations_df.iterrows():
            for o in ORIGINS:
                vx = model.NewIntVar(0, cap, f"x_l{l}_s{sid}_o{o}")
                vy = model.NewBoolVar(f"y_l{l}_s{sid}_o{o}")
                x[(l, sid, o)] = vx
                y[(l, sid, o)] = vy
                model.Add(x[(l, sid, o)] <= cap * y[(l, sid, o)])
                if MIN_LOAD_KG > 0:
                    model.Add(x[(l, sid, o)] >= MIN_LOAD_KG * y[(l, sid, o)])


    for l in EXTRA_ROTATION_LCVS:
        for sid, srow in stations_df.iterrows():
            for o in ORIGINS:
                dist = float(srow.get(f"Distance from {o}", 999999))
                if dist > MAX_NEAR_DISTANCE_KM:
                    model.Add(y[(l, sid, o)] == 0)
    slack = {}
    for sid, srow in stations_df.iterrows():
        max_d = int(srow.get('Capacity', 0))
        slack[(sid,)] = model.NewIntVar(0, max_d, f"slack_s{sid}")
    for sid, srow in stations_df.iterrows():
        demand = int(srow.get('Capacity', 0))
        if demand <= 0:
            continue
        model.Add(
            sum(x[(int(lcv['lcv_no']), sid, o)] for lcv in lcv_list for o in ORIGINS) + slack[(sid,)] >= demand
        )

    for o in ORIGINS:
        cap_o = int(origin_capacity.get(o, 0))
        model.Add(
            sum(x[(int(lcv['lcv_no']), sid, o)] for lcv in lcv_list for sid, srow in stations_df.iterrows()) <= cap_o
        )

    for lcv in lcv_list:
        l = int(lcv['lcv_no'])
        limit = int(trip_limit_map[l])
        model.Add(
            sum(y[(l, sid, o)] for sid, srow in stations_df.iterrows() for o in ORIGINS) <= limit
        )

    total_cost = sum(
        x[(l, sid, o)] * int(
            get_rate(
                contract_map[l],
                transporter_map[l],
                int(round((stations_df.loc[sid, f"Distance from {o}"] if pd.notna(stations_df.loc[sid, f"Distance from {o}"]) else 1e6) * 2))
            ) * 100
        )
        for (l, sid, o) in x
    )
    
    model.Add(total_cost <= MAX_COST_PAISE)

    
    total_slack = sum(slack[(sid,)] for sid, srow in stations_df.iterrows())
    total_trips = sum(y[(int(lcv['lcv_no']), sid, o)] for lcv in lcv_list for sid, srow in stations_df.iterrows() for o in ORIGINS)
    total_delivered = sum(x.values())
    
    model.Minimize(
        BIG_SLACK * total_slack
        - DELIVERY_WEIGHT * total_delivered
        + total_trips
    )
    model.Add(total_delivered >= 160000)

    return model, x, y, slack, capacities

def run_stage1_only(stations_df, lcv_list):
    #Executes the solver and formats the output into a readable report.
    origin_capacity = compute_origin_daily_capacity(compressor_config)
    print("Origin daily capacity:", origin_capacity)
    print("Stations loaded:", len(stations_df), "LCVs loaded:", len(lcv_list))
    print("MIN_LOAD_KG:", MIN_LOAD_KG, "BIG_SLACK:", BIG_SLACK, "DELIVERY_WEIGHT:", DELIVERY_WEIGHT)
    print(f"Target dispatch: {TARGET_DISPATCH_KG} kg, Max cost: ₹{MAX_COST_PAISE/100:.2f}")

    model, x_vars, y_vars, slack_vars, capacities = build_stage1_model(stations_df, lcv_list)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIME_LIMIT_SEC
    solver.parameters.num_search_workers = NUM_WORKERS
    print("Solving Stage-1 model (minimize slack, maximize delivery, minimize trips)...")
    status = solver.Solve(model)
    
    status_map = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.UNKNOWN: "UNKNOWN"
    }
    
    print("Solver status:", status_map.get(status, status))
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("Stage-1 infeasible — review capacities, trip limits or increase TIME_LIMIT_SEC.")
        return None

    
    rows = []
    total_cost_paise = 0
    trips_per_lcv = {int(lcv['lcv_no']): 0 for lcv in lcv_list}
    total_dispatched = 0
    total_slack = 0
    rate_paise = {}
    for lcv in lcv_list:
        l = int(lcv['lcv_no'])
        for sid, srow in stations_df.iterrows():
            for o in ORIGINS:
                dist_col = f"Distance from {o}"
                km_one_way = float(srow.get(dist_col, float('inf')))
                round_km = int(round(km_one_way * 2.0)) if math.isfinite(km_one_way) else 99999
                r = get_rate(lcv.get('contract'), lcv.get('transporter'), round_km)
                rate_paise[(l, sid, o)] = int(round(r * 100))

    for lcv in lcv_list:
        l = int(lcv['lcv_no'])
        cap = int(lcv.get('capacity', 400))
        for sid, srow in stations_df.iterrows():
            sname = srow['Station Name']
            for o in ORIGINS:
                xval = solver.Value(x_vars[(l, sid, o)])
                yval = solver.Value(y_vars[(l, sid, o)])
                if yval == 1 and xval > 0:
                    trips_per_lcv[l] += 1
                    total_dispatched += xval
                    km_one_way = float(srow.get(f"Distance from {o}", float('inf')))
                    round_km = km_one_way * 2.0 if math.isfinite(km_one_way) else 99999.0
                    rp = rate_paise[(l, sid, o)]
                    cost_paise = rp * xval
                    total_cost_paise += cost_paise

                    distances = {
                        origin: float(srow.get(f"Distance from {origin}", float('inf')))
                        for origin in ORIGINS
                    }
                    next_loading = min(distances, key=lambda k: (distances[k], ORIGINS.index(k)))
                    helper = distances[next_loading]
                    rows.append({
                        "LCV No": l,
                        "Contract": lcv.get("contract"),
                        "Transporter": lcv.get("transporter"),
                        "From": o,
                        "To": sname,
                        "Next Loading Station": next_loading,
                        "Quantity (kg)": int(xval),
                        "Trips (Round Trips)": int(yval),
                        "One-way km": km_one_way,
                        "Round Trip km": round_km,
                        "Rate (₹/kg)": round(rp / 100.0, 2),
                        "Cost (₹)": round(cost_paise / 100.0, 2)
                    })

    slack_rows = []
    for sid, srow in stations_df.iterrows():
        sname = srow['Station Name']
        slack_kg = solver.Value(slack_vars[(sid,)])
        total_slack += slack_kg
        if slack_kg > 0:
            slack_rows.append({"Station (Unloading)": sname, "Unmet Demand (kg)": int(slack_kg)})

    out_df = pd.DataFrame(rows)
    out_dir = os.path.dirname(OUT_CSV_PATH)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(OUT_CSV_PATH, index=False)
    if slack_rows:
        slack_df = pd.DataFrame(slack_rows)
        slack_path = os.path.splitext(OUT_CSV_PATH)[0] + "_slack.csv"
        slack_df.to_csv(slack_path, index=False)
        print("Slack (unmet demand) written to:", slack_path)

    trips_by_contract = {}
    for lcv in lcv_list:
        key = f"{lcv.get('contract')}__{lcv.get('transporter')}"
        trips_by_contract.setdefault(key, 0)
        trips_by_contract[key] += trips_per_lcv.get(int(lcv['lcv_no']), 0)

    summary = {
        "total_rows": len(out_df),
        "total_slack_kg": int(total_slack),
        "total_trips": sum(trips_per_lcv.values()),
        "total_cost_₹": round(total_cost_paise / 100.0, 2),
        "total_dispatched_kg": float(total_dispatched),
        "target_dispatch_kg": TARGET_DISPATCH_KG,
        "dispatch_achievement_%": round(100 * total_dispatched / TARGET_DISPATCH_KG, 2),
        "origin_capacity": origin_capacity,
        "trips_per_lcv": trips_per_lcv,
        "trips_by_contract": trips_by_contract
    }

    with open(OUT_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("Stage-1 CP-SAT optimization completed. Output saved to:", OUT_CSV_PATH)
    print(f"Total slack (kg): {total_slack}, "
          f"Total dispatched (kg): {summary['total_dispatched_kg']:.0f} ({summary['dispatch_achievement_%']}% of target), "
          f"Total trips: {summary['total_trips']}, "
          f"Total cost (₹): {summary['total_cost_₹']:.2f}")
    return {"out_df": out_df, "summary": summary, "slack_rows": slack_rows}


def main():
    stations_df, lcv_list = load_data()
    print("Loaded:", len(stations_df), "stations;", len(lcv_list), "LCVs")
    res = run_stage1_only(stations_df, lcv_list)
    if res is None:
        print("No feasible solution. Consider relaxing trip limits or increasing TIME_LIMIT_SEC.")
    else:
        print("Summary written to", OUT_SUMMARY_PATH)

if __name__ == "__main__":
    main()
