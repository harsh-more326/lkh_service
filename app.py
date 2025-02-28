from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import subprocess
from supabase import create_client
from dotenv import load_dotenv
import json
import math
import logging
from collections import defaultdict

# Load environment variables
load_dotenv()

# Initialize Supabase
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY) if SUPABASE_URL and SUPABASE_ANON_KEY else None

# Setup logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Constants
MAX_STOPS_PER_ROUTE = 60  # Adjust this value as needed

# Haversine formula to calculate distance
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Data Models
class BusStop(BaseModel):
    id: str
    latitude: float
    longitude: float
    priority: int

class Depot(BaseModel):
    id: str
    latitude: float
    longitude: float
    
def fetch_all_rows(table_name: str):
    all_rows = []
    chunk_size = 500  # Supabase default limit
    start = 0

    while True:
        response = supabase.table(table_name).select("*").range(start, start + chunk_size - 1).execute()
        if response.data:
            all_rows.extend(response.data)
            start += chunk_size
        else:
            break  # Stop when no more data

    return all_rows

def assign_stops_to_depots(depots: List[Depot], stops: List[BusStop]) -> Dict[str, Dict]:
    clusters = {}
    for depot in depots:
        clusters[depot.id] = {"depot": depot, "stops": []}
        logging.info(f"Depot {depot.id} assigned {len(clusters[depot.id]['stops'])} stops")


    for stop in stops:
        min_dist = float('inf')
        closest_depot_id = None
        for depot in depots:
            dist = haversine(depot.latitude, depot.longitude, stop.latitude, stop.longitude)
            if dist < min_dist:
                min_dist = dist
                closest_depot_id = depot.id
        if closest_depot_id:
            clusters[closest_depot_id]["stops"].append(stop)

    return clusters

def generate_tsp_file(depots: List[Depot], stops: List[BusStop]) -> str:
    all_points = depots + sorted(stops, key=lambda stop: stop.priority, reverse=True)
    tsp_content = f"NAME : BusRoute\nTYPE : TSP\nDIMENSION : {len(all_points)}\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n"
    for i, point in enumerate(all_points, start=1):
        tsp_content += f"{i} {point.latitude} {point.longitude}\n"
    tsp_content += "EOF\n"
    return tsp_content

def generate_par_file(tsp_filename: str, num_vehicles: int, route_filename: str) -> str:
    return f"""PROBLEM_FILE = {tsp_filename}
SALESMEN = {num_vehicles}
DEPOT = 1
EXCESS = 0.15  # Increased flexibility
GAIN23 = YES
INITIAL_TOUR_ALGORITHM = NEAREST-NEIGHBOR
CANDIDATE_SET_TYPE = NEAREST-NEIGHBOR
KICKS = 3
MAX_TRIALS = 500
MOVE_TYPE = 5
MTSP_MIN_SIZE = 15  
MTSP_MAX_SIZE = 30
POPMUSIC_INITIAL_TOUR = YES
TIME_LIMIT = 300.0
MTSP_OBJECTIVE = MINMAX_SIZE  
MTSP_SOLUTION_FILE = {route_filename}
"""

def parse_lkh_output(route_file_path: str, all_points: List) -> List[List[Dict]]:
    try:
        with open(route_file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading {route_file_path}: {str(e)}")

    routes = []
    current_route = []

    for line in lines:
        line = line.strip()
        if line.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
            route_ids = [item for item in line.split() if item.isdigit()]
            if route_ids:
                current_route = [{"lat": all_points[int(i)-1].latitude, "lng": all_points[int(i)-1].longitude} for i in route_ids]
                routes.append(current_route)
        elif line.startswith("Cost"):
            continue

    if not routes:
        raise HTTPException(status_code=500, detail="LKH-3 returned empty routes.")
    return routes

@app.get("/optimize")
async def optimize_route():
    try:
        depots_data = fetch_all_rows("depots")
        stops_data = fetch_all_rows("bus_stops")
       
        if not depots_data or not stops_data:
            raise HTTPException(status_code=404, detail="No depots or bus stops found.")

        depots = [Depot(**depot) for depot in depots_data]
        stops = [BusStop(**stop) for stop in stops_data]

        clusters = assign_stops_to_depots(depots, stops)

        lkh_folder = "/LKH-3/LKH-3.0.13"
        lkh_path = os.path.join(lkh_folder, "LKH")
        routes_info = []

        for depot_id, cluster in clusters.items():
            depot = cluster["depot"]
            cluster_stops = cluster["stops"]
            if not cluster_stops:
                logging.info(f"Skipping depot {depot.id} with no stops.")
                continue
            
            # Calculate number of vehicles based on MAX_STOPS_PER_ROUTE
            cluster_vehicles = max(1, math.ceil(len(cluster_stops) / MAX_STOPS_PER_ROUTE))


            cluster_id = depot.id
            tsp_filename = f"BusRoute_{cluster_id}.tsp"
            tsp_path = os.path.join(lkh_folder, tsp_filename)
            par_filename = f"BusRoute_{cluster_id}.par"
            par_path = os.path.join(lkh_folder, par_filename)
            route_filename = f"route_{cluster_id}.txt"
            route_file_path = os.path.join(lkh_folder, route_filename)

            # Generate TSP and PAR files
            tsp_content = generate_tsp_file([depot], cluster_stops)
            with open(tsp_path, 'w') as f:
                f.write(tsp_content)

            par_content = generate_par_file(tsp_filename, cluster_vehicles, route_filename)
            with open(par_path, 'w') as f:
                f.write(par_content)

            # Run LKH-3
            result = subprocess.run([lkh_path, par_path], capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"LKH-3 error for {cluster_id}: {result.stderr}")
                continue

            # Parse routes
            all_points_cluster = [depot] + sorted(cluster_stops, key=lambda s: s.priority, reverse=True)
            try:
                cluster_routes = parse_lkh_output(route_file_path, all_points_cluster)
            except HTTPException as e:
                logging.error(f"Parse error for {cluster_id}: {str(e)}")
                continue

            # Calculate route info
            for route in cluster_routes:
                total_distance = sum(haversine(route[i]["lat"], route[i]["lng"], route[i+1]["lat"], route[i+1]["lng"]) for i in range(len(route)-1))
                stops_count = len(route) - 2  # Correctly exclude depot start and end
                routes_info.append({
                    "waypoints": route,
                    "distance": round(total_distance, 2),
                    "duration": int(total_distance * 2),
                    "depot_id": depot_id,
                    "stops_number": stops_count,
                    "num_vehicles": cluster_vehicles
                })

        # Update database
        existing_routes = supabase.table("optimized_routes").select("id").execute()
        for record in existing_routes.data:
            supabase.table("optimized_routes").delete().eq("id", record["id"]).execute()

        for route in routes_info:
            supabase.table("optimized_routes").insert({
                "name": f"Optimized Route (Depot {route['depot_id']})",
                "stops": json.dumps({"waypoints": route['waypoints']}),
                "distance": route["distance"],
                "duration": route["duration"],
                "buses": route["num_vehicles"],
                "frequency": 1,
                "stops_number": route["stops_number"]
            }).execute()

        return {"routes": routes_info}

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Bus Route Optimization API is Running"}