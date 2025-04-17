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
import time
import requests
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
MAX_STOPS_PER_ROUTE = 35
OSRM_ROUTING_URL = "http://router.project-osrm.org/route/v1/driving/"

# Haversine formula to calculate distance
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def generate_alphabetical_identifier(index):
    """Generate alphabetical identifier: A, B, ..., Z, AA, AB, etc."""
    # Calculate how many letters we need
    if index <= 26:
        # Single letter (A through Z)
        return chr(64 + index)  # ASCII: A=65, so we add to 64
    else:
        # Multiple letters (AA, AB, etc.)
        q, r = divmod(index - 1, 26)
        return chr(65 + q - 1) + chr(65 + r)

# Calculate route distance and duration using OSRM (Leaflet Routing Machine backend)
def get_osrm_route_data(waypoints):
    try:
        # Format waypoints for OSRM API
        coordinates = ";".join([f"{point['lng']},{point['lat']}" for point in waypoints])
        url = f"{OSRM_ROUTING_URL}{coordinates}?overview=false"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data['code'] == 'Ok' and data['routes']:
                # Convert distance from meters to kilometers
                distance = data['routes'][0]['distance'] / 1000
                # Convert duration from seconds to minutes
                duration = data['routes'][0]['duration'] / 60
                return round(distance, 2), round(duration, 2)
        
        # If API call fails, fall back to haversine calculation
        logging.warning("OSRM routing failed, falling back to haversine calculation")
    except Exception as e:
        logging.error(f"Error with OSRM API: {str(e)}")
    
    # Fallback calculation using haversine
    total_distance = sum(haversine(waypoints[i]["lat"], waypoints[i]["lng"], 
                                  waypoints[i+1]["lat"], waypoints[i+1]["lng"]) 
                         for i in range(len(waypoints)-1))
    # Estimate duration based on distance (assuming avg speed of 30 km/h)
    estimated_duration = total_distance * 2
    
    return round(total_distance, 2), round(estimated_duration, 2)

# Data Models
class BusStop(BaseModel):
    id: str
    latitude: float
    longitude: float
    priority: int

class Depot(BaseModel):
    id: str
    name: str
    latitude: float
    longitude: float
    
def fetch_all_rows(table_name: str):
    all_rows = []
    chunk_size = 500
    start = 0

    while True:
        response = supabase.table(table_name).select("*").range(start, start + chunk_size - 1).execute()
        if response.data:
            all_rows.extend(response.data)
            start += chunk_size
        else:
            break

    return all_rows

def fetch_all_worker_ids():
    all_worker_ids = []
    chunk_size = 500
    start = 0

    while True:
        response = supabase.table("workers").select("id").range(start, start + chunk_size - 1).execute()
        if response.data:
            all_worker_ids.extend(record["id"] for record in response.data)  # Extract only "id"
            start += chunk_size
        else:
            break

    return all_worker_ids


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
CANDIDATE_SET_TYPE = NEAREST-NEIGHBOR
PATCHING_A = 2 RESTRICTED
PATCHING_C = 2 RESTRICTED
INITIAL_TOUR_ALGORITHM = NEAREST-NEIGHBOR
KICKS = 3
MAX_TRIALS = 5
MOVE_TYPE = 5
MTSP_MIN_SIZE = 15  
MTSP_MAX_SIZE = {MAX_STOPS_PER_ROUTE}
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
                # Include point_id (bus stop id) along with coordinates
                current_route = [
                    {
                        "lat": all_points[int(i)-1].latitude, 
                        "lng": all_points[int(i)-1].longitude,
                        "id": all_points[int(i)-1].id if hasattr(all_points[int(i)-1], 'id') else None
                    } 
                    for i in route_ids
                ]
                routes.append(current_route)
        elif line.startswith("Cost"):
            continue

    if not routes:
        raise HTTPException(status_code=500, detail="LKH-3 returned empty routes.")
    return routes

@app.get("/optimize")
async def optimize_route():
    try:
        start_time = time.time()

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

            result = subprocess.run([lkh_path, par_path], capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"LKH-3 error for {cluster_id}: {result.stderr}")
                continue

            all_points_cluster = [depot] + sorted(cluster_stops, key=lambda s: s.priority, reverse=True)
            try:
                cluster_routes = parse_lkh_output(route_file_path, all_points_cluster)
            except HTTPException as e:
                logging.error(f"Parse error for {cluster_id}: {str(e)}")
                continue

            for route in cluster_routes:
                # Calculate priority only for actual stops (not depots)
                stops_in_route = [point for point in route if point["id"] is not None and point["id"] != depot.id]
                stops_count = len(stops_in_route)
                
                # Find the corresponding BusStop objects to get priorities
                stop_priorities = []
                for stop_point in stops_in_route:
                    for original_stop in cluster_stops:
                        if original_stop.id == stop_point["id"]:
                            stop_priorities.append(original_stop.priority)
                            break
                
                # Calculate average priority, round up (ceiling), and cap at 10
                avg_priority = sum(stop_priorities) / len(stop_priorities) if stop_priorities else 0
                avg_priority = min(math.ceil(avg_priority), 10)
                
                # Get accurate distance and duration using OSRM (Leaflet Routing Machine)
                distance, duration = get_osrm_route_data(route)
                
                routes_info.append({
                    "waypoints": route,
                    "distance": distance,
                    "duration": duration,
                    "depot_id": depot_id,
                    "depot_name": depot.name,
                    "stops_number": stops_count,
                    "num_vehicles": cluster_vehicles,
                    "avg_priority": avg_priority
                })

        # Delete existing routes
        existing_schedule = supabase.table("schedule").select("id").execute()
        for record in existing_schedule.data:
            supabase.table("schedule").delete().eq("id", record["id"]).execute()

        
        worker_ids = fetch_all_worker_ids()
        for worker_id in worker_ids:
            supabase.table("workers").update({"employee_schedule": None}).eq("id", worker_id).execute()
            print(f"Cleared schedule for worker {worker_id}")


        existing_routes = supabase.table("optimized_routes").select("id").execute()
        for record in existing_routes.data:
            supabase.table("optimized_routes").delete().eq("id", record["id"]).execute()

        # Group routes by depot_id
        depot_routes = defaultdict(list)
        for route in routes_info:
            depot_routes[route["depot_id"]].append(route)

        # Generate alphabetical identifiers for each depot
        depot_ids = list(depot_routes.keys())
        depot_identifiers = {depot_id: generate_alphabetical_identifier(i) 
                            for i, depot_id in enumerate(depot_ids, start=1)}

        # Insert routes with new naming scheme
        for depot_id, routes in depot_routes.items():
            depot_letter = depot_identifiers[depot_id]
            for route_index, route in enumerate(routes, start=1):
                route_name = f"{depot_letter}{route_index}"
                supabase.table("optimized_routes").insert({
                    "name": route_name,
                    "stops": json.dumps({"waypoints": route['waypoints']}),
                    "stops_number": route["stops_number"],
                    "avg_priority": route["avg_priority"],
                    "depot_id": depot_id,
                    "distance": route["distance"],
                    "duration": route["duration"]
                }).execute()
        
        end_time = time.time()
        execution_time = end_time - start_time
        minutes = int(execution_time // 60)
        seconds = execution_time % 60
        return {
            "routes": routes_info,
            "execution_time": f"{minutes} min {seconds:.2f} sec"
        }

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
def read_root():
    return {"message": "Bus Route Optimization API is Running"}