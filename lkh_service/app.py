from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os
import subprocess
from supabase import create_client
from dotenv import load_dotenv
import json
import math
import logging

# Load environment variables
load_dotenv()

# Initialize Supabase
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY) if SUPABASE_URL and SUPABASE_ANON_KEY else None

# Setup logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Haversine formula to calculate distance
def haversine(lat1, lon1, lat2, lon2):
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

# Generate .tsp file for LKH
def generate_tsp_file(depots: List[Depot], stops: List[BusStop]) -> str:
    all_points = depots + sorted(stops, key=lambda stop: stop.priority, reverse=True)
    
    tsp_content = f"NAME : BusRoute\nTYPE : TSP\nDIMENSION : {len(all_points)}\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n"
    
    for i, point in enumerate(all_points, start=1):
        tsp_content += f"{i} {point.latitude} {point.longitude}\n"

    tsp_content += "EOF\n"
    logging.info(f"Generated .tsp file:\n{tsp_content}")
    return tsp_content

def generate_par_file(tsp_path: str, num_vehicles: int) -> str:
    return f"""PROBLEM_FILE = BusRoute.tsp
SALESMEN = 400
DEPOT = 1
EXCESS = 0.10
GAIN23 = YES
INITIAL_TOUR_ALGORITHM = CVRP
KICKS = 3
MAX_TRIALS = 500
MOVE_TYPE = 5
MTSP_MIN_SIZE = 5
MTSP_MAX_SIZE = 20
POPMUSIC_INITIAL_TOUR = YES
TIME_LIMIT = 300.0
MTSP_OBJECTIVE = MINMAX
MTSP_SOLUTION_FILE = route.txt
"""


# Parse LKH output from route.txt
def parse_lkh_output(route_file_path: str, all_points: List[Dict]) -> List[List[Dict]]:
    logging.info(f"Reading LKH output from {route_file_path}")
    
    try:
        with open(route_file_path, 'r') as f:
            # Read all lines from the file
            lines = f.readlines()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading {route_file_path}: {str(e)}")

    routes = []
    current_route = []

    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace

        # Only process lines starting with a number (route identifiers)
        if line.startswith(tuple(str(i) for i in range(1, 10))):  # Check if it starts with a number (1-9)
            # Extract the numbers, ignoring non-numeric characters
            route_ids = [item for item in line.split() if item.isdigit()]
            
            if route_ids:
                # Add the waypoints to the current route (use the correct index in all_points)
                current_route = [{"lat": all_points[int(i) - 1].latitude, "lng": all_points[int(i) - 1].longitude} for i in route_ids]
                
                if current_route:
                    routes.append(current_route)

        if line.startswith("Cost"):
            continue  # Skip cost lines

    if current_route:
        routes.append(current_route)

    if not routes:
        raise HTTPException(status_code=500, detail="LKH-3 returned empty routes.")

    return routes



@app.get("/optimize")
async def optimize_route():
    try:
        # Fetch data from Supabase
        depots_data = supabase.table("depots").select("*").execute()
        stops_data = supabase.table("bus_stops").select("*").execute()

        if not depots_data.data or not stops_data.data:
            raise HTTPException(status_code=404, detail="No depots or bus stops found in the database.")

        depots = [Depot(**depot) for depot in depots_data.data]
        stops = [BusStop(**stop) for stop in stops_data.data]

        # Define the number of vehicles (buses)
        num_vehicles = 3  # Change based on the number of buses available

        # Specify the LKH folder
        lkh_folder = "/LKH-3/LKH-3.0.13"  # Folder where LKH is located
        tsp_path = os.path.join(lkh_folder, "BusRoute.tsp")
        par_path = os.path.join(lkh_folder, "BusRoute.par")
        route_file_path = os.path.join(lkh_folder, "route.txt")

        # Generate and save .tsp and .par files
        tsp_content = generate_tsp_file(depots, stops)
        with open(tsp_path, 'w') as tsp_file:
            tsp_file.write(tsp_content)

        par_content = generate_par_file(tsp_path, num_vehicles)
        with open(par_path, 'w') as par_file:
            par_file.write(par_content)

        # Run LKH-3
        lkh_path = os.path.join(lkh_folder, "LKH")
        result = subprocess.run([lkh_path, par_path], capture_output=True, text=True)

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"LKH-3 Error: {result.stderr}")

        # Check if route.txt exists and is generated
        if not os.path.exists(route_file_path):
            raise HTTPException(status_code=500, detail="LKH-3 did not generate route.txt.")



        # Parse LKH output from route.txt
        all_points = depots + sorted(stops, key=lambda stop: stop.priority, reverse=True)
        optimized_routes = parse_lkh_output(route_file_path, all_points)
        data = supabase.table("optimized_routes").select("id").execute()
        for record in data.data:
            supabase.table("optimized_routes").delete().eq("id", record["id"]).execute()

        # Calculate total distance for each route
        routes_info = []
        for route in optimized_routes:
            total_distance = sum(
                haversine(route[i]["lat"], route[i]["lng"],
                          route[i + 1]["lat"], route[i + 1]["lng"] )
                for i in range(len(route) - 1)
            )
            routes_info.append({
                "waypoints": route,
                "distance": round(total_distance, 2),
                "duration": int(total_distance * 2)  # Approximate duration
            })

        # Insert optimized routes into the database
        for route in routes_info:
            supabase.table("optimized_routes").insert({
                "name": "Optimized Bus Route",
                "stops": json.dumps({"waypoints": route['waypoints']}),  # Store waypoints in JSON format
                "distance": route["distance"],
                "duration": route["duration"],
                "buses": num_vehicles,
                "frequency": 1
            }).execute()

        # Clean up the generated files
        os.remove(tsp_path)
        os.remove(par_path)

        return {"routes": routes_info}

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Bus Route Optimization API is Running"}
