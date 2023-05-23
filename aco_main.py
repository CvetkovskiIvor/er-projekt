import pandas as pd
import numpy as np
from aco import ACO, Graph
from plot import plot
import datetime
import pickle
import argparse
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
import xgboost as xgb
import pprint
import matplotlib.pyplot as plt

# Constants
DATE_LIST = [2016, 5, 27]  # May 27, 2016
DEFAULT_LOC_COUNT = 15
DEFAULT_ANT_COUNT = 10
DEFAULT_GENERATIONS = 100
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 10.0
DEFAULT_RHO = 0.5
DEFAULT_Q = 10.0
FILENAME = "../Util-Functions/xgb_model.xgb"


# Load the XGBoost model
def load_model(filename):
    loaded_model = xgb.Booster()
    loaded_model.load_model(filename)
    return loaded_model


# Calculate the time (in minutes) between two points using the trained XGB model
def time_cost_between_points(loc1, loc2, passenger_count, store_and_fwd_flag=0):
    year, month, day = DATE_LIST

    my_date = datetime.date(year, month, day)

    model_data = {
        'passenger_count': passenger_count,
        'pickup_longitude': loc1['x'],
        'pickup_latitude': loc1['y'],
        'dropoff_longitude': loc2['x'],
        'dropoff_latitude': loc2['y'],
        'store_and_fwd_flag': bool(store_and_fwd_flag),
        'pickup_month': my_date.month,
        'pickup_day': my_date.day,
        'pickup_weekday': my_date.weekday(),
        'pickup_hour': 23,
        'pickup_minute': 10,
        'latitude_difference': loc2['y'] - loc1['y'],
        'longitude_difference': loc2['x'] - loc1['x'],
        'trip_distance': trip_distance_cost(loc1, loc2)
    }

    df = pd.DataFrame([model_data], columns=model_data.keys())
    pred = np.exp(loaded_model.predict(xgb.DMatrix(df))) - 1
    return pred[0]


# Calculate the Manhattan distance between two points using polar coordinates
def trip_distance_cost(loc1, loc2):
    lat_diff = np.abs(loc2['y'] - loc1['y']) * np.pi / 180
    lon_diff = np.abs(loc2['x'] - loc1['x']) * np.pi / 180
    lat_term = np.sin(lat_diff / 2) ** 2
    lon_term = np.sin(lon_diff / 2) ** 2
    a = lat_term + np.cos(loc1['y'] * np.pi / 180) * np.cos(loc2['y'] * np.pi / 180) * lon_term
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 0.621371 * 6371 * c


# Handle geocoding timeout by printing an informative error message
def reverse_geocode_with_timeout(geolocator, point):
    try:
        return geolocator.reverse(f"{point[1]}, {point[0]}").address
    except GeocoderTimedOut as e:
        print(f"Error: Geocoding failed with message: {e}")
        return None


# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("loc_count", type=int, default=DEFAULT_LOC_COUNT, nargs='?',
                        help="number of locations (default is 15)")
    parser.add_argument("ant_count", type=int, default=DEFAULT_ANT_COUNT, nargs='?',
                        help="number of ants to use (default is 10)")
    parser.add_argument("g", type=int, default=DEFAULT_GENERATIONS, nargs='?',
                        help="number of generations (default is 100)")
    parser.add_argument("alpha", type=float, default=DEFAULT_ALPHA, nargs='?',
                        help="relative importance of pheromone (default is 1.0)")
    parser.add_argument("beta", type=float, default=DEFAULT_BETA, nargs='?',
                        help="relative importance of heuristic information (default is 10.0)")
    parser.add_argument("rho", type=float, default=DEFAULT_RHO, nargs='?',
                        help="pheromone residual coefficient (default is 0.5)")
    parser.add_argument("q", type=float, default=DEFAULT_Q, nargs='?',
                        help="pheromone intensity (default is 10.0)")
    parser.add_argument("--verbose", action="store_true",
                        help="print out each generation cost and best path")
    return parser.parse_args()


# Load the XGBoost model
loaded_model = load_model(FILENAME)
geolocator = Nominatim(user_agent="aco-application")


# Main function
def main():
    args = parse_arguments()
    locations, points = load_locations_and_points(args.loc_count)

    cost_matrix = build_cost_matrix(locations)

    aco = ACO(ant_count=args.ant_count, generations=args.g, alpha=args.alpha,
              beta=args.beta, rho=args.rho, q=args.q, strategy=2)

    graph = Graph(cost_matrix, len(locations))
    best_path, cost, avg_costs, best_costs = aco.solve(graph, args.verbose)

    print('Final cost: {} minutes, path: {}'.format(cost, best_path))
    print("Min cost mean:", np.mean(best_costs))
    print("Min cost standard deviation:", np.std(best_costs))

    if args.verbose:
        print("Final path addresses:")
        addresses = [reverse_geocode_with_timeout(geolocator, points[p]) for p in best_path]
        pprint.pprint(addresses)

    plot(points, best_path)

    plot_costs(range(args.g), best_costs, "Best Cost vs Generation for " + str(args.ant_count) + " Ants")
    plot_costs(range(args.g), avg_costs, "Avg Cost vs Generation for " + str(args.ant_count) + " Ants")

    plt.show()


# Load locations and points from the test data
def load_locations_and_points(loc_count):
    df = pd.read_csv("../Util-Functions/test.csv")[:loc_count]
    locations = []
    points = []
    for index, row in df.iterrows():
        locations.append({
            'index': index,
            'x': row['pickup_longitude'],
            'y': row['pickup_latitude']
        })
        points.append((row['pickup_longitude'], row['pickup_latitude']))
    return locations, points


# Build the cost matrix based on time between points
def build_cost_matrix(locations):
    rank = len(locations)
    cost_matrix = []
    for i in range(rank):
        row = []
        for j in range(rank):
            row.append(time_cost_between_points(locations[i], locations[j], 1, 0))
        cost_matrix.append(row)
    return cost_matrix


# Plot costs
def plot_costs(x_values, costs, title):
    plt.title(title)
    plt.ylabel("Cost")
    plt.xlabel("Generation")
    plt.plot(x_values, costs)


if __name__ == "__main__":
    main()
