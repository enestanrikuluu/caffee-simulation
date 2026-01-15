import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


class RealisticCafeDataGenerator:
    def __init__(self, random_seed=42):
        self.rng = np.random.default_rng(random_seed)
    
    def generate_period_data(
        self,
        period_name: str,
        n_customers: int,
        lambda_per_minute: float,
        p_drink: float,
        drink_service_params: dict,
        food_service_params: dict,
        rush_factor: float = 1.0,
    ) -> pd.DataFrame:
        service_types = self.rng.choice(
            ["drink", "food"], 
            size=n_customers, 
            p=[p_drink, 1 - p_drink]
        )
        
        service_times_sec = np.zeros(n_customers)
        
        for i in range(n_customers):
            if service_types[i] == "drink":
                base_mean = drink_service_params["mean_sec"]
                base_std = drink_service_params["std_sec"]
                min_time = drink_service_params["min_sec"]
                max_time = drink_service_params["max_sec"]
                
                service_times_sec[i] = self.rng.normal(base_mean, base_std)
                service_times_sec[i] = np.clip(service_times_sec[i], min_time, max_time)
            else:
                distribution = food_service_params["distribution"]
                if distribution == "lognormal":
                    mu = food_service_params["mu"]
                    sigma = food_service_params["sigma"]
                    service_times_sec[i] = self.rng.lognormal(mu, sigma)
                elif distribution == "gamma":
                    shape = food_service_params["shape"]
                    scale = food_service_params["scale"]
                    service_times_sec[i] = self.rng.gamma(shape, scale)
                
                min_time = food_service_params.get("min_sec", 60)
                max_time = food_service_params.get("max_sec", 400)
                service_times_sec[i] = np.clip(service_times_sec[i], min_time, max_time)
        
        mean_interarrival_min = 1.0 / lambda_per_minute
        mean_interarrival_sec = mean_interarrival_min * 60
        
        shape = 2.0 + rush_factor
        scale = mean_interarrival_sec / shape
        
        interarrival_times_sec = self.rng.gamma(shape, scale, size=n_customers)
        interarrival_times_sec = np.maximum(interarrival_times_sec, 5.0)
        
        arrival_times_sec = np.cumsum(interarrival_times_sec)
        arrival_times_sec = arrival_times_sec - arrival_times_sec[0]
        
        start_time = datetime(2024, 1, 1, 8, 0, 0)
        
        arrival_datetimes = [start_time + timedelta(seconds=float(t)) for t in arrival_times_sec]
        
        queue_position = []
        service_start_times = []
        service_end_times = []
        barista_available_at = 0.0
        
        for i in range(n_customers):
            arrival_sec = arrival_times_sec[i]
            
            if arrival_sec > barista_available_at:
                service_start_sec = arrival_sec
                queue_position.append(0)
            else:
                service_start_sec = barista_available_at
                queue_position.append(1)
            
            service_end_sec = service_start_sec + service_times_sec[i]
            barista_available_at = service_end_sec
            
            service_start_times.append(start_time + timedelta(seconds=float(service_start_sec)))
            service_end_times.append(start_time + timedelta(seconds=float(service_end_sec)))
        
        df = pd.DataFrame({
            "customer_id": range(1, n_customers + 1),
            "arrival_time": arrival_datetimes,
            "service_start_time": service_start_times,
            "service_end_time": service_end_times,
            "interarrival_sec": interarrival_times_sec,
            "service_time_sec": service_times_sec,
            "service_type": service_types,
            "queue_at_arrival": queue_position,
        })
        
        df["wait_time_sec"] = (df["service_start_time"] - df["arrival_time"]).dt.total_seconds()
        df["system_time_sec"] = (df["service_end_time"] - df["arrival_time"]).dt.total_seconds()
        
        df["interarrival_min"] = df["interarrival_sec"] / 60.0
        df["service_time_min"] = df["service_time_sec"] / 60.0
        df["wait_time_min"] = df["wait_time_sec"] / 60.0
        df["system_time_min"] = df["system_time_sec"] / 60.0
        
        return df
    
    def generate_all_periods(self, output_dir: Path):
        output_dir.mkdir(exist_ok=True)
        
        periods_config = {
            "Monday_Morning": {
                "n_customers": 120,
                "lambda_per_minute": 1.8,
                "p_drink": 0.25,
                "drink_service_params": {
                    "mean_sec": 45, "std_sec": 12, "min_sec": 25, "max_sec": 85
                },
                "food_service_params": {
                    "distribution": "lognormal",
                    "mu": 5.1, "sigma": 0.35,
                    "min_sec": 100, "max_sec": 350
                },
                "rush_factor": 0.8,
            },
            "Monday_Afternoon": {
                "n_customers": 100,
                "lambda_per_minute": 1.2,
                "p_drink": 0.55,
                "drink_service_params": {
                    "mean_sec": 50, "std_sec": 15, "min_sec": 28, "max_sec": 90
                },
                "food_service_params": {
                    "distribution": "gamma",
                    "shape": 8.0, "scale": 20.0,
                    "min_sec": 90, "max_sec": 280
                },
                "rush_factor": 1.2,
            },
            "Monday_Evening": {
                "n_customers": 80,
                "lambda_per_minute": 1.0,
                "p_drink": 0.35,
                "drink_service_params": {
                    "mean_sec": 48, "std_sec": 14, "min_sec": 30, "max_sec": 85
                },
                "food_service_params": {
                    "distribution": "lognormal",
                    "mu": 4.9, "sigma": 0.3,
                    "min_sec": 85, "max_sec": 240
                },
                "rush_factor": 1.0,
            },
            "Tuesday_Morning": {
                "n_customers": 150,
                "lambda_per_minute": 2.0,
                "p_drink": 0.87,
                "drink_service_params": {
                    "mean_sec": 42, "std_sec": 11, "min_sec": 25, "max_sec": 75
                },
                "food_service_params": {
                    "distribution": "lognormal",
                    "mu": 4.8, "sigma": 0.25,
                    "min_sec": 80, "max_sec": 200
                },
                "rush_factor": 0.5,
            },
            "Tuesday_Afternoon": {
                "n_customers": 130,
                "lambda_per_minute": 1.5,
                "p_drink": 0.72,
                "drink_service_params": {
                    "mean_sec": 46, "std_sec": 13, "min_sec": 27, "max_sec": 80
                },
                "food_service_params": {
                    "distribution": "gamma",
                    "shape": 6.5, "scale": 22.0,
                    "min_sec": 95, "max_sec": 260
                },
                "rush_factor": 0.9,
            },
            "Tuesday_Evening": {
                "n_customers": 90,
                "lambda_per_minute": 0.9,
                "p_drink": 0.42,
                "drink_service_params": {
                    "mean_sec": 52, "std_sec": 16, "min_sec": 30, "max_sec": 95
                },
                "food_service_params": {
                    "distribution": "lognormal",
                    "mu": 5.2, "sigma": 0.4,
                    "min_sec": 110, "max_sec": 380
                },
                "rush_factor": 1.1,
            },
        }
        
        print("Generating Realistic Café Mock Data")
        print("=" * 80)
        
        for period_name, config in periods_config.items():
            print(f"\nGenerating {period_name}...")
            df = self.generate_period_data(period_name, **config)
            
            output_file = output_dir / f"{period_name}_mock.csv"
            df.to_csv(output_file, index=False)
            
            print(f"  ✓ {len(df)} customers")
            print(f"    Service times: {df['service_time_sec'].mean():.1f}±{df['service_time_sec'].std():.1f} sec "
                  f"(range: {df['service_time_sec'].min():.1f}-{df['service_time_sec'].max():.1f})")
            print(f"    Interarrivals: {df['interarrival_sec'].mean():.1f}±{df['interarrival_sec'].std():.1f} sec")
            print(f"    Drink ratio: {(df['service_type']=='drink').mean():.2%}")
            print(f"    Mean wait: {df['wait_time_sec'].mean():.1f} sec")
            print(f"    Saved to: {output_file}")
        
        print("\n" + "=" * 80)
        print("Mock data generation complete!")
        print(f"Output directory: {output_dir}")


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "simulation_data" / "mock"
    
    generator = RealisticCafeDataGenerator(random_seed=42)
    generator.generate_all_periods(output_dir)


if __name__ == "__main__":
    main()
