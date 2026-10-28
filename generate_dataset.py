
import pandas as pd
import numpy as np

np.random.seed(42)

positions = ["Defender", "Forward", "Goalkeeper", "Midfielder"]
rows = []

for position in positions:
    for i in range(903):

        if position == "Defender":
            goals = np.random.poisson(1.2)
            assists = np.random.poisson(2)
            passes = np.random.randint(300, 900)
            pass_accuracy = np.random.uniform(75, 95)
            shots = np.random.uniform(0, 1.5)
            tackles = np.random.uniform(1.5, 5)
            saves = 0

        elif position == "Forward":
            goals = np.random.poisson(12)
            assists = np.random.poisson(5)
            passes = np.random.randint(100, 500)
            pass_accuracy = np.random.uniform(65, 90)
            shots = np.random.uniform(2, 6)
            tackles = np.random.uniform(0, 2)
            saves = 0

        elif position == "Goalkeeper":
            goals = np.random.poisson(0.1)
            assists = np.random.poisson(0.5)
            passes = np.random.randint(100, 600)
            pass_accuracy = np.random.uniform(55, 90)
            shots = 0
            tackles = np.random.uniform(0, 1)
            saves = np.random.uniform(2, 6)

        else:
            goals = np.random.poisson(5)
            assists = np.random.poisson(7)
            passes = np.random.randint(400, 1200)
            pass_accuracy = np.random.uniform(75, 95)
            shots = np.random.uniform(0.5, 3)
            tackles = np.random.uniform(1, 4)
            saves = 0

        rows.append({
            "age": np.random.randint(18, 36),
            "height_cm": np.random.randint(165, 201),
            "weight_kg": np.random.randint(58, 96),
            "matches_played": np.random.randint(5, 39),
            "goals": goals,
            "assists": assists,
            "passes_completed": passes,
            "pass_accuracy_percent": round(pass_accuracy, 2),
            "shots_per_match": round(shots, 2),
            "tackles_per_match": round(tackles, 2),
            "saves_per_match": round(saves, 2),
            "class": position
        })

data = pd.DataFrame(rows)

filename = "football_player_position_synthetic.csv"
data.to_csv(filename, index=False)

print("Dataset generated successfully!")
print("Dataset shape:", data.shape)
print("\nFirst five records:")
print(data.head())
print("\nPlayers by position:")
print(data["class"].value_counts())
