import pandas as pd
import random

labs = ["Lab1", "Lab2", "Lab3", "Lab4"]
rows = 300

data = []

for i in range(rows):
    time = f"{random.randint(8,18)}:00"
    lab = random.choice(labs)
    occupancy = random.randint(0, 40)
    temperature = random.randint(20, 32)

    if occupancy > 25 and temperature > 28:
        cooling = "High"
    elif occupancy > 10 and temperature > 24:
        cooling = "Medium"
    else:
        cooling = "Low"

    data.append([time, lab, occupancy, temperature, cooling])

df = pd.DataFrame(data, columns=[
    "time",
    "lab_id",
    "occupancy",
    "temperature",
    "cooling_needed"
])

df.to_csv("hvac_data.csv", index=False)

print("Dataset generated successfully: hvac_data.csv")
