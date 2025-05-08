import sqlite3
import random
from datetime import datetime, timedelta

# Connect to database
conn = sqlite3.connect('information.db')
cursor = conn.cursor()

# Student names (modify as needed)
students = ['Burhanuddin', 'Mudasir', 'Wajid', 'Wali', 'Hasnain', 'Ali']

def generate_april_days(count):
    """Generate random days in April 2025"""
    return random.sample(range(1, 31), min(count, 30))  # Max 30 unique days

def generate_work_time():
    """Generate realistic work hours (8AM-5PM)"""
    hour = random.choices(
        [8,9,10,11,12,13,14,15,16], 
        weights=[5,20,15,5,5,20,15,5,5],  # Peaks at 9AM and 1PM
        k=1
    )[0]
    return f"{hour:02d}:{random.randint(0,59):02d}"

# Generate 60 new attendance records
added_count = 0
april_days = generate_april_days(60)

for day in april_days:
    date = f"2025-04-{day:02d}"
    
    # Each student has 70% chance of attending on each day
    for student in students:
        if random.random() < 0.7:  # 70% attendance probability
            time = generate_work_time()
            
            # Insert only if not already exists (same student+date)
            cursor.execute(
                """INSERT INTO Attendance 
                (NAME, Time, Date) VALUES (?, ?, ?)""",
                (student, time, date)
            )
            
            if cursor.rowcount > 0:  # If new record was inserted
                added_count += 1
                print(f"Added: {student} on {date} at {time}")

conn.commit()
conn.close()
print(f"\nSuccessfully added {added_count} new attendance records for April 2025!")