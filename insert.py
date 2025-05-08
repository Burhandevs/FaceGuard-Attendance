import sqlite3
import random
from datetime import datetime, timedelta

# Connect to database
conn = sqlite3.connect('information.db')
cursor = conn.cursor()

# Configuration
TOTAL_ENTRIES = 100  # Number of new recognition logs to add
PASS_RATE = 0.95     # 95% passing rate

# Student names (should match your training data)
KNOWN_STUDENTS = ['Burhanuddin', 'Mudasir', 'Wajid', 'Wali', 'Hasnain', 'Ali']

# Liveness check options
LIVENESS_PASS = ['Passed']
LIVENESS_FAIL = ['Failed - Eyes closed', 'Failed - Photo', 'Failed - Mask']

def generate_april_2025_date():
    """Generate random date in April 2025"""
    day = random.randint(1, 30)
    return f"2025-04-{day:02d}"

def generate_timestamp(base_date):
    """Generate realistic timestamp (8AM-5PM)"""
    hour = random.choices(
        [8,9,10,11,12,13,14,15,16],
        weights=[5,20,15,5,5,20,15,5,5],  # More at 9AM and 1PM
        k=1
    )[0]
    return f"{base_date} {hour:02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"

print("Adding high-quality recognition logs...")
added_count = 0

for _ in range(TOTAL_ENTRIES):
    # Generate random April 2025 date
    date = generate_april_2025_date()
    timestamp = generate_timestamp(date)
    
    # 95% chance of successful recognition
    if random.random() < PASS_RATE:
        name = random.choice(KNOWN_STUDENTS)
        recognition_result = "Correct"
        is_spoofing = 0
        confidence = round(random.uniform(0.90, 0.99), 2)  # High confidence
        liveness_check = random.choice(LIVENESS_PASS)
    else:
        name = "Unknown"
        recognition_result = "Incorrect"
        is_spoofing = 1
        confidence = round(random.uniform(0.10, 0.40), 2)  # Low confidence
        liveness_check = random.choice(LIVENESS_FAIL)
    
    # Insert recognition log
    cursor.execute(
        """INSERT INTO RecognitionLogs 
        (timestamp, name, recognition_result, is_spoofing, confidence, liveness_check)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (timestamp, name, recognition_result, is_spoofing, confidence, liveness_check)
    )
    
    added_count += 1
    if added_count % 10 == 0:
        print(f"Added {added_count} entries...")

conn.commit()
conn.close()
print(f"\nSuccessfully added {added_count} recognition logs with {PASS_RATE*100}% pass rate!")