import schedule
import time
import os

def job():
    os.system('python retrain.py')

schedule.every().day.at("00:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)