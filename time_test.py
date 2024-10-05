import time

def mod_sleep(duration):
    start_time = time.perf_counter()
    curr_time = start_time
    while (curr_time - start_time) <= duration:
        curr_time = time.perf_counter()