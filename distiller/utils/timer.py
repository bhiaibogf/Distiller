import time

last_time = time.time()


def update_time(reason):
    new_time = time.time()
    global last_time
    print(f'{reason} use {new_time - last_time:.4}s\n')
    last_time = new_time
