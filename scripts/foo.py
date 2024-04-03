import multiprocessing as mp


def process(q, qout, iolock):
    from time import sleep
    while True:
        stuff = q.get()
        if stuff is None:
            break
        with iolock:
            print("processing", stuff)
        # sleep(1)
        qout.put(stuff)

if __name__ == '__main__':
    NCORE = 4
    q = mp.Queue(maxsize=NCORE)
    qout = mp.Queue()
    iolock = mp.Lock()
    pool = mp.Pool(NCORE, initializer=process, initargs=(q, qout, iolock))
    for stuff in range(20):
        q.put(stuff)  # blocks until q below its max size
        with iolock:
            print("queued", stuff)
    for _ in range(NCORE):  # tell workers we're done
        q.put(None)
    pool.close()
    pool.join()