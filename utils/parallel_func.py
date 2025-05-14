#!/usr/bin/python
# -*- coding:utf-8 -*-
from multiprocessing import Process, Queue
from .logger import print_log


def func_wrapper(func, task_queue, result_queue):
    while True:
        (i, inputs) = task_queue.get()
        if inputs is None: break
        try:
            outputs = func(*inputs)
        except Exception as e:
            outputs = None
            print_log(f'{e}, {i}, inputs {inputs} failed', level='WARN')
        result_queue.put((i, outputs))


def parallel_func(func, inputs, n_cpus, unordered=False):
    task_queue = Queue()
    result_queue = Queue()

    # create worker process
    processes = []
    for _ in range(n_cpus):
        p = Process(target=func_wrapper, args=(func, task_queue, result_queue))
        processes.append(p)
        p.start()

    # Distribute tasks to workers
    for i, args in enumerate(inputs): task_queue.put((i, args))

    # Add a sentinel (None) to signal workers to exit
    for _ in range(n_cpus): task_queue.put((-1, None)) # end

    # Collect results from workers
    if unordered: # don't care ordering
        for _ in inputs:
            _, outputs = result_queue.get()
            # print_log(f'queue size: {result_queue.qsize()}')
            yield outputs
    else: # the same ordering as inputs
        stored_outputs, current = {}, 0
        for _ in inputs:
            i, outputs = result_queue.get()
            stored_outputs[i] = outputs
            if current in stored_outputs:
                yield stored_outputs.pop(current)
                current += 1
        while len(stored_outputs):
            yield stored_outputs.pop(current)
            current += 1
    
    # Ensure all processes have finished
    for p in processes:
        p.join()

