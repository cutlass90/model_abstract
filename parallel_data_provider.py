import time
import random
import multiprocessing 
import ctypes

from multiprocessing.sharedctypes import Value

import numpy as np

class ParallelDataProvider:

    def __init__(self, n_processes, files, capacity,
                 batch_size, shuffle=True, verbose=False,
                 name='ParallelDataProvider'):

        """
        Initializes an instance of parallel data provider capable of processing 
        training batches in separate processes.

        After creating an instance of this class, `read` function has to be set.
        This function raises `NonImplementedError` by default. After being set
        (i.e. by something like `reader.read = external_read`) `n_processes` 
        processes will be launched.

        Args:
            n_processes: number of processes to use.
            files: list of files to be fed into read function (technically, this
                could be an iterable of any objects).
            capacity: maximum capacity of the queue.
            batch_size: minibatch_size.
            shuffle: if set to True, queue is shuffled before providing
            every new batch. Defaults to True.
            verbose: if set to True, periodically prints statistics over the queue.
                Default to False. 
            name: optional name. Used in logging.
        """

        self.n_processes = n_processes
        self.files = files
        self.capacity = capacity
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.name = name
        self.launched = False

        self._create_queue()

    def _create_queue(self):
        self.manager = multiprocessing.Manager()
        self.queue = self.manager.list()
        self.should_stop = Value(ctypes.c_bool, False)

    @property
    def read(self):
        raise NotImplementedError

    @read.setter
    def read(self, read_fn):

        """This function is to be set by user. 
           It should take an element of self.files 
           and return an iterable containing the processed objects.
        """
        self._read = read_fn
        self._launch_processes()

    @property
    def qsize(self):
        return len(self.queue)

    def _process_single(self):

        if isinstance(self.files, list):
            file_ = random.choice(self.files)
        elif isinstance(self.files, dict):
            category = random.choice(list(self.files.keys()))
            file_ = random.choice(self.files[category])
        else:
            raise TypeError("Unrecognized type: {}.".format(type(self.files)))

        processed_file = list(self._read(file_))
        return processed_file

    def _target(self, queue):
        while not self.should_stop.value:
            
            processed_files = self._process_single()

            while self.qsize >= self.capacity or len(processed_files) < self.batch_size:

                if self.should_stop.value:
                    break

                # keep processing if queue is full 
                # stop, if local queue is bigger than capacity / n_processes
                if len(processed_files) < self.capacity / self.n_processes:
                    processed_files.extend(self._process_single())

                time.sleep(0.001)
            
            # if qsize < capacity, fill the queue with processed files
            queue.extend(processed_files)
            
    def _launch_processes(self):
        self.processes = [multiprocessing.Process(
                          target=self._target, args=(self.queue,))
                          for _ in range(self.n_processes)]
        for p in self.processes:
            p.start()
        self.lauched = True

    def new_batch(self):

        """Returns a new batch by drawing objects from queue. If `self.shuffle` 
           is True, queue is shuffled on every call."""

        if not self.lauched:
            raise ValueError("`read` function must be set before `new_batch` is called")

        while not self.qsize >= self.batch_size:
            time.sleep(0.0001)

        if self.shuffle:
            random.shuffle(self.queue)

        if self.verbose:
            print('Queue size of the {}: {}'.format(self.name, self.qsize))

        batch = [self.queue.pop() for _ in range(self.batch_size)]
        return tuple(np.array(a) for a in zip(*batch))

    def free(self):
        self.should_stop.value = True
        for p in self.processes:
            p.join(10)
