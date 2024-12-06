from time import time


class LoadingBar:
    def __init__(self, start, steps, length, batch_size=1, empty="-", fill="█", caps="|"):
        self.start = start
        self.steps = steps
        self.length = length
        self.empty = empty
        self.fill = fill
        self.caps = caps
        self.num_fill = -1
        self.batch_size = batch_size

    def update(self, position):
        total = self.length * self.batch_size
        number_z_fill = len(str(total))
        if position % (self.length // self.steps) == 0:
            self.num_fill += 1

        time_taken = time() - self.start
        hours_taken, minutes_taken = self._hours_minutes(time_taken)
        hours_taken = str(int(hours_taken)).zfill(2)
        minutes_taken = str(int(minutes_taken)).zfill(2)
        iters_completed = str(position * self.batch_size).zfill(number_z_fill)
        if position == 0:
            average_time_per_batch = time_taken
        else:
            average_time_per_batch = (time_taken / position) * self.batch_size
        hours_remaining, minutes_remaining = self._hours_minutes(self.length * (average_time_per_batch / self.batch_size) - time_taken)
        hours_remaining = str(int(hours_remaining)).zfill(2)
        minutes_remaining = str(int(minutes_remaining)).zfill(2)

        loading_full = self.fill * self.num_fill
        loading_empty = self.empty * ((self.steps - 1) - self.num_fill)
        bar = ""
        bar += f"\r|{loading_full}{loading_empty}| did {iters_completed}/{total}! "
        bar += f"Elapsed Time {hours_taken}:{minutes_taken}. "
        bar += f"Taking {average_time_per_batch:.2f} seconds/batch. "
        bar += f"Estimated remaining time {hours_remaining}:{minutes_remaining} "
        return bar

    def finish(self):
        time_taken = time() - self.start
        hours_taken, minutes_taken = self._hours_minutes(time_taken)
        hours_taken = str(int(hours_taken)).zfill(2)
        minutes_taken = str(int(minutes_taken)).zfill(2)
        total = self.length * self.batch_size
        number_z_fill = len(str(total))
        iters_completed = str(total).zfill(number_z_fill)
        average_time_per_batch = (time_taken / self.length) * self.batch_size
        bar = f"\r|{self.fill * (self.steps - 1)}| finished {iters_completed}/{total}! "
        bar += f"Took {hours_taken}:{minutes_taken}. "
        bar += f"Taking {average_time_per_batch:.2f} seconds/batch.\n"
        bar += f"All done!\n"
        return bar

    @staticmethod
    def _hours_minutes(time_to_convert):
        hours = time_to_convert / 60 / 60
        minutes = (hours % 1) * 60
        return hours, minutes

