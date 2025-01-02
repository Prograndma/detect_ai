import matplotlib.pyplot as plt
import numpy as np
from time import time

class ThresholdFinder:
    def __init__(self, save_probs, save_labels, save_classes, save_sequences):
        self.labels: list[int] = []
        self.probs: list[float] = []
        self.classes: list[str] = []
        self.sequences: list[str] = []
        self.unique_classes: dict[str] = {}
        self.save_probs = save_probs
        self.save_labels = save_labels
        self.save_classes = save_classes
        self.save_sequences = save_sequences
        self.reordered_labels = None
        self.reordered_classes = None
        self.reordered_sequences = None
        self.sorted_probs = []
        self.from_the_left = None
        self.from_the_right = None

    def load(self):
        self.probs = []
        with open(f"{self.save_probs}", "r") as f:
            for line in f:
                current_line = line.split(",")
                for prob in current_line:
                    prob = prob.strip()
                    if len(prob) == 0:
                        continue
                    self.probs.append(float(prob))

        self.labels = []
        with open(f"{self.save_labels}", "r") as f:
            for line in f:
                current_line = line.split(",")
                for label in current_line:
                    label = label.strip()
                    if len(label) == 0:
                        continue
                    self.labels.append(int(label))

        self.classes = []
        with open(f"{self.save_classes}", "r") as f:
            for line in f:
                current_line = line.split(",")
                for current_class in current_line:
                    current_class = current_class.strip()
                    if len(current_class) == 0:
                        continue
                    self.classes.append(current_class)
                    self.unique_classes[current_class] = None

        self.sequences = []
        with open(f"{self.save_sequences}", "r", encoding='utf-8-sig') as f:
            for line in f:
                current_line = line.split("\n")
                for current_sequence2 in current_line:
                    if len(current_sequence2) != 0:
                        all_empty = False
                if all_empty:
                    print("All empty")
                    self.sequences.append(", ")
                    continue
                for current_sequence in current_line:
                    current_sequence = current_sequence.strip()
                    all_empty = True

                    if len(current_sequence) == 0:
                        continue
                    self.sequences.append(current_sequence)
                    break
        if len(self.sequences) != len(self.probs):
            raise Exception("Fix the way you load/save your stuff!")
        self._reorganize()

    def add(self, label: int, prob: float, a_class: str, sequence: str):
        self.labels.append(label)
        self.probs.append(prob)
        self.classes.append(a_class)
        sequence = self._clean_sequence(sequence)
        self.sequences.append(sequence)
        self.reordered_labels = None
        self.reordered_classes = None
        self.sorted_probs = None
        self.from_the_left = None
        self.from_the_right = None

    @staticmethod
    def _clean_sequence(seq: str):
        return " ".join(seq.splitlines())

    def _reorganize(self):
        zip_sorted = sorted(zip(self.probs, self.labels, self.classes, self.sequences), key=lambda pair: pair[0])
        self.sorted_probs = [x for x, _, _, _ in zip_sorted]
        self.reordered_labels = [x for _, x, _, _ in zip_sorted]
        self.reordered_classes = [x for _, _, x, _ in zip_sorted]
        self.reordered_sequences = [x for _, _, _, x in zip_sorted]
        self.from_the_left = []
        num_human_seen = 0
        self.from_the_right = []
        num_generated_seen = 0
        for i in range(len(self.reordered_labels)):
            if self.reordered_labels[i] == 0:
                num_human_seen += 1
            if self.reordered_labels[-1 - i] == 1:
                num_generated_seen += 1
            self.from_the_left.append(num_human_seen)
            self.from_the_right.append(num_generated_seen)
        self.from_the_right.reverse()

    def get_false_negative_sample(self, how_many, threshold):
        raise Exception("Currently broken")
        samples = []
        i = 0
        have_done = 0
        while True:
            i += 1
            if i > len(self.from_the_left) or have_done >= how_many:
                break
            if self.from_the_left[i] > self.from_the_left[i - 1]:
                # from_the_left is amount of human samples seen. If that hasn't increased, it's a generated sample
                continue
            if self.sorted_probs[i] > threshold:
                break
            samples.append((self.reordered_classes[i], self.reordered_sequences[i]))
            have_done += 1
        return samples

    def get_false_positive_sample(self, how_many, threshold):
        raise Exception("Currently broken")
        # self._reorganize()
        self.reorganize_modify()
        samples = []
        i = len(self.from_the_right) - 1
        have_done = 0
        while True:
            i -= 1
            if i < 0 or have_done >= how_many:
                break
            if self.from_the_right[i] > self.from_the_right[i + 1]:
                # from_the_right is amount of generated samples seen, coming from the right
                # If that hasn't increased, it's a human sample
                continue
            if self.sorted_probs[i] < threshold:
                break
            samples.append((self.reordered_classes[i], self.reordered_sequences[i]))
            have_done += 1
        return samples

    def save(self):
        self._reorganize()
        with open(f"{self.save_probs}", "w") as f:
            for prob in self.probs:
                f.write(f"{prob}, ")
        with open(f"{self.save_labels}", "w") as f:
            for label in self.labels:
                f.write(f"{label}, ")
        with open(f"{self.save_classes}", "w") as f:
            for a_class in self.classes:
                f.write(f"{a_class}, ")
        with open(f"{self.save_sequences}", "w", encoding='utf-8-sig') as f:
            for seq in self.sequences:
                f.write(f"{seq}\n")

    def find_optimal_f1(self):
        raise Warning(Exception("Depricated, please use and upper and lower threshold"))
        if (self.reordered_labels is None or self.sorted_probs is None or self.from_the_left is None or
                self.from_the_right is None):
            self._reorganize()
        previous_f1 = 0.0
        have_found_f1 = False
        best_threshold = -1111111
        best_precision = 0.0
        best_recall = 0.0
        for i in range(len(self.reordered_labels)):
            total_labeled_human = len(self.sorted_probs[i:])
            total_labeled_human_correctly = sum(self.reordered_labels[i:])

            if total_labeled_human_correctly == 0:  # Not good enough.
                continue
            total_humans_labeled_ai = sum(self.reordered_labels[:i])

            precision = total_labeled_human_correctly / total_labeled_human
            recall = total_labeled_human_correctly / (total_labeled_human_correctly + total_humans_labeled_ai)

            if (precision + recall) == 0:
                continue
            f1 = (2 * precision * recall) / (precision + recall)
            if f1 >= previous_f1:
                previous_f1 = f1
                best_threshold = self.sorted_probs[i]
                best_recall = recall
                best_precision = precision
                have_found_f1 = True
                best_total_labeled_human_correctly = total_labeled_human_correctly
                best_total_labeled_human = total_labeled_human
                best_total_humans_labeled_ai = total_humans_labeled_ai
                best_total_labeled_ai = len(self.probs) - total_labeled_human
                # return (best_threshold, best_precision, best_recall, previous_f1, best_total_labeled_human_correctly,
                # best_total_labeled_human, best_total_humans_labeled_ai, best_total_labeled_ai)

        if not have_found_f1:
            raise Exception("Didn't find a good f1!")
        return (best_threshold, best_precision, best_recall, previous_f1, best_total_labeled_human_correctly,
                best_total_labeled_human, best_total_humans_labeled_ai, best_total_labeled_ai)

    def slow_find_best_upper_and_lower(self):
        if (self.reordered_labels is None or self.sorted_probs is None or self.from_the_left is None or
                self.from_the_right is None):
            self._reorganize()
        best_f1 = 0.0
        best_lower = self.sorted_probs[0]
        best_upper = self.sorted_probs[-1]
        best_lower_index = -1
        best_upper_index = -1
        n = len(self.sorted_probs)
        step = 500
        lower_start = int(n / 7)
        lower_end = int(n / 2)
        upper_start = lower_end + int(n / 6)
        # upper_end = int((n / 10) * 9)
        upper_end = n - 1
        print(f"{lower_start=}({self.sorted_probs[lower_start]})")
        print(f"{lower_end=}({self.sorted_probs[lower_end]})")
        print(f"{upper_start=}({self.sorted_probs[upper_start]})")
        print(f"{upper_end=}({self.sorted_probs[upper_end]})")
        start = time()
        if n < 10:
            return best_lower, best_upper
        for lower in range(lower_start, lower_end, step):
            print(f"\r{lower}/{lower_end} time: {time() - start}", end="")
            for upper in range(upper_start, upper_end, step):
                lower_thresh = self.sorted_probs[lower]
                upper_thresh = self.sorted_probs[upper]
                _, _, f1, _, _, _, _ = self.find_f1_upper_lower_threshold(lower_thresh, upper_thresh)
                if f1 > best_f1:
                    best_f1 = f1
                    best_lower_index = lower
                    best_upper_index = upper
                    best_lower = lower_thresh
                    best_upper = upper_thresh
        print("\n Done!")
        print(f"{best_lower_index=}({self.sorted_probs[best_lower_index]})")
        print(f"{best_upper_index=}({self.sorted_probs[best_upper_index]})")
        return best_lower, best_upper

    def find_f1_upper_lower_threshold(self, lower, upper):
        if (self.reordered_labels is None or self.sorted_probs is None or self.from_the_left is None or
                self.from_the_right is None):
            self._reorganize()

        lower_index = -1
        for i in range(len(self.sorted_probs)):
            if self.sorted_probs[i] > lower:
                lower_index = i
                break

        upper_index = -1
        for i in range(len(self.sorted_probs)):
            if self.sorted_probs[i] > upper:
                upper_index = i
                break

        total_labeled_ai = lower_index
        total_labeled_ai += len(self.sorted_probs[upper_index:])
        total_labeled_human = upper_index - lower_index
        total_labeled_human_correctly = sum(self.reordered_labels[lower_index:upper_index])

        total_labeled_ai_correctly = sum(self.reordered_labels[:lower_index])
        total_labeled_ai_correctly += sum(self.reordered_labels[upper_index:])

        total_ai_labeled_human = total_labeled_human - total_labeled_human_correctly
        total_humans_labeled_ai = total_labeled_ai - total_labeled_ai_correctly
        if total_labeled_human == 0:
            return 0.0, 0.0, 0.0, 0, 0, 0, 0

        num_ai    = total_labeled_ai_correctly    + total_ai_labeled_human
        num_human = total_labeled_human_correctly + total_humans_labeled_ai
        assert(num_ai + num_human == len(self.reordered_labels))
        precision_human = total_labeled_human_correctly / (total_labeled_human_correctly + total_ai_labeled_human)
        precision_ai    = total_labeled_ai_correctly    / (total_labeled_ai_correctly    + total_humans_labeled_ai)

        recall_human    = total_labeled_human_correctly / (total_labeled_human_correctly + total_humans_labeled_ai)
        recall_ai       = total_labeled_ai_correctly    / (total_labeled_ai_correctly    + total_ai_labeled_human)

        weighted_precision = ((num_ai * precision_ai) + (num_human * precision_human)) / (len(self.reordered_labels))
        weighted_recall    = ((num_ai * recall_ai)    + (num_human * recall_human))    / (len(self.reordered_labels))

        f1 = (2 * weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)

        return weighted_precision, weighted_recall, f1, total_labeled_human_correctly, total_labeled_human, total_humans_labeled_ai, total_labeled_ai

    def visualize(self):
        # Split into bins
        num_bins = 320
        bin_size = len(self.reordered_labels) // num_bins
        bin_proportions = []
        x_values = []
        x_indices = []
        for i in range(num_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < num_bins - 1 else len(self.reordered_labels)
            bin_labels = self.reordered_labels[start:end]
            x_values.append(self.sorted_probs[start])
            x_indices.append(i)
            proportion_ones = sum(bin_labels) / len(bin_labels)
            bin_proportions.append(proportion_ones)

        # Map proportions to colors (Red for 1's, Green for 0's)
        colors = [(p, 1 - p, 0) for p in bin_proportions]  # RGB tuples

        plt.figure(figsize=(12, 6))

        plt.imshow([colors], aspect="auto", extent=(float(self.sorted_probs[0]), float(self.sorted_probs[-1]), 0.0, 1.0))

        num_ticks = 20
        tick_positions = np.linspace(self.sorted_probs[0], self.sorted_probs[-1], num_ticks)  # Evenly spaced positions
        tick_indices = np.linspace(0, len(self.sorted_probs) - 1, num_ticks, dtype=int)  # Corresponding indices
        tick_labels = [f"{self.sorted_probs[idx]:.8f}" for idx in tick_indices]  # Labels from sorted_probs

        plt.xticks(tick_positions, tick_labels)

        plt.gca().set_yticks([])  # Hide y-axis
        plt.xlabel("Sorted Data (Highest generated score to Lowest)")
        plt.title("Model Performance Visualization")
        plt.show()

    def visualize_for_each_class(self):
        # TODO: Fix this, it's good functionality
        # self._reorganize()
        for current_class in self.unique_classes.keys():
            if current_class == "wikipedia":
                continue
            current_labels = []
            current_probs = []
            for i, thing in enumerate(self.reordered_classes):
                if thing == current_class or thing == "wikipedia":
                    current_labels.append(self.reordered_labels[i])
                    current_probs.append(self.sorted_probs[i])

            num_bins = 320
            bin_size = len(current_labels) // num_bins
            bin_proportions = []

            for i in range(num_bins):
                start = i * bin_size
                end = (i + 1) * bin_size if i < num_bins - 1 else len(current_labels)
                bin_labels = current_labels[start:end]
                if len(bin_labels) == 0:
                    bin_proportions.append(1.0)
                    continue
                proportion_ones = sum(bin_labels) / len(bin_labels)
                bin_proportions.append(proportion_ones)

            colors = [(p, 1 - p, 0) for p in bin_proportions]  # RGB tuples

            fig, ax = plt.subplots(figsize=(10, 1))

            ax.imshow([colors], aspect="auto", extent=[current_probs[0], current_probs[-1], 0, 1])

            color_map = plt.imshow([colors], aspect="auto", extent=[current_probs[0], current_probs[-1], 0, 1])
            plt.gca().set_yticks([])  # Hide y-axis
            plt.xlabel("Sorted Data (Highest generated score to Lowest)")
            plt.title(f"Model Performance Visualization for text generated by {current_class}")
            from matplotlib.colors import Normalize
            from matplotlib.cm import ScalarMappable

            norm = Normalize(vmin=0, vmax=1)  # Normalize proportions (0 to 1)
            sm = ScalarMappable(cmap="RdYlGn", norm=norm)  # Create mappable object
            sm.set_array([])  # Needed for the color bar

            plt.show()

    def amount_classes(self):
        count_dict = {}
        for a_class in self.classes:
            if a_class not in count_dict:
                count_dict[a_class] = 0
            count_dict[a_class] += 1

        return count_dict


if __name__ == "__main__":
    base_dir = "gpt/"
    end = ".txt"
    modified = f"{base_dir}reordered_"
    pb = f"{base_dir}probs{end}"
    sorted_pb = f"{base_dir}sorted_probs{end}"

    labels = f"{base_dir}labels{end}"
    re_labels = f"{modified}labels{end}"

    classes = f"{base_dir}classes{end}"
    re_classes = f"{modified}classes{end}"

    sequences = f"{base_dir}sequences{end}"
    re_sequences = f"{modified}sequences{end}"
    thresh = ThresholdFinder(pb, labels, classes, sequences, sorted_pb, re_labels, re_classes, re_sequences)

    thresh.load()
    info_counts = thresh.amount_classes()
    for key in info_counts.keys():
        print(f"{key}: {info_counts[key]}")
    main_lower, main_upper = thresh.slow_find_best_upper_and_lower()
    (main_precision, main_recall, main_f1, num_humans_labeled_human, num_labeled_human,
     num_humans_labeled_ai, num_labeled_ai) = thresh.find_f1_upper_lower_threshold(main_lower, main_upper)
    print("###############################################################")
    print(f"LOWER: {main_lower}\n"
          f"UPPER: {main_upper}")
    print(f"Best Precision: {main_precision}\n"
          f"Best Recall   : {main_recall}\n"
          f"Best F1       : {main_f1}\n"
          f"Num humans labeled human : {num_humans_labeled_human}\n"
          f"Num humans labeled AI    : {num_humans_labeled_ai}\n"
          f"Num AI labeled human     : {num_labeled_human - num_humans_labeled_human}\n"
          f"Num AI labeled AI        : {num_labeled_ai - num_humans_labeled_ai}")
    thresh.visualize()
    thresh.visualize_for_each_class()