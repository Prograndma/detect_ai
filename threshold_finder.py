import matplotlib.pyplot as plt


class ThresholdFinder:
    def __init__(self, save_probs, save_labels, save_classes, save_sorted_probs, save_reordered_labels, save_reordered_classes):
        self.labels: list[int] = []
        self.probs: list[float] = []
        self.classes: list[str] = []
        self.save_probs = save_probs
        self.save_labels = save_labels
        self.save_classes = save_classes
        self.save_sorted_probs = save_sorted_probs
        self.save_reordered_labels = save_reordered_labels
        self.save_reordered_classes = save_reordered_classes
        self.reordered_labels = None
        self.reordered_classes = None
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
        self._reorganize()

    def add(self, label: int, prob: float, a_class: str):
        self.labels.append(label)
        self.probs.append(prob)
        self.classes.append(a_class)
        self.reordered_labels = None
        self.reordered_classes = None
        self.sorted_probs = None
        self.from_the_left = None
        self.from_the_right = None

    def _reorganize(self):
        zip_sorted = sorted(zip(self.probs, self.labels, self.classes), key=lambda pair: pair[0])
        self.sorted_probs = [x for x, _, _ in zip_sorted]
        self.reordered_labels = [x for _, x, _ in zip_sorted]
        self.reordered_classes = [x for _, _, x in zip_sorted]
        middle = sum(self.sorted_probs) / len(self.sorted_probs)
        print(f"Middle value: {middle}.\n"
              f"Apply function of\n"
              f"score = (({middle}) x method(input)) ^ 2")
        self.sorted_probs = [(prob - middle) ** 2 for prob in self.sorted_probs]
        zip_sorted = sorted(zip(self.sorted_probs, self.reordered_labels, self.reordered_classes), key=lambda pair: pair[0])
        self.sorted_probs = [x for x, _, _ in zip_sorted]
        self.reordered_labels = [x for _, x, _ in zip_sorted]
        self.reordered_classes = [x for _, _, x in zip_sorted]
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

    def save(self):
        self._reorganize()
        with open(f"{self.save_reordered_classes}", "w") as f:
            for thing in self.reordered_classes:
                f.write(f"{thing}, ")
        with open(f"{self.save_sorted_probs}", "w") as f:
            for thing in self.sorted_probs:
                f.write(f"{thing}, ")
        with open(f"{self.save_reordered_labels}", "w") as f:
            for thing in self.reordered_labels:
                f.write(f"{thing}, ")
        with open(f"{self.save_probs}", "w") as f:
            for prob in self.probs:
                f.write(f"{prob}, ")
        with open(f"{self.save_labels}", "w") as f:
            for label in self.labels:
                f.write(f"{label}, ")
        with open(f"{self.save_classes}", "w") as f:
            for a_class in self.classes:
                f.write(f"{a_class}, ")

    def find_optimal_f1(self):
        if (self.reordered_labels is None or self.sorted_probs is None or self.from_the_left is None or
                self.from_the_right is None):
            self._reorganize()
        previous_f1 = 0.0
        have_found_f1 = False
        best_threshold = -1111111
        best_precision = 0.0
        best_recall = 0.0
        for i in range(len(self.reordered_labels)):
            total_positive = len(self.sorted_probs[i:])
            true_positive = sum(self.reordered_labels[i:])

            if true_positive == 0:  # Not good enough.
                continue
            false_negative = sum(self.reordered_labels[:i])

            precision = true_positive / total_positive
            recall = true_positive / (true_positive + false_negative)

            if (precision + recall) == 0:
                continue
            f1 = (2 * precision * recall) / (precision + recall)
            if f1 >= previous_f1:
                previous_f1 = f1
                best_threshold = self.sorted_probs[i]
                best_recall = recall
                best_precision = precision
                have_found_f1 = True

        if not have_found_f1:
            raise Exception("Didn't find a good f1!")
        return best_threshold, best_precision, best_recall, previous_f1


    def visualize(self):
        # Split into bins
        num_bins = 320
        bin_size = len(self.reordered_labels) // num_bins
        bin_proportions = []

        for i in range(num_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < num_bins - 1 else len(self.reordered_labels)
            bin_labels = self.reordered_labels[start:end]
            proportion_ones = sum(bin_labels) / len(bin_labels)
            bin_proportions.append(proportion_ones)

        # Map proportions to colors (Red for 1's, Green for 0's)
        colors = [(p, 1 - p, 0) for p in bin_proportions]  # RGB tuples

        # Create a color bar visualization
        # plt.figure(figsize=(10, 1))
        fig, ax = plt.subplots(figsize=(10, 1))

        ax.imshow([colors], aspect="auto", extent=[self.sorted_probs[0], self.sorted_probs[-1], 0, 1])

        color_map = plt.imshow([colors], aspect="auto", extent=[self.sorted_probs[0], self.sorted_probs[-1], 0, 1])
        plt.gca().set_yticks([])  # Hide y-axis
        plt.xlabel("Sorted Data (Highest generated score to Lowest)")
        plt.title("Model Performance Visualization")
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        norm = Normalize(vmin=0, vmax=1)  # Normalize proportions (0 to 1)
        sm = ScalarMappable(cmap="RdYlGn", norm=norm)  # Create mappable object
        sm.set_array([])  # Needed for the color bar

        fig, ax = plt.subplots(figsize=(10, 1))
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]

        plt.colorbar(sm, cax=cax, label="Proportion of AI (Red) to human (Green)")
        # plt.colorbar(sm, label="Proportion of 1's (Green)")

        plt.show()
        # plt.colorbar(plt.cm.ScalarMappable(cmap="RdYlGn"), label="Proportion of 1's (Green)")
        # plt.show()


if __name__ == "__main__":
    thresh = ThresholdFinder("probs.txt", "labels.txt", "classes.txt",
                             "sorted_probs.txt", "reordered_labels.txt", "reordered_classes.txt")

    thresh.load()
    main_threshold, main_precision, main_recall, main_f1 = thresh.find_optimal_f1()
    print(f"Best Threshold: {main_threshold}\n"
          f"Best Precision: {main_precision}\n"
          f"Best Recall   : {main_recall}\n"
          f"Best F1       : {main_f1}")
