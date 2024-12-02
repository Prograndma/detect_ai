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
        print(f"Middle value{middle}. Apply function of score = (({middle}) x method(input)) ^ 2")
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
