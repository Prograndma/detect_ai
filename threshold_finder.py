class ThresholdFinder:
    def __init__(self, save_probs, save_labels):
        self.labels: list[int] = []
        self.probs: list[float] = []
        self.save_probs = save_probs
        self.save_labels = save_labels
        self.reordered_labels = None
        self.sorted_probs = []
        self.from_the_left = None
        self.from_the_right = None

    def add(self, label: int, prob: float):
        self.labels.append(label)
        self.probs.append(prob)
        self.reordered_labels = None
        self.sorted_probs = None
        self.from_the_left = None
        self.from_the_right = None

    def _reorganize(self):
        zip_sorted = sorted(zip(self.probs, self.labels), key=lambda pair: pair[0])
        self.sorted_guesses = [x for x, _ in zip_sorted]
        self.reordered_labels = [x for _, x in zip_sorted]
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
        with open(f"{self.save_probs}", "w") as f:
            f.write(self.probs)
        with open(f"{self.save_labels}", "w") as f:
            f.write(self.labels)

    # Todo: Yeah so this probably is completely borked. And useless. But! If it saves its data I'll update it to be
    #  useful.
    # Todo: Make a way to visualize the vector, like, red is generated and green is human. Or like, the amount of
    #  shading corresponds with how many in the group of 1,000 are red or human. This way I can visualize if my way of
    #  splitting up the text is good or not. Ideally the sorted probs would sort most or all of the generated content to
    #  one side