import numpy as np

class TextDataset:
    def __init__(self, X, Y, batch_size=32, shuffle=True):
        """
        Simple text dataset for batching.

        Args:
            X (ndarray or list): Input sequences (can be list of lists or array).
            Y (ndarray or list): Target sequences.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle data at the start of each epoch.
        """
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(X)
        self.indices = np.arange(self.num_samples)

    def __iter__(self):
        """
        Create an iterator over the dataset.

        Shuffles the data if specified, and yields batches.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)

        for start_idx in range(0, self.num_samples, self.batch_size):
            batch_indices = self.indices[start_idx:start_idx + self.batch_size]
            yield self.X[batch_indices], self.Y[batch_indices]

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        return int(np.ceil(self.num_samples / self.batch_size))
