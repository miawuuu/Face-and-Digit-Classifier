"""
util.py

A simple Counter class for the CS188 Perceptron and related projects.
Missing keys return 0.0. Supports addition, subtraction, dot-product,
and convenience methods like argMax and sortedKeys.
"""

class Counter(dict):
    """A counter keeps counts of keys (internally a dict).
    Provides default 0.0 for missing keys."""
    def __getitem__(self, key):
        return dict.get(self, key, 0.0)

    def __setitem__(self, key, value):
        super().__setitem__(key, float(value))

    def copy(self):
        """Return a shallow copy of the counter."""
        return Counter(dict.copy(self))

    def __add__(self, other):
        """Add two counters: result[k] = self[k] + other[k]"""
        result = Counter()
        for k in set(self.keys()) | set(other.keys()):
            result[k] = self[k] + other[k]
        return result

    def __iadd__(self, other):
        """Increment this counter by another: self[k] += other[k]"""
        for k, v in other.items():
            self[k] = self[k] + v
        return self

    def __sub__(self, other):
        """Subtract two counters: result[k] = self[k] - other[k]"""
        result = Counter()
        for k in set(self.keys()) | set(other.keys()):
            result[k] = self[k] - other[k]
        return result

    def __isub__(self, other):
        """Decrement this counter by another: self[k] -= other[k]"""
        for k, v in other.items():
            self[k] = self[k] - v
        return self

    def __mul__(self, other):  # dot product or scalar multiplication
        """
        If other is a Counter, returns the dot product.
        If other is a number, returns a new Counter with all values scaled.
        """
        if isinstance(other, Counter):
            # dot product
            return sum(self[k] * other[k] for k in self.keys() if k in other)
        elif isinstance(other, (int, float)):
            result = Counter()
            for k in self.keys():
                result[k] = self[k] * other
            return result
        else:
            return NotImplemented

    def argMax(self):
        """Return the key with the highest value."""
        if not self:
            return None
        best_key = max(self.keys(), key=lambda k: self[k])
        return best_key

    def sortedKeys(self):
        """Return keys sorted by their values in descending order."""
        return sorted(self.keys(), key=lambda k: self[k], reverse=True)

# Alias
# util.Counter = Counter  # optionally, if imported differently
