import numpy as np


class BiometricMatcher:

    def __init__(self, threshold=0.75):
        # ArcFace decision boundary threshold
        self.threshold = threshold

    def compute_similarity(self, vector_a, vector_b):
        """Calculates Cosine Similarity between two 512-D vectors."""
        # Dot product of normalized vectors = Cosine Similarity
        cosine_score = float(np.dot(vector_a, vector_b))
        match_percentage = round(cosine_score * 100, 2)
        is_same_animal = cosine_score >= self.threshold

        return {
            "cosine_score": cosine_score,
            "match_percentage": match_percentage,
            "is_matched": is_same_animal,
            "verdict": (
                "SAME ANIMAL DETECTED"
                if is_same_animal
                else "DIFFERENT ANIMAL (REJECTED)"
            ),
        }