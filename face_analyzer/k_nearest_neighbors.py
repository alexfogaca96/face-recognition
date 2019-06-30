from math import pow, sqrt
from face_analyzer.enumerators.match_type import MatchType


class KNN:
    def __init__(self, k_neighbors, data):
        self.k_neighbors = k_neighbors
        self.network = KNN.process_image_differences(data)

    def do_they_match(self, face_one, face_two):
        if len(self.network) == 0:
            raise Exception("There's no data on the KNN's network.")
        desired_difference = KNN.process_image_difference(face_one, face_two)
        differences = []
        print("calculating distances...")
        for difference in range(len(self.network)):
            distance = KNN.euclidean_distance(desired_difference, self.network[difference][0])
            differences.append([distance, self.network[difference][1]])
        differences.sort(key=lambda x: x[0])
        matches = 0
        for neighbor in range(self.k_neighbors):
            if differences[neighbor][1] == MatchType.MATCH:
                matches += 1
        if matches == (self.k_neighbors / 2):
            return differences[0][1]
        if matches > (self.k_neighbors / 2):
            return MatchType.MATCH
        return MatchType.MISMATCH

    @staticmethod
    def euclidean_distance(point_one, point_two):
        if len(point_one) != len(point_two):
            raise Exception("There's no distance between points with different dimensions.")
        distance = 0
        for dimension in range(len(point_one)):
            distance += pow((point_one[dimension] - point_two[dimension]), 2)
        return sqrt(distance)

    @staticmethod
    def process_image_differences(data):
        """
        Builds a network of the differences between every pair of image given.

        :param data: list of data_item pairs
        :return: network containing the difference of all images labeled by match_type
        """
        network = []
        for image_pair in data:
            difference = KNN.process_image_difference(image_pair.face_one, image_pair.face_two)
            network.append([difference, image_pair.match_type])
        return network

    @staticmethod
    def process_image_difference(image_one, image_two):
        """
        Process the quadratic difference between images pre-processed by the HOG descriptor
        (or any other one that results on a list full of numbers)

        :param image_one: first comparison
        :param image_two: second comparison
        :return: the quadratic difference between the two (feature by feature)
        """
        difference = []
        for feature in range(len(image_one)):
            raw_difference = image_one[feature] - image_two[feature]
            difference.append(raw_difference * raw_difference)
        return difference
