"""
    Rarely used, but might be useful before the projects end.
"""

def new_mean_value_test(subjects, featureIndex):
    mean = 0
    for s in subjects:
        mean += int(s.features[featureIndex])

    mean /= len(subjects[0].features)

    def test(subject):
        return subject.features[featureIndex] < mean

    return test