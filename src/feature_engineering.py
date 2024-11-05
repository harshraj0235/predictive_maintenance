import pandas as pd

def create_features(data):
    """Create additional features based on existing data."""
    # Example: Create a feature for the ratio of RPM to age
    data['rpm_to_age'] = data['rpm'] / (data['age'] + 1)  # Avoid division by zero
    return data