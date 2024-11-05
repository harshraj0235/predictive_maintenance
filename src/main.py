import logging
from src.model import generate_data, train_model
from src.visualization import plot_feature_importance

logging.basicConfig(level=logging.INFO)

def main():
    logging.info("Generating simulated data...")
    simulated_data = generate_data(10000)  # Generate 10,000 samples

    logging.info("Training model...")
    model, X_train, y_train = train_model(simulated_data)

    logging.info("Visualizing feature importance...")
    plot_feature_importance(model, X_train, y_train)

if __name__ == "__main__":
    main()