# Predictive Maintenance for Automotive OEMs

This project implements a prototype for predictive maintenance in the automotive industry using AI/ML techniques.

## Setup

1. Clone this repository
2. Create a virtual environment:

# Predictive Maintenance Prototype

## Overview
This project aims to develop a predictive maintenance prototype using machine learning techniques. The goal is to predict potential failures in vehicles based on various sensor data and operational parameters. This prototype is designed to demonstrate the capabilities of AI and ML in enhancing maintenance strategies for Original Equipment Manufacturers (OEMs) like Volkswagen.

## Features
The dataset includes the following features:

- **Sensor Data:**
  - `temperature`: Engine temperature (°C)
  - `vibration`: Vibration levels (g)
  - `pressure`: Engine oil pressure (psi)
  - `age`: Age of the vehicle (days)
  - `rpm`: Engine speed (RPM)
  - `fuel_consumption`: Fuel consumption (L/100km)
  - `oil_temperature`: Oil temperature (°C)
  - `battery_voltage`: Battery voltage (V)
  - `brake_pad_wear`: Brake pad thickness (mm)
  - `tire_pressure`: Tire pressure (psi)

- **Environmental Factors:**
  - `ambient_temperature`: External temperature (°C)
  - `road_condition`: Quality of the road (categorical: smooth, rough, potholes)

- **Vehicle Usage Patterns:**
  - `driving_style`: Driving behavior (categorical: aggressive, normal, gentle)
  - `mileage`: Total distance traveled (km)
  - `time_since_last_maintenance`: Days since last maintenance
  - `component_age`: Age of critical components (years)

- **Target Variable:**
  - `failure`: Binary indicator of failure (1 = failure, 0 = no failure)

## Installation
To run this project, you need to have Python installed along with the required libraries. You can install the necessary libraries using pip:

```bash
pip install -r requirements.txt