# Machine Learning Challenge
### Luis Patiño
---

## Overview

This repository contains my solution to the ABI machine learning challenge focused on deploying a model prediction API. The API is built with FastAPI, adhering to object-oriented programming principles, and is designed to support individual and batch predictions. The deployment environment is Dockerized for portability and scalability.

## Features
- API Framework: FastAPI
- Database Integration: SQLAlchemy for flexible database operations
- Deployment: Docker and Docker Compose for containerized deployment
- Testing: Integrated with Sonar Cloud for code quality and testing
- Prediction: Supports both individual and batch model predictions

## Project Structure
```
├── .github                      # GitHub configuration files (hidden directory)
│   └── build.yml                # Configuration for Sonar Cloud
├── .venv                        # Python virtual environment directory (hidden directory)
├── app                          # Main application directory
│   ├── data                     # Directory for storing data files
│   │   └── train.csv            # Training dataset in CSV format
│   ├── models                   # Directory for model files
│   │   └── model.pkl            # Serialized machine learning model file (pickle format)
│   ├── test                     # Directory for test files
│   │   ├── test_main.py         # Tests for the main API endpoints
│   │   ├── test_model.py        # Tests for the machine learning model
│   │   └── __init__.py          # Init file for the test module
│   ├── database.py              # Database configuration and connection handling
│   ├── main.py                  # Entry point for the FastAPI application
│   ├── model.py                 # Machine learning model handling logic
│   ├── schemas.py               # Pydantic schemas for data validation
│   └── train_model.py           # Script for training the machine learning model
├── docker-compose.yml           # Docker Compose configuration file for orchestrating services
├── Dockerfile                   # Docker configuration file for building the API container
├── requirements.txt             # Python dependencies file
└── sonar-project.properties     # Sonar Cloud configuration for code quality checks
```

## Architecture
![image](https://github.com/user-attachments/assets/03273d9c-34a7-44da-a882-a78fbf0a4aac)


Thanks to the organizers for the opportunity to demonstrate skills in deploying machine learning models with modern tools and best practices.

