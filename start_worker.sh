#!/bin/bash

prefect deploy -p "ML_Uebung2" -n "ML_Uebung2_aufgabe1" ./flows/aufgabe1.py:train_model_flow
prefect deploy -p "ML_Uebung2" -n "ML_Uebung2_aufgabe1_grid_search" ./flows/aufgabe1.py:grid_search
prefect deploy -p "ML_Uebung2" -n "ML_Uebung2_aufgabe1_cross_validation" ./flows/aufgabe2.py:cross_validation
prefect deploy -p "ML_Uebung2" -n "ML_Uebung2_aufgabe3_nested_cross_validation" ./flows/aufgabe3.py:nested_cross_validation


prefect worker start --pool "ML_Uebung2"