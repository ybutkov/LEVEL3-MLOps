"""Serving layer: a standalone FastAPI service for the production model.

Decoupled from the Dagster pipeline — it shares only config and feature
definitions and talks to the MLflow registry. Run with
``uvicorn bike_rental.serving.app:app``.
"""
