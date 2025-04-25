# helper functions for the project
import os
import yaml


def read_yaml(path: str) -> dict:
  with open(path, 'r') as f:
    return yaml.safe_load(f)


def makedirs(path: str):
  os.makedirs(path, exist_ok=True)