# src/evaluation/evaluate_metrics_total.py
# Alias — runs the full multi-task evaluation from evaluate_metrics.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.evaluation.evaluate_metrics import evaluate

if __name__ == '__main__':
    evaluate()
