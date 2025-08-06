import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import lightgbm as lgb
from xgboost import XGBRFRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
print("\n=== 快速查看 ===")
print(f"当前工作目录: {os.getcwd()}")
print(f"脚本位置: {Path(__file__).resolve()}")
print(f"Python可执行文件: {sys.executable}")
print("所有机器学习依赖导入成功！")
