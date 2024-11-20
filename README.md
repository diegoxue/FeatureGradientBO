# FeatureGradientBO

Codes related with "Leveraging Feature Gradient for Efficient Acquisition Function Maximization in Material Composition Design"

## Installation

Clone the repository
```
git clone https://github.com/wsxyh107165243/FeatureGradientBO.git
```
Navigate into the project directory
```
cd FeatureGradientBO
```

## Requirements
- Python 3.10.11
- other requirements are listed in `requirements.txt`, install via pip
```
pip install -r requirements.txt
```

## Usage

```
cd SMA_C4
```

For enumeration-based BO
```
python bo_botorch_enumerate.py
```

For gradient-based BO
```
python bo_botorch_grad_opt.py
```