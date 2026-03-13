"""
Logistic regression is used for binary classification problems like:
spam detection
fraud detection
market direction prediction

Idea: y = mx + c but we need probabilities btwn 0 and 1 so we pass sidmoid function where z = mx + c

if p >= 0.5 → class = 1
else → class = 0

updating gradients:
dw = (1/n) * Xᵀ(p - y)
db = (1/n) * Σ(p - y)

"""
