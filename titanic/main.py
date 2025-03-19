from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import washData

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

Xtrain, Xtest, ytrain = washData.getData()

model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0)
model.fit(Xtrain, ytrain)
predictions = model.predict(Xtest)
output = pd.DataFrame(
    {"PassengerId": Xtest.index + Xtrain.shape[0] + 1, "Survived": predictions}
)
output.to_csv("submission.csv", index=False)
