from flask import Flask, render_template
from forms import InformationForm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from yellowbrick.regressor import ResidualsPlot
from datetime import datetime
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)
app.config['SECRET_KEY'] = "asdaqwawd68448awd6a8w4d6a84wd"
pathName = "C:/Users/Pat/OneDrive/Documents/Desktop/insuranceApp/"


@app.route('/')
def welcome():  # put application's code here
    return render_template("index.html")


@app.route('/applicantInformation', methods=["POST", "GET"])
def getApplicantInformation():
    form = InformationForm()
    if form.validate_on_submit():

        userAge = form.userAge.data
        userSex = form.userSex.data
        userBMI = form.userBMI.data
        userSmoker = form.userSmoker.data

        if userSmoker == "yes":
            userSmoker = 1
        else:
            userSmoker = 0

        if userSex == "male":
            userSex = 1
        else:
            userSex = 0

        # did numerous models and had the highest R2 score with region and children removed
        df = pd.read_csv("static/data/insurance_dataset.csv", sep=",")
        df.drop(["region", "children"], axis=1, inplace=True)

        # used a box plot and a count plot to visualize the smoking group and used a distribution plot
        # to show age distribution
        plt.figure(figsize=(15, 8))

        sns.countplot(x=df["smoker"], data=df, palette="Spectral")
        countplotFileName = ("plots/count_plot" + str(datetime.now()).replace("-", "_")
                             .replace(" ", "_").replace(".", "_")
                             .replace(":", "_") + ".png")
        plt.savefig(os.path.join(pathName, "static/" + countplotFileName), bbox_inches="tight", pad_inches=0)
        plt.close()

        sns.distplot(df["age"])
        distplotFileName = ("plots/dist_plot" + str(datetime.now()).replace("-", "_")
                            .replace(" ", "_").replace(".", "_")
                            .replace(":", "_") + ".png")
        plt.savefig(os.path.join(pathName, "static/", distplotFileName), bbox_inches="tight", pad_inches=0)
        plt.close()

        sns.boxplot(x="smoker", y="charges", data=df, palette="Spectral")
        boxplotFileName = ("plots/box_plot" + str(datetime.now()).replace("-", "_")
                           .replace(" ", "_").replace(".", "_")
                           .replace(":", "_") + ".png")
        plt.savefig(os.path.join(pathName, "static/", boxplotFileName), bbox_inches="tight", pad_inches=0)
        plt.close()

        # Data manipulation
        def smoker(x):
            if x == "yes":
                return 1
            else:
                return 0

        def gender(x):
            if x == "male":
                return 1
            else:
                return 0

        df["smoker"] = df["smoker"].apply(smoker)
        df["sex"] = df["sex"].apply(gender)

        # Model building
        X = df.drop(["charges"], axis=1)
        y = df["charges"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        user = [userAge, userSex, userBMI, userSmoker]
        userDict = {"age": user[0], "sex": user[1], "bmi": user[2], "smoker": user[3]}
        userDF = pd.DataFrame(userDict, index=[0])
        estimatedCharges = model.predict(userDF)

        def eval_metrics(actual, pred):
            MAE = mean_absolute_error(actual, pred)
            MSE = mean_squared_error(actual, pred)
            RMSE = np.sqrt(MSE)
            R2SCR = r2_score(actual, pred)
            return [R2SCR, MAE, MSE, RMSE]

        modelScores = eval_metrics(y_test, y_pred)
        residualPlot = ResidualsPlot(model)
        residualPlot.fit(X_train, y_train)
        residualPlot.score(X_test, y_test)
        residplotFileName = ("plots/residplot" + str(datetime.now()).replace("-", "_")
                             .replace(" ", "_").replace(".", "_")
                             .replace(":", "_") + ".png")
        plt.savefig(os.path.join(pathName, "static", residplotFileName), bbox_inches="tight", pad_inches=0)
        plt.close()

        plots = [countplotFileName, distplotFileName, boxplotFileName, residplotFileName]

        return render_template("viewResults.html", user=user, plots=plots, modelScores=modelScores,
                               estimatedCharges=estimatedCharges)

    else:
        return render_template("applicantInfo.html", form=form)


if __name__ == '__main__':
    app.run()
