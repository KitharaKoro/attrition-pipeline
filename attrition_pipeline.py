import luigi
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
import pickle
import sqlite3


class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict attrition.


        Outputs a pickled dictionary including the model, label binarizer/encoder, the logistic regression coefficients,
        coefficient names, and training set accuracy.
    """
    output_file = luigi.Parameter(default='model.pkl')

    def run(self):
        with sqlite3.connect('nudge_test.db') as conn:
            df = pd.read_sql_query('Select * From user_tp_attrition', conn)

        logistic_model = LogisticRegression(multi_class="ovr", solver='newton-cg')
        binarizer = LabelBinarizer()

        logistic_model.fit(
            pd.concat((pd.DataFrame(binarizer.fit_transform(df['department'])), df["nb_of_sessions"]), axis=1)
            , df["attrition"])

        training_set_accuracy = sum(logistic_model.predict(pd.concat((pd.DataFrame(binarizer.fit_transform(
                                    df['department'])), df["nb_of_sessions"]), axis=1)) == df["attrition"]) / df.shape[0]

        output = {"model": logistic_model, "binarizer": binarizer,
                  "coefficients": [logistic_model.intercept_[0], *logistic_model.coef_.tolist()[0]],
                  "coefficient_names": ["intercept", *binarizer.classes_, "nb_of_sessions"],
                  "training_set_accuracy": training_set_accuracy}
        with open(self.output_file, 'wb') as file:
            pickle.dump(output, file, pickle.HIGHEST_PROTOCOL)

    def output(self):
        return luigi.LocalTarget(self.output_file)


class SomeOtherTask(luigi.Task):
    """ Any task we would like to accomplish using the trained model.

        Possible examples: predict attrition on new data, calculate accuracy on new data, sort individuals, departments,
        or some other hierarchy based on probability of attrition
    """
    output_file = luigi.Parameter(default='some_file.csv')

    def requires(self):
        return TrainModelTask()

    def run(self):
        with open(self.input().path, 'rb') as file:
            output = pickle.load(file)
            model = output['model']
            binarizer = output['binarizer']
            coefficients = output['coefficients']
            coefficient_names = output['coefficient_names']
            training_set_accuracy = output['training_set_accuracy']

    def output(self):
        pass


if __name__ == "__main__":
    luigi.run()
