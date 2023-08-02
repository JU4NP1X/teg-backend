import pandas as pd
import torch
from django.db.models.functions import Concat
from django.db.models import Value, CharField, Subquery, OuterRef, Func
from ..models import Categories
from datasets.models import Datasets_English_Translations
import numpy as np
from collections import Counter
from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer


class GroupConcat(Func):
    function = "GROUP_CONCAT"
    template = "%(function)s(%(expressions)s)"


class Data_Processer:
    def __init__(self):
        self.datasets = Datasets_English_Translations.objects.filter(
            dataset__categories__deprecated=False
        ).annotate(
            CONTEXT=Concat(
                "paper_name", Value(": "), "summary", output_field=CharField()
            ),
            CATEGORIES=Subquery(
                Categories.objects.filter(
                    datasets=OuterRef("dataset"), deprecated=False
                )
                .values("id")
                .annotate(ids=GroupConcat("id"))
                .values("ids"),
                output_field=CharField(),
            ),
            RELATED_CATEGORIES=Subquery(
                Categories.objects.filter(
                    datasets=OuterRef("dataset"),
                    deprecated=False,
                    related_categories__deprecated=False,
                )
                .values("related_categories__id")
                .annotate(ids=GroupConcat("related_categories__id"))
                .values("ids"),
                output_field=CharField(),
            ),
            PARENT_CATEGORIES=Subquery(
                Categories.objects.filter(
                    datasets=OuterRef("dataset"),
                    deprecated=False,
                    parent_category__deprecated=False,
                )
                .values("parent_category__id")
                .annotate(ids=GroupConcat("parent_category__id"))
                .values("ids"),
                output_field=CharField(),
            ),
        )
        self.mlb = MultiLabelBinarizer()

    def get_data(self):
        df = pd.DataFrame.from_records(
            self.datasets.values(
                "CATEGORIES", "CONTEXT", "RELATED_CATEGORIES", "PARENT_CATEGORIES"
            )
        )
        df = df.sample(frac=1).reset_index(drop=True)

        # Combine the categories, related thesauri, and parent categories into a single column
        df["CATEGORIES"] = df.apply(
            lambda row: list(
                set(
                    (row["CATEGORIES"].split(",") if row["CATEGORIES"] else [])
                    + (
                        row["RELATED_CATEGORIES"].split(",")
                        if row["RELATED_CATEGORIES"]
                        else []
                    )
                    + (
                        row["PARENT_CATEGORIES"].split(",")
                        if row["PARENT_CATEGORIES"]
                        else []
                    )
                )
            ),
            axis=1,
        )

        # Drop the individual category, related thesauri, and parent categories columns
        df.drop(
            ["RELATED_CATEGORIES", "PARENT_CATEGORIES"],
            axis=1,
            inplace=True,
        )
        df["CATEGORIES"] = df["CATEGORIES"].apply(
            lambda labels: [int(label_id) for label_id in labels if label_id]
        )
        return df

    def get_categories(self, trained=True):
        self.categories = Categories.objects.filter(deprecated=False)
        if trained:
            self.categories = self.categories.exclude(label_index__isnull=True)
        return self.categories.values_list("id", "name")

    def preprocess_data(self):
        df = self.get_data()
        # Make the list of possible results
        self.labels = self.get_categories(False)

        # Extract only the IDs
        label_ids = [label_id for label_id, _ in self.labels]
        self.num_outputs = len(label_ids)

        # Fit the MultiLabelBinarizer using the IDs only
        self.mlb.fit([label_ids])

        df = self.balance_data(df)
        binary_categories = pd.DataFrame(
            self.mlb.transform(df["CATEGORIES"]),
            columns=[label_name for _, label_name in self.labels],
        )
        self.outputs = binary_categories.columns.values.tolist()
        df = pd.concat([df, binary_categories], axis=1)
        self.plot_category_counts(df)

        # Create the weights per category
        category_counts = df[self.outputs].sum().sort_index().fillna(0).replace(0, 1)
        min_category_count = category_counts.min()
        category_weights = 1 / (category_counts / min_category_count)
        pd.set_option("display.float_format", "{:.8f}".format)
        output_indices = category_counts.index.get_indexer(self.outputs)
        ordered_weights = category_weights.iloc[output_indices]
        self.weights = ordered_weights

        # Split the data into training and validation sets
        train_df = df.sample(frac=0.8, random_state=200).reset_index(drop=True)
        val_df = df.drop(train_df.index).reset_index(drop=True)

        std_dev = 1 / fitness_function(df.index.to_numpy(), df, label_ids)
        print("Data Size: ", len(df))
        print("Std dev: ", std_dev)
        return train_df, val_df

    # Really better solution, but slower
    def balance_data(self, df):
        # First, flatten the list of categories and count the frequency of each one
        categories = [
            label for sublist in df["CATEGORIES"].tolist() for label in sublist
        ]
        counter = Counter(categories)

        # Find the minimum category to balance
        min_category = min(counter, key=counter.get)

        # Create an empty DataFrame to store the balanced data
        balanced_df = pd.DataFrame(columns=df.columns)

        # Iterate over each row in the original DataFrame
        for _, row in df.iterrows():
            # If the row contains the minimum category, add it to the balanced DataFrame
            if min_category in row["CATEGORIES"]:
                balanced_df = pd.concat([balanced_df, row.to_frame().T])
            else:
                worst_proportion_in_row = float("inf")
                tolerance = 200
                for category in row["CATEGORIES"]:
                    category_count = counter[category]

                    if category_count < worst_proportion_in_row:
                        worst_proportion_in_row = category_count

                # If not, add the row to the balanced DataFrame with a probability equal to the proportion of the minimum category
                if (worst_proportion_in_row <= tolerance) or (
                    (np.random.rand() * (counter[min_category] + tolerance))
                    > (np.random.rand() * worst_proportion_in_row)
                ):
                    balanced_df = pd.concat([balanced_df, row.to_frame().T])

        # Shuffle the balanced DataFrame to ensure the data is randomly distributed
        balanced_df = shuffle(balanced_df)

        # Reset the indices of the balanced DataFrame
        balanced_df.reset_index(drop=True, inplace=True)
        return balanced_df

    def plot_category_counts(self, df):
        # Count the frequency of each category in the DataFrame
        fig = df[self.outputs].sum().plot.bar(figsize=(25, 8), width=1.0).get_figure()
        fig.savefig("category_counts.png")


# Define the fitness function
def fitness_function(individual, df, categories_ids):
    # Calculate the total sum of each category in the individual
    category_sums = [0] * len(categories_ids)

    for dataset_id in individual:
        dataset_categories_ids = df.loc[dataset_id, "CATEGORIES"]
        for category_id in dataset_categories_ids:
            category_index = categories_ids.index(category_id)
            category_sums[category_index] += 1

    # Calculate the standard deviation of the category sums
    std_dev = np.std(category_sums)

    # Return the inverse of the standard deviation as the fitness value
    return 1 / std_dev
