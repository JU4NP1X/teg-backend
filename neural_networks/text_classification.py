import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from django.db.models import Func
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertModel
from collections import Counter
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class GroupConcat(Func):
    function = "GROUP_CONCAT"
    template = "%(function)s(%(expressions)s)"


class BERTClass(torch.nn.Module):
    def __init__(self, size):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained(
            "bert-base-uncased", return_dict=True
        )
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, size)

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output


class CustomDataset:
    def __init__(self, df, outputs, max_len):
        self.tokenizer = tokenizer
        self.df = df
        self.max_len = max_len
        self.title = df["CONTEXT"]
        self.targets = self.df[outputs].values

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        title = str(self.title[idx])
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            max_length=self.max_len,
            padding="max_length",
            add_special_tokens=True,
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "token_type_ids": inputs["token_type_ids"].flatten(),
            "targets": torch.FloatTensor(self.targets[idx]),
        }


class TextClassifier:
    def __init__(self):
        self.categories = None
        self.num_outputs = None
        self.outputs = []
        self.num_inputs = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.mlb = MultiLabelBinarizer()
        self.model_path = "/home/juan/projects/teg/backend"
        self.max_len = 256
        self.train_batch_size = 3
        self.valid_batch_size = 1
        self.learning_rate = 1e-05
        self.epochs = 2

    def get_data(self):
        pass

    def get_categories(self):
        pass

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
                tolerance = 20
                for category in row["CATEGORIES"]:
                    category_count = counter[category]
                    close_to_mode = False

                    if category_count < worst_proportion_in_row:
                        worst_proportion_in_row = category_count
                    if category_count > 500:
                        tolerance = 1 / category_count
                        close_to_mode = True

                # If not, add the row to the balanced DataFrame with a probability equal to the proportion of the minimum category
                if (worst_proportion_in_row <= tolerance and not close_to_mode) or (
                    (np.random.rand() * (counter[min_category] + tolerance))
                    > (np.random.rand() * worst_proportion_in_row)
                ):
                    balanced_df = pd.concat([balanced_df, row.to_frame().T])

        # Shuffle the balanced DataFrame to ensure the data is randomly distributed
        balanced_df = shuffle(balanced_df)

        # Reset the indices of the balanced DataFrame
        balanced_df.reset_index(drop=True, inplace=True)
        return balanced_df

    def balance_data2(self, df, categories_ids):

        # Create the fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Initialize the toolbox
        toolbox = base.Toolbox()

        # Register the necessary functions
        toolbox.register("attribute", random.choice, df.index.to_numpy())
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attribute,
            n=3000,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Register the genetic operators
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register(
            "mutate", tools.mutUniformInt, low=0, up=df.shape[0] - 1, indpb=0.05
        )
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", lambda ind: fitness_function(ind, df, categories_ids))

        # Set the parameters
        population_size = 50
        num_generations = 10
        offspring_size = 5 * population_size  # Increase the number of offspring

        # Create the initial population
        population = toolbox.population(n=population_size)

        # Evaluate the fitness of the initial population
        fitness_values = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitness_values):
            ind.fitness.values = (fit,)

        # Perform the evolution
        for generation in range(num_generations):
            print("Generation:", generation + 1)

            # Select the next generation individuals
            offspring = toolbox.select(population, offspring_size)

            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the fitness of the offspring
            invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
            fitness_values = map(toolbox.evaluate, invalid_individuals)
            for ind, fit in zip(invalid_individuals, fitness_values):
                ind.fitness.values = (fit,)

            # Replace the population with the offspring
            population[:] = offspring

            # Print the best fitness value in the current generation
            best_fitness = np.max([ind.fitness.values[0] for ind in population])
            print("Best Fitness:", best_fitness)

        # Select the best individual from the final population
        best_combination = tools.selBest(population, k=1)[0]

        # df a DataFrame to store the balanced data
        df = df.iloc[best_combination]
        df = df.reset_index(drop=True)
        return df

    def plot_category_counts(self, df):
        # Count the frequency of each category in the DataFrame
        category_counts = df[self.outputs].sum().sort_index()

        # Create a figure and a subplot to show the graph
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the number of occurrences for each category
        ax.bar(category_counts.index, category_counts.values)
        ax.set_title("Category Counts")
        ax.set_ylabel("Number of Occurrences")
        # Remove the x-axis labels
        ax.set_xticklabels([])
        # Adjust the spacing between the subplots
        plt.tight_layout()

        # Save the graph as an image in PNG format
        plt.savefig("category_counts.png")

    def preprocess_data(self):
        df = self.get_data()
        # Make the list of possible results
        label_ids_and_names = self.get_categories()

        # Extract only the IDs
        label_ids = [label_id for label_id, _ in label_ids_and_names]
        self.num_outputs = len(label_ids)
        # Fit the MultiLabelBinarizer using the IDs only
        self.mlb.fit([label_ids])
        
        df = self.balance_data(df)
        #df = self.balance_data2(df, label_ids)

        binary_categories = pd.DataFrame(
            self.mlb.transform(df["CATEGORIES"]),
            columns=[label_name for _, label_name in label_ids_and_names],
        )
        self.outputs = binary_categories.columns.values.tolist()
        df = pd.concat([df, binary_categories], axis=1)
        self.plot_category_counts(df)

        # Split the data into training, validation, and test sets
        train_df = df.sample(frac=0.8, random_state=200).reset_index(drop=True)
        val_df = df.drop(train_df.index).reset_index(drop=True)

        std_dev = 1/ fitness_function(df.index.to_numpy(), df, label_ids)
        print("Data Size: ", len(df))
        print("Std dev: ", std_dev)
        print(train_df.head())
        print(val_df.head())
        return train_df, val_df

    def create_model(self):
        # Define the model architecture
        self.model = BERTClass(self.num_outputs)
        self.model.to(self.device)

    def loss_fn(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    def train_model(self, training_loader, validation_loader):
        val_targets = []
        val_outputs = []

        # Initialize tracker for minimum validation loss
        valid_loss_min = np.Inf
        for epoch in range(1, self.epochs + 1):
            train_loss = 0
            valid_loss = 0

            self.model.train()

            print("Training Epoch {}".format(epoch))

            for batch_idx, data in enumerate(training_loader, 0):
                ids = data["input_ids"].to(self.device, dtype=torch.long)
                mask = data["attention_mask"].to(self.device, dtype=torch.long)
                token_type_ids = data["token_type_ids"].to(
                    self.device, dtype=torch.long
                )
                targets = data["targets"].to(self.device, dtype=torch.float)

                outputs = self.model(ids, mask, token_type_ids)

                self.optimizer.zero_grad()
                loss = self.loss_fn(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss = train_loss + (
                    (1 / (batch_idx + 1)) * (loss.item() - train_loss)
                )

            ######################
            # Validate the model #
            ######################
            print("Validation Epoch {}".format(epoch))

            self.model.eval()

            with torch.no_grad():
                for batch_idx, data in enumerate(validation_loader, 0):
                    ids = data["input_ids"].to(self.device, dtype=torch.long)
                    mask = data["attention_mask"].to(self.device, dtype=torch.long)
                    token_type_ids = data["token_type_ids"].to(
                        self.device, dtype=torch.long
                    )
                    targets = data["targets"].to(self.device, dtype=torch.float)
                    outputs = self.model(ids, mask, token_type_ids)

                    loss = self.loss_fn(outputs, targets)
                    valid_loss = valid_loss + (
                        (1 / (batch_idx + 1)) * (loss.item() - valid_loss)
                    )
                    val_targets.extend(targets.cpu().detach().numpy().tolist())
                    val_outputs.extend(
                        torch.sigmoid(outputs).cpu().detach().numpy().tolist()
                    )

            # Calculate average losses
            train_loss = train_loss / len(training_loader)
            valid_loss = valid_loss / len(validation_loader)

            # Print training/validation statistics
            print(
                "Epoch: {} \tAverage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}".format(
                    epoch, train_loss, valid_loss
                )
            )

            # Create checkpoint variable and add important data
            checkpoint = {
                "epoch": epoch + 1,
                "valid_loss_min": valid_loss,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            # Save checkpoint
            self.save_ckp(checkpoint, False)

            ## TODO: save the model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print(
                    "Validation loss decreased ({:.6f} --> {:.6f}). Saving model...".format(
                        valid_loss_min, valid_loss
                    )
                )
                # Save checkpoint as best model
                self.save_ckp(checkpoint, True)
                valid_loss_min = valid_loss

            print("############# Epoch {} Done #############\n".format(epoch))

    def load_ckp(self, checkpoint_fpath):
        # Load checkpoint
        checkpoint = torch.load(checkpoint_fpath)
        # Initialize state_dict from checkpoint to model
        self.model.load_state_dict(checkpoint["state_dict"])
        # Initialize optimizer from checkpoint to optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        # Initialize valid_loss_min from checkpoint to valid_loss_min
        valid_loss_min = checkpoint["valid_loss_min"]
        # Return model, optimizer, epoch value, min validation loss
        return checkpoint["epoch"], valid_loss_min

    def save_ckp(self, state, is_best):
        pass

    def retrain_model(self, model_path, new_num_outputs):
        self.model = TextClassifier(self.num_inputs, new_num_outputs)
        self.model.load_state_dict(torch.load(model_path))

        # Here you would need to define your new training loop

    def train(self):
        train_df, val_df = self.preprocess_data()
        train_data = CustomDataset(train_df, self.outputs, self.max_len)
        val_data = CustomDataset(val_df, self.outputs, self.max_len)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.create_model()
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.learning_rate
        )
        train_data_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_data_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=self.valid_batch_size,
            shuffle=True,
            num_workers=0,
        )

        self.train_model(train_data_loader, val_data_loader)

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