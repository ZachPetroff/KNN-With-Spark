from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, mean, stddev

# read in data
spark = SparkSession.builder.appName("IRIS").getOrCreate()
df = spark.read.text("iris.data")

'''
    For some reason, the read function
    did not separate the columns, so this
    was done manually.
'''
df = df.withColumn("Sepal Length", split(col("value"), 
             ",").getItem(0)).withColumn("Sepal Width", split(col("value"),
             ",").getItem(1)).withColumn("Petal Length", split(col("value"),
             ",").getItem(2)).withColumn("Petal Width", split(col("value"),
             ",").getItem(3)).withColumn("class", split(col("value"), 
             ",").getItem(4))
                   
# drop the left over column
df = df.drop("value")

# show resulting DataFrame
print("Display DataFrame . . . ")
df.show()

# normalize function
def norm(df, cols):
    aggExpr = []
    for c in cols:
        aggExpr.append(mean(df[c]).alias(c))
    averages = df.agg(*aggExpr).collect()[0]
    selectExpr = []
    for c in cols:
        selectExpr.append(df[c] - averages[c])
    for exp in range(len(selectExpr)):
        selectExpr[exp] = selectExpr[exp].alias(cols[exp])
    selectExpr.append(df["class"])
    df = df.select(selectExpr)
    aggExpr = []
    for c in cols:
        aggExpr.append(stddev(df[c]).alias(c))
    stddevs = df.agg(*aggExpr).collect()[0]
    selectExpr = []
    for c in cols:
        selectExpr.append(df[c] / stddevs[c])
    for exp in range(len(selectExpr)):
        selectExpr[exp] = selectExpr[exp].alias(cols[exp])
    selectExpr.append(df["class"])
    return df.select(selectExpr)

# normalize data and display
norm_df = norm(df, ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
print("Display Normalized DataFrame . . . ")
norm_df.show()

# split into training, validation, and testing sets
train_df = norm_df.sample(.6, 123)
val_df = norm_df.sample(.2, 456)
test_df = norm_df.sample(.2, 789)

# show train_df
print("Display Training DataFrame . . . ")
train_df.show()

# Function For Finding Euclidean Distance Between Two Rows
def euclidean_dist(row1, row2):
    dist = 0
    for entry in range(len(row1)-1):
        if row1[entry] == None or row2[entry] == None:
            return 1e10
        dist += (float(row1[entry]) - float(row2[entry])) ** 2 
    return dist

# show example distance
print("Example Distance Output: ", euclidean_dist(train_df.collect()[0], train_df.collect()[1]))

# define knn algorithm
def KNN(k, observation, train_data=train_df):
    dists = []
    classes = []
    for i in range(train_data.count()):
        dist = euclidean_dist(train_data.collect()[i], observation)
        if len(dists) <= k:
            dists.append(dist)
            classes.append(train_data.collect()[i][len(observation)-1])
        else:
            for d in range(len(dists)):
                if dist < dists[d]:
                    del dists[d] 
                    del classes[d]
                    dists.append(dist)
                    classes.append(train_data.collect()[i][len(observation)-1])
    poss_classes = []
    class_counts = []
    for c in classes:
        if c not in poss_classes:
            poss_classes.append(c)
            class_counts.append(1)
        else:
            class_idx = poss_classes.index(c)
            class_counts[class_idx] += 1
    max_class = poss_classes[0]
    max_count = class_counts[0]
    for i in range(len(poss_classes)):
        if max_count < class_counts[i]:
            max_class = poss_classes[i]
            max_count = class_counts[i]
    return max_class

Ks = [5, 10, 15, 20, 25]

# get accuracy of classifier
def accuracy(train_data, test_data, k):
    acc = 0
    preds = []
    actuals = []
    for obs in range(test_data.count()):
        pred = KNN(k, test_data.collect()[obs], train_data)
        if pred == test_data.collect()[obs][len(test_data.collect()[0])-1]:
            acc += 1
        preds.append(pred)
        actuals.append(test_data.collect()[obs][len(test_data.collect()[0])-1])
    return acc / test_data.count(), preds, actuals

iris_accs = []
for k in range(len(Ks)):
    acc, _, _ = accuracy(train_df, val_df, Ks[k])
    iris_accs.append(acc)

# Read in fertility data
spark = SparkSession.builder.appName("FERTILITY").getOrCreate()
df2 = spark.read.csv("fertility_Diagnosis.txt")

# rename columns
df2 = df2.withColumnRenamed("_c0", "Season") \
         .withColumnRenamed("_c1", "Age") \
         .withColumnRenamed("_c2", "Childish Diseases") \
         .withColumnRenamed("_c3", "Accident") \
         .withColumnRenamed("_c4", "Surgery") \
         .withColumnRenamed("_c5", "Fever") \
         .withColumnRenamed("_c6", "Alcohol") \
         .withColumnRenamed("_c7", "Smoking") \
         .withColumnRenamed("_c8", "Sitting") \
         .withColumnRenamed("_c9", "Output")

# show DataFrame
df2.show()

# Values are already normalized

# Split into training, testing, and validation sets
train_df2 = df2.sample(.6, 123)
val_df2 = df2.sample(.2, 456)
test_df2 = df2.sample(.2, 789)

fert_accs = []
for k in range(len(Ks)):
    acc, _, _ = accuracy(train_df2, val_df2, Ks[k])
    fert_accs.append(acc)

for acc in range(len(iris_accs)):
    print("Iris accuracy is ", iris_accs[acc], " when k = ", Ks[acc])

for acc in range(len(fert_accs)):
    print("Fertility accuracy is ", fert_accs[acc], " when k = ", Ks[acc])
    
'''
    The optimal K for both datasets is 5. For the Iris data,
    all values of K, expect for 25, achieve an accuracy of 100%.
    However, because 5 is lowest of the K's, it's the most efficient.
    For the fertility dataset, K = 5 achieves the highest accuracy (about 94%)
'''

test_acc, test_preds, test_actuals = accuracy(train_df, test_df, 5)
print("Iris Test Predictions: ", test_preds)
print("Iris Test Actual Values: ", test_actuals)
print("Iris Test Accuracy: ", test_acc)

test_acc, test_preds, test_actuals = accuracy(train_df2, test_df2, 5)
print("Fertility Test Predictions: ", test_preds)
print("Fertility Test Actual Values: ", test_actuals)
print("Fertility Test Accuracy: ", test_acc)
