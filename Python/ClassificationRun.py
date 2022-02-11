import pandas as pd
import Classification
import math
#Line 1: Reads CSV, 2 creates a data frame, 3 Shuffles rows
file = pd.read_csv("wdbc.csv")
df = pd.DataFrame(file)
df = df.sample(frac=1)

length = len(df.index) #568
length = math.floor(length*(3/4)) #426

train = df[:length] #0-426
train_length = len(train.index)
print("Total Train size: ",train_length)
#print(train)

test = df[train_length:len(df.index)] #426-568 (142)
test_length = len(test.index)
print("Total Test size: ",test_length)
#print(test)

if __name__ == "__main__":
    Classification.setData(train)
    Classification.run()

    Classification.setData(test)
    Classification.run()