def select_features(df):
    X = df.drop("Machine failure", axis=1)
    y = df["Machine failure"]
    return X, y